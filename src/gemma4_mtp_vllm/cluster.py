from __future__ import annotations

from dataclasses import dataclass
from importlib import resources
import hashlib
import json
from pathlib import Path
import shlex
from typing import Any, Iterable

import yaml

from gemma4_mtp_vllm.profiles import ModelProfile

DEFAULT_CLUSTER_TOPOLOGIES_PATH = (
    Path(__file__).resolve().parents[2] / "config" / "cluster_topologies.example.yaml"
)
DEFAULT_TRANSPORT_PROFILE = "socket"
DEFAULT_FABRIC_IFACE = "fabric0"
DEFAULT_FABRIC_CIDR = "198.51.100.0/24"
DEFAULT_CLUSTER_RUN_ROOT = "artifacts/cluster-runs"
DEFAULT_CLUSTER_SSH_OPTIONS = (
    "-n",
    "-o",
    "BatchMode=yes",
    "-o",
    "ConnectTimeout=8",
    "-o",
    "ConnectionAttempts=1",
)
DEFAULT_CLUSTER_VLLM_RUNTIME_ENV = (
    "VLLM_WORKER_MULTIPROC_METHOD=spawn",
    "SAFETENSORS_FAST_GPU=1",
    "NVIDIA_TF32_OVERRIDE=1",
    "VLLM_LOGGING_LEVEL=INFO",
)
SUPPORTED_LIVE_NODE_COUNTS = {2, 4, 6, 8}
SUPPORTED_TRANSPORT_PROFILES = {"socket", "roce-a"}
CLUSTER_PLAN_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class ClusterNode:
    name: str
    fabric_ip: str
    gpus: int = 1


@dataclass(frozen=True)
class ClusterTopology:
    id: str
    label: str
    nodes: tuple[ClusterNode, ...]

    def select_nodes(self, node_count: int) -> tuple[ClusterNode, ...]:
        if node_count < 2:
            raise ValueError("cluster node_count must be at least 2 nodes")
        if node_count > len(self.nodes):
            raise ValueError(
                f"topology {self.id!r} has only {len(self.nodes)} nodes"
            )
        selected = self.nodes[:node_count]
        for node in selected:
            if not node.fabric_ip:
                raise ValueError(f"node {node.name!r} must define fabric_ip")
            if node.gpus <= 0:
                raise ValueError(f"node {node.name!r} must have a positive GPU count")
        return selected


@dataclass(frozen=True)
class ClusterLaunchCommand:
    target: str
    role: str
    command: str
    rank: int

    def to_dict(self) -> dict[str, object]:
        return {
            "target": self.target,
            "role": self.role,
            "rank": self.rank,
            "command": self.command,
        }


@dataclass(frozen=True)
class ClusterLaunchPlan:
    schema_version: int
    dry_run_only: bool
    runtime_id: str
    profile: str
    topology: dict[str, object]
    node_count: int
    tensor_parallel_size: int
    transport_profile: str
    commands: tuple[ClusterLaunchCommand, ...]
    environment: tuple[str, ...]
    resolved_environment_sha256: str
    resolved_command_sha256: str
    dry_run_fingerprint: str
    expected_live_gates: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "dry_run_only": self.dry_run_only,
            "runtime_id": self.runtime_id,
            "profile": self.profile,
            "topology": self.topology,
            "node_count": self.node_count,
            "tensor_parallel_size": self.tensor_parallel_size,
            "transport_profile": self.transport_profile,
            "commands": [command.to_dict() for command in self.commands],
            "environment": list(self.environment),
            "resolved_environment_sha256": self.resolved_environment_sha256,
            "resolved_command_sha256": self.resolved_command_sha256,
            "dry_run_fingerprint": self.dry_run_fingerprint,
            "expected_live_gates": list(self.expected_live_gates),
        }

    def render_shell(self) -> str:
        lines = [
            (
                f"# dry_run_only=true runtime_id={self.runtime_id} "
                f"topology={self.topology['id']} transport={self.transport_profile}"
            ),
            f"# dry_run_fingerprint={self.dry_run_fingerprint}",
        ]
        for command in self.commands:
            lines.append(
                f"# role={command.role} target={command.target} rank={command.rank}"
            )
            lines.append(command.command)
        return "\n".join(lines) + "\n"


@dataclass(frozen=True)
class ClusterExecutionCommand:
    target: str
    role: str
    rank: int
    ssh_host: str
    argv: tuple[str, ...]
    remote_command: str
    background: bool = False
    log_path: str | None = None

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "target": self.target,
            "role": self.role,
            "rank": self.rank,
            "ssh_host": self.ssh_host,
            "argv": list(self.argv),
            "remote_command": self.remote_command,
            "background": self.background,
        }
        if self.log_path is not None:
            payload["log_path"] = self.log_path
        return payload


def build_cluster_execution_commands(
    plan: ClusterLaunchPlan,
    *,
    ssh_bin: str = "ssh",
    ssh_user: str | None = None,
    ssh_options: tuple[str, ...] = (),
    ssh_host_field: str = "name",
    run_root: str = DEFAULT_CLUSTER_RUN_ROOT,
) -> tuple[ClusterExecutionCommand, ...]:
    if ssh_host_field not in {"name", "fabric-ip"}:
        raise ValueError("ssh_host_field must be name or fabric-ip")

    node_hosts = _execution_hosts(plan, ssh_host_field)
    resolved_ssh_options = ssh_options or DEFAULT_CLUSTER_SSH_OPTIONS
    commands: list[ClusterExecutionCommand] = []
    for command in plan.commands:
        ssh_host = node_hosts[command.target]
        if ssh_user:
            ssh_host = f"{ssh_user}@{ssh_host}"
        background = command.role == "vllm-serve"
        log_path = None
        remote_command = command.command
        if background:
            log_path = (
                f"{run_root.rstrip('/')}/{plan.runtime_id}/"
                f"{command.role}.{command.target}.log"
            )
            remote_command = _background_remote_command(command.command, log_path)
        argv = _ssh_argv(
            ssh_bin=ssh_bin,
            ssh_options=resolved_ssh_options,
            ssh_host=ssh_host,
            remote_command=remote_command,
        )
        commands.append(
            ClusterExecutionCommand(
                target=command.target,
                role=command.role,
                rank=command.rank,
                ssh_host=ssh_host,
                argv=argv,
                remote_command=remote_command,
                background=background,
                log_path=log_path,
            )
        )
    return tuple(commands)


def build_cluster_rollback_commands(
    plan: ClusterLaunchPlan,
    *,
    ssh_bin: str = "ssh",
    ssh_user: str | None = None,
    ssh_options: tuple[str, ...] = (),
    ssh_host_field: str = "name",
) -> tuple[ClusterExecutionCommand, ...]:
    if ssh_host_field not in {"name", "fabric-ip"}:
        raise ValueError("ssh_host_field must be name or fabric-ip")

    node_hosts = _execution_hosts(plan, ssh_host_field)
    resolved_ssh_options = ssh_options or DEFAULT_CLUSTER_SSH_OPTIONS
    commands: list[ClusterExecutionCommand] = []
    raw_nodes = plan.topology.get("nodes")
    if not isinstance(raw_nodes, list):
        raise ValueError("plan topology nodes must be a list")
    for rank, raw_node in enumerate(reversed(raw_nodes)):
        if not isinstance(raw_node, dict):
            raise ValueError("plan topology node entries must be mappings")
        target = raw_node.get("name")
        if not isinstance(target, str):
            raise ValueError("plan topology nodes must include name")
        ssh_host = node_hosts[target]
        if ssh_user:
            ssh_host = f"{ssh_user}@{ssh_host}"
        remote_command = "ray stop --force"
        commands.append(
            ClusterExecutionCommand(
                target=target,
                role="rollback-ray-stop",
                rank=rank,
                ssh_host=ssh_host,
                argv=_ssh_argv(
                    ssh_bin=ssh_bin,
                    ssh_options=resolved_ssh_options,
                    ssh_host=ssh_host,
                    remote_command=remote_command,
                ),
                remote_command=remote_command,
            )
        )
    return tuple(commands)


def load_cluster_topologies(path: Path | None = None) -> dict[str, ClusterTopology]:
    topologies_file = _cluster_topologies_file(path)
    if path is not None and not Path(path).is_file():
        raise FileNotFoundError(f"topology file not found: {path}")
    with topologies_file.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    items = raw.get("topologies")
    if not isinstance(items, dict):
        raise ValueError("cluster topology file must define a topologies mapping")
    return {
        topology_id: _topology_from_config(topology_id, config)
        for topology_id, config in items.items()
    }


def normalize_env_assignments(items: list[str]) -> list[str]:
    by_key: dict[str, str] = {}
    order: list[str] = []
    for item in items:
        if "=" not in item or item.startswith("="):
            raise ValueError(f"environment assignments must be KEY=VALUE: {item}")
        key = item.split("=", 1)[0]
        if key not in by_key:
            order.append(key)
        by_key[key] = item
    return [by_key[key] for key in order]


def transport_environment(
    *,
    transport_profile: str,
    fabric_iface: str | None = DEFAULT_FABRIC_IFACE,
    fabric_cidr: str = DEFAULT_FABRIC_CIDR,
    runtime_id: str,
    extra_env: list[str] | None = None,
) -> list[str]:
    if transport_profile not in SUPPORTED_TRANSPORT_PROFILES:
        raise ValueError(f"unsupported transport profile: {transport_profile}")

    items: list[str] = []
    if fabric_iface:
        items.extend(
            [
                f"NCCL_SOCKET_IFNAME={fabric_iface}",
                f"GLOO_SOCKET_IFNAME={fabric_iface}",
            ]
        )
    items.extend(extra_env or [])
    if transport_profile == "socket":
        items.append("NCCL_IB_DISABLE=1")
    elif transport_profile == "roce-a":
        items.extend(
            [
                "NCCL_IB_DISABLE=0",
                "NCCL_IB_ADDR_FAMILY=AF_INET",
                f"NCCL_IB_ADDR_RANGE={fabric_cidr}",
                "NCCL_IB_ROCE_VERSION_NUM=2",
                "NCCL_DEBUG=INFO",
                "NCCL_DEBUG_SUBSYS=INIT,NET,COLL,PROXY",
                f"GEMMA4_MTP_RUNTIME_ID={runtime_id}",
                (
                    "NCCL_DEBUG_FILE="
                    f"${{GEMMA4_MTP_RUN_ROOT:-{DEFAULT_CLUSTER_RUN_ROOT}}}/"
                    f"{runtime_id}/nccl/nccl.%h.%p.log"
                ),
            ]
        )
    return normalize_env_assignments(items)


def build_cluster_launch_plan(
    *,
    profile: ModelProfile,
    topology: ClusterTopology,
    node_count: int,
    runtime_id: str,
    transport_profile: str = DEFAULT_TRANSPORT_PROFILE,
    head_ip: str | None = None,
    fabric_iface: str | None = DEFAULT_FABRIC_IFACE,
    fabric_cidr: str = DEFAULT_FABRIC_CIDR,
    port: int = 8000,
    ray_port: int = 6379,
    vllm_bin: str = "vllm",
    venv: str | None = None,
    model_path: str | None = None,
    served_model_name: str | None = None,
    max_model_len: int | None = None,
    gpu_memory_utilization: float | None = None,
    max_num_seqs: int | None = None,
    max_num_batched_tokens: int | None = None,
    enable_mtp: bool = True,
    extra_env: list[str] | None = None,
) -> ClusterLaunchPlan:
    selected_nodes = topology.select_nodes(node_count)
    resolved_head_ip = head_ip or selected_nodes[0].fabric_ip
    tensor_parallel_size = sum(node.gpus for node in selected_nodes)
    environment = tuple(
        transport_environment(
            transport_profile=transport_profile,
            fabric_iface=fabric_iface,
            fabric_cidr=fabric_cidr,
            runtime_id=runtime_id,
            extra_env=extra_env,
        )
    )
    commands = tuple(
        _build_commands(
            profile=profile,
            nodes=selected_nodes,
            tensor_parallel_size=tensor_parallel_size,
            environment=environment,
            head_ip=resolved_head_ip,
            port=port,
            ray_port=ray_port,
            runtime_id=runtime_id,
            vllm_bin=vllm_bin,
            venv=venv,
            model_path=model_path,
            served_model_name=served_model_name,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            max_num_seqs=max_num_seqs,
            max_num_batched_tokens=max_num_batched_tokens,
            enable_mtp=enable_mtp,
        )
    )
    environment_hash = _sha256_json(list(environment))
    command_hash = _sha256_json([command.to_dict() for command in commands])
    topology_payload = {
        "id": topology.id,
        "label": topology.label,
        "head": selected_nodes[0].name,
        "nodes": [
            {"name": node.name, "fabric_ip": node.fabric_ip, "gpus": node.gpus}
            for node in selected_nodes
        ],
    }
    fingerprint = _sha256_json(
        {
            "runtime_id": runtime_id,
            "profile": profile.name,
            "topology": topology_payload,
            "transport_profile": transport_profile,
            "environment_sha256": environment_hash,
            "command_sha256": command_hash,
        }
    )
    return ClusterLaunchPlan(
        schema_version=CLUSTER_PLAN_SCHEMA_VERSION,
        dry_run_only=True,
        runtime_id=runtime_id,
        profile=profile.name,
        topology=topology_payload,
        node_count=node_count,
        tensor_parallel_size=tensor_parallel_size,
        transport_profile=transport_profile,
        commands=commands,
        environment=environment,
        resolved_environment_sha256=environment_hash,
        resolved_command_sha256=command_hash,
        dry_run_fingerprint=fingerprint,
        expected_live_gates=_expected_live_gates(transport_profile),
    )


def _build_commands(
    *,
    profile: ModelProfile,
    nodes: tuple[ClusterNode, ...],
    tensor_parallel_size: int,
    environment: tuple[str, ...],
    head_ip: str,
    port: int,
    ray_port: int,
    runtime_id: str,
    vllm_bin: str,
    venv: str | None,
    model_path: str | None,
    served_model_name: str | None,
    max_model_len: int | None,
    gpu_memory_utilization: float | None,
    max_num_seqs: int | None,
    max_num_batched_tokens: int | None,
    enable_mtp: bool,
) -> Iterable[ClusterLaunchCommand]:
    env_prefix = _env_prefix(environment)
    log_preamble = _cluster_log_preamble(environment, runtime_id)
    yield ClusterLaunchCommand(
        target=nodes[0].name,
        role="ray-head",
        rank=0,
        command=_with_venv(
            (
                f"{log_preamble}"
                f"{env_prefix}"
                f"ray start --head --node-ip-address={shlex.quote(head_ip)} "
                f"--port={ray_port}"
            ),
            venv,
        ),
    )
    for rank, node in enumerate(nodes[1:], start=1):
        yield ClusterLaunchCommand(
            target=node.name,
            role="ray-worker",
            rank=rank,
            command=_with_venv(
                (
                    f"{log_preamble}"
                    f"{env_prefix}"
                    f"ray start --address={shlex.quote(head_ip)}:{ray_port} "
                    f"--node-ip-address={shlex.quote(node.fabric_ip)}"
                ),
                venv,
            ),
        )
    serve_args = _build_vllm_serve_args(
        profile=profile,
        tensor_parallel_size=tensor_parallel_size,
        host="0.0.0.0",
        port=port,
        vllm_bin=vllm_bin,
        model_path=model_path,
        served_model_name=served_model_name,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_num_batched_tokens,
        enable_mtp=enable_mtp,
    )
    serve_environment = tuple(
        normalize_env_assignments([*environment, *DEFAULT_CLUSTER_VLLM_RUNTIME_ENV])
    )
    yield ClusterLaunchCommand(
        target=nodes[0].name,
        role="vllm-serve",
        rank=0,
        command=_with_venv(
            (
                f"{log_preamble}"
                f"ray status --address={shlex.quote(head_ip)}:{ray_port} && "
                f"{_ray_wait_command(head_ip, ray_port, len(nodes))} && "
                f"{_env_prefix(serve_environment)}{shlex.join(serve_args)}"
            ),
            venv,
        ),
    )


def _build_vllm_serve_args(
    *,
    profile: ModelProfile,
    tensor_parallel_size: int,
    host: str,
    port: int,
    vllm_bin: str,
    model_path: str | None,
    served_model_name: str | None,
    max_model_len: int | None,
    gpu_memory_utilization: float | None,
    max_num_seqs: int | None,
    max_num_batched_tokens: int | None,
    enable_mtp: bool,
) -> list[str]:
    args = [
        vllm_bin,
        "serve",
        model_path or profile.target,
        "--host",
        host,
        "--port",
        str(port),
        "--tensor-parallel-size",
        str(tensor_parallel_size),
        "--distributed-executor-backend",
        "ray",
        "--max-model-len",
        str(max_model_len or profile.max_model_len),
        "--gpu-memory-utilization",
        _format_float(gpu_memory_utilization or profile.gpu_memory_utilization),
        "--cpu-offload-gb",
        _format_float(profile.cpu_offload_gb),
        "--reasoning-parser",
        "gemma4",
    ]
    resolved_max_num_seqs = max_num_seqs or profile.max_num_seqs
    resolved_max_num_batched_tokens = (
        max_num_batched_tokens or profile.max_num_batched_tokens
    )
    if resolved_max_num_seqs is not None:
        args.extend(["--max-num-seqs", str(resolved_max_num_seqs)])
    if resolved_max_num_batched_tokens is not None:
        args.extend(["--max-num-batched-tokens", str(resolved_max_num_batched_tokens)])
    if profile.enforce_eager:
        args.append("--enforce-eager")
    if profile.language_model_only:
        args.append("--language-model-only")
    if profile.quantization is not None:
        args.extend(["--quantization", profile.quantization])
    if profile.kv_cache_dtype is not None:
        args.extend(["--kv-cache-dtype", profile.kv_cache_dtype])
    if served_model_name:
        args.extend(["--served-model-name", served_model_name])
    if enable_mtp:
        spec = {
            "method": "mtp",
            "model": profile.drafter,
            "num_speculative_tokens": profile.num_speculative_tokens,
        }
        args.extend(["--speculative-config", json.dumps(spec, separators=(",", ":"))])
    return args


def _cluster_topologies_file(path: Path | None) -> Any:
    if path is not None:
        return Path(path)
    package_topologies = resources.files(__package__).joinpath(
        "config/cluster_topologies.example.yaml"
    )
    if package_topologies.is_file():
        return package_topologies
    return DEFAULT_CLUSTER_TOPOLOGIES_PATH


def _topology_from_config(topology_id: str, config: Any) -> ClusterTopology:
    if not isinstance(config, dict):
        raise ValueError("topology entries must be mappings")
    label = config.get("label") or topology_id
    if not isinstance(label, str):
        raise ValueError("topology label must be a string")
    gpus_per_node = config.get("gpus_per_node", 1)
    if not isinstance(gpus_per_node, int) or gpus_per_node <= 0:
        raise ValueError("gpus_per_node must be a positive integer")
    raw_nodes = config.get("nodes")
    if not isinstance(raw_nodes, list):
        raise ValueError("topology nodes must be a list")
    nodes = []
    for raw_node in raw_nodes:
        if not isinstance(raw_node, dict):
            raise ValueError("topology node entries must be mappings")
        name = raw_node.get("name")
        fabric_ip = raw_node.get("fabric_ip")
        gpus = raw_node.get("gpus", gpus_per_node)
        if not isinstance(name, str) or not name:
            raise ValueError("topology node name must be a non-empty string")
        if not isinstance(fabric_ip, str) or not fabric_ip:
            raise ValueError(f"node {name!r} must define fabric_ip")
        if not isinstance(gpus, int) or gpus <= 0:
            raise ValueError(f"node {name!r} must have a positive GPU count")
        nodes.append(ClusterNode(name=name, fabric_ip=fabric_ip, gpus=gpus))
    return ClusterTopology(id=topology_id, label=label, nodes=tuple(nodes))


def _env_prefix(environment: tuple[str, ...]) -> str:
    if not environment:
        return ""
    return "env " + shlex.join(environment) + " "


def _with_venv(command: str, venv: str | None) -> str:
    if not venv:
        return command
    return f"source {shlex.quote(venv)}/bin/activate && {command}"


def _cluster_log_preamble(environment: tuple[str, ...], runtime_id: str) -> str:
    if not any(item.startswith("NCCL_DEBUG_FILE=") for item in environment):
        return ""
    nccl_dir = f"${{GEMMA4_MTP_RUN_ROOT:-{DEFAULT_CLUSTER_RUN_ROOT}}}/{runtime_id}/nccl"
    return f"mkdir -p {nccl_dir} && "


def _ray_wait_command(head_ip: str, ray_port: int, expected_nodes: int) -> str:
    code = (
        "import ray, sys, time\n"
        f"ray.init(address={f'{head_ip}:{ray_port}'!r}, ignore_reinit_error=True)\n"
        "deadline = time.time() + 180\n"
        "while time.time() < deadline:\n"
        "    alive = [node for node in ray.nodes() if node.get('Alive')]\n"
        f"    if len(alive) >= {expected_nodes}:\n"
        "        print(f'ray_ready alive_nodes={len(alive)}')\n"
        "        ray.shutdown()\n"
        "        sys.exit(0)\n"
        "    time.sleep(2)\n"
        "alive = [node for node in ray.nodes() if node.get('Alive')]\n"
        "print(f'ray_not_ready alive_nodes={len(alive)}')\n"
        "ray.shutdown()\n"
        "sys.exit(1)\n"
    )
    return "python3 -c " + shlex.quote(code)


def _expected_live_gates(transport_profile: str) -> tuple[str, ...]:
    gates = [
        "operator_approval_before_execute",
        "dry_run_fingerprint_matches_target_topology",
        "runtime_health_recorded_separately_from_model_list",
        "generation_smoke_exact_pong",
        "rollback_evidence_preserved",
    ]
    if transport_profile == "roce-a":
        gates.extend(
            [
                "models_endpoint_not_sufficient_for_roce_health",
                "runtime_bound_nccl_net_ib_logs",
                "zero_net_socket_fallback",
                "queue_drain_running_waiting_zero",
                "ray_node_actor_worker_continuity",
                "soak_without_new_shm_broadcast_warnings",
                "socket_fallback_preserved",
            ]
        )
    return tuple(gates)


def _execution_hosts(
    plan: ClusterLaunchPlan,
    ssh_host_field: str,
) -> dict[str, str]:
    hosts: dict[str, str] = {}
    raw_nodes = plan.topology.get("nodes")
    if not isinstance(raw_nodes, list):
        raise ValueError("plan topology nodes must be a list")
    for raw_node in raw_nodes:
        if not isinstance(raw_node, dict):
            raise ValueError("plan topology node entries must be mappings")
        name = raw_node.get("name")
        fabric_ip = raw_node.get("fabric_ip")
        if not isinstance(name, str) or not isinstance(fabric_ip, str):
            raise ValueError("plan topology nodes must include name and fabric_ip")
        hosts[name] = name if ssh_host_field == "name" else fabric_ip
    return hosts


def _background_remote_command(command: str, log_path: str) -> str:
    log_dir = str(Path(log_path).parent)
    log_stem = Path(log_path).stem
    pid_path = str(Path(log_path).with_suffix(".pid"))
    return (
        f"mkdir -p {shlex.quote(log_dir)} && "
        f"if [ -e {shlex.quote(log_path)} ]; then "
        f"mv {shlex.quote(log_path)} "
        f"{shlex.quote(log_dir)}/{shlex.quote(log_stem)}-$(date -u +%Y%m%dT%H%M%SZ).log; "
        "fi; "
        f"nohup setsid bash -lc {shlex.quote(command)} "
        f"> {shlex.quote(log_path)} 2>&1 < /dev/null & "
        f"echo $! > {shlex.quote(pid_path)}; disown; cat {shlex.quote(pid_path)}"
    )


def _ssh_argv(
    *,
    ssh_bin: str,
    ssh_options: tuple[str, ...],
    ssh_host: str,
    remote_command: str,
) -> tuple[str, ...]:
    return (
        ssh_bin,
        *ssh_options,
        ssh_host,
        "bash -lc " + shlex.quote(remote_command),
    )


def _sha256_json(payload: object) -> str:
    body = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def _format_float(value: float) -> str:
    value = float(value)
    if value.is_integer():
        return str(int(value))
    return str(value)
