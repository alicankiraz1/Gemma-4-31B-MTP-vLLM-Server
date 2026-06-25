from __future__ import annotations

import json
import shlex
from pathlib import Path

import pytest

from gemma4_mtp_vllm.cluster import (
    ClusterNode,
    ClusterTopology,
    DEFAULT_CLUSTER_SSH_OPTIONS,
    build_cluster_execution_commands,
    build_cluster_launch_plan,
    build_cluster_rollback_commands,
    load_cluster_topologies,
    normalize_env_assignments,
    transport_environment,
)
from gemma4_mtp_vllm.profiles import load_profiles, resolve_profile


def _nodes(count: int) -> list[ClusterNode]:
    return [
        ClusterNode(name=f"spark-{index:02d}", fabric_ip=f"198.51.100.{index}", gpus=1)
        for index in range(1, count + 1)
    ]


def _profile():
    return resolve_profile("tp2_2x32_fp8_gpuonly", load_profiles())


def test_cluster_launch_plan_scales_two_four_six_eight_nodes():
    for node_count in (2, 4, 6, 8):
        topology = ClusterTopology(
            id=f"dgx-spark-{node_count}",
            label=f"{node_count}x DGX Spark",
            nodes=tuple(_nodes(node_count)),
        )

        plan = build_cluster_launch_plan(
            profile=_profile(),
            topology=topology,
            node_count=node_count,
            runtime_id="run-123",
            transport_profile="socket",
            served_model_name="gemma-4-31b-mtp",
        )

        assert [command.role for command in plan.commands] == [
            "ray-head",
            *["ray-worker"] * (node_count - 1),
            "vllm-serve",
        ]
        assert plan.node_count == node_count
        assert plan.dry_run_only is True
        assert plan.tensor_parallel_size == node_count
        assert plan.commands[-1].target == "spark-01"
        assert f"--tensor-parallel-size {node_count}" in plan.commands[-1].command
        assert f"if len(alive) >= {node_count}" in plan.commands[-1].command
        assert "VLLM_WORKER_MULTIPROC_METHOD=spawn" in plan.commands[-1].command
        assert "SAFETENSORS_FAST_GPU=1" in plan.commands[-1].command
        assert "NVIDIA_TF32_OVERRIDE=1" in plan.commands[-1].command
        assert "VLLM_LOGGING_LEVEL=INFO" in plan.commands[-1].command


def test_socket_and_roce_transport_profiles_are_deduped_and_runtime_bound():
    socket_env = transport_environment(
        transport_profile="socket",
        fabric_iface="fabric0",
        fabric_cidr="198.51.100.0/24",
        runtime_id="run-123",
        extra_env=["NCCL_IB_DISABLE=0", "NCCL_DEBUG=WARN"],
    )
    assert "NCCL_IB_DISABLE=1" in socket_env
    assert "NCCL_IB_DISABLE=0" not in socket_env
    assert "NCCL_SOCKET_IFNAME=fabric0" in socket_env

    roce_env = transport_environment(
        transport_profile="roce-a",
        fabric_iface="fabric0",
        fabric_cidr="198.51.100.0/24",
        runtime_id="run-123",
        extra_env=["NCCL_IB_DISABLE=1", "NCCL_DEBUG=WARN"],
    )
    assert "NCCL_IB_DISABLE=0" in roce_env
    assert "NCCL_IB_ADDR_FAMILY=AF_INET" in roce_env
    assert "NCCL_IB_ADDR_RANGE=198.51.100.0/24" in roce_env
    assert "NCCL_IB_ROCE_VERSION_NUM=2" in roce_env
    assert "NCCL_DEBUG=INFO" in roce_env
    assert "NCCL_DEBUG_SUBSYS=INIT,NET,COLL,PROXY" in roce_env
    assert "GEMMA4_MTP_RUNTIME_ID=run-123" in roce_env
    assert any(
        item.startswith("NCCL_DEBUG_FILE=${GEMMA4_MTP_RUN_ROOT:-artifacts/cluster-runs}/run-123/nccl/")
        for item in roce_env
    )
    assert not any(item.startswith("NCCL_IB_HCA=") for item in roce_env)
    assert not any(item.startswith("NCCL_IB_GID_INDEX=") for item in roce_env)
    assert not any(item.startswith("NCCL_IB_MERGE_NICS=") for item in roce_env)


def test_cluster_plan_json_is_dry_run_and_has_required_evidence_fields():
    topology = ClusterTopology(
        id="dgx-spark-4",
        label="4x DGX Spark",
        nodes=tuple(_nodes(4)),
    )
    plan = build_cluster_launch_plan(
        profile=_profile(),
        topology=topology,
        node_count=4,
        runtime_id="roce-run-123",
        transport_profile="roce-a",
        fabric_cidr="198.51.100.0/24",
        served_model_name="gemma-4-31b-mtp",
    )
    payload = plan.to_dict()

    assert payload["schema_version"] == 1
    assert payload["dry_run_only"] is True
    assert payload["runtime_id"] == "roce-run-123"
    assert payload["profile"] == "tp2_2x32_fp8_gpuonly"
    assert payload["transport_profile"] == "roce-a"
    assert len(payload["resolved_environment_sha256"]) == 64
    assert len(payload["resolved_command_sha256"]) == 64
    assert len(payload["dry_run_fingerprint"]) == 64
    assert "runtime_bound_nccl_net_ib_logs" in payload["expected_live_gates"]
    assert "models_endpoint_not_sufficient_for_roce_health" in payload["expected_live_gates"]

    rendered = json.dumps(payload)
    forbidden = ("ray stop", "pkill", "killall", "ssh ", "systemctl stop")
    assert not any(token in rendered for token in forbidden)


def test_shell_rendering_round_trips_vllm_serve_command():
    topology = ClusterTopology(
        id="dgx-spark-2",
        label="2x DGX Spark",
        nodes=tuple(_nodes(2)),
    )
    plan = build_cluster_launch_plan(
        profile=_profile(),
        topology=topology,
        node_count=2,
        runtime_id="run-123",
        transport_profile="socket",
        served_model_name="gemma-4-31b-mtp",
    )

    serve_command = plan.commands[-1].command
    parsed = shlex.split(serve_command)
    assert "vllm" in parsed
    assert "serve" in parsed
    assert "--speculative-config" in parsed


def test_cluster_execution_commands_wrap_plan_for_live_ssh():
    topology = ClusterTopology(
        id="dgx-spark-4",
        label="4x DGX Spark",
        nodes=tuple(_nodes(4)),
    )
    plan = build_cluster_launch_plan(
        profile=_profile(),
        topology=topology,
        node_count=4,
        runtime_id="run-live-123",
        transport_profile="socket",
        served_model_name="gemma-4-31b-mtp",
    )

    commands = build_cluster_execution_commands(
        plan,
        ssh_user="operator",
        ssh_options=("-o", "BatchMode=yes"),
        ssh_host_field="fabric-ip",
    )

    assert len(commands) == 5
    assert commands[0].argv[:4] == (
        "ssh",
        "-o",
        "BatchMode=yes",
        "operator@198.51.100.1",
    )
    assert commands[0].background is False
    assert "ray start --head" in commands[0].remote_command
    assert commands[-1].role == "vllm-serve"
    assert commands[-1].background is True
    assert commands[-1].log_path == (
        "artifacts/cluster-runs/run-live-123/vllm-serve.spark-01.log"
    )
    assert "nohup setsid bash -lc" in commands[-1].remote_command
    assert "disown" in commands[-1].remote_command
    assert "vllm-serve.spark-01.pid" in commands[-1].remote_command
    assert "--distributed-executor-backend ray" in commands[-1].remote_command
    assert "--speculative-config" in commands[-1].remote_command
    assert "ray stop" not in commands[-1].remote_command


def test_cluster_live_helpers_use_safe_default_ssh_and_explicit_rollback():
    topology = ClusterTopology(
        id="dgx-spark-2",
        label="2x DGX Spark",
        nodes=tuple(_nodes(2)),
    )
    plan = build_cluster_launch_plan(
        profile=_profile(),
        topology=topology,
        node_count=2,
        runtime_id="run-live-123",
        transport_profile="socket",
        served_model_name="gemma-4-31b-mtp",
    )

    commands = build_cluster_execution_commands(plan, ssh_host_field="fabric-ip")
    assert commands[0].argv[: len(DEFAULT_CLUSTER_SSH_OPTIONS) + 1] == (
        "ssh",
        *DEFAULT_CLUSTER_SSH_OPTIONS,
    )

    rollback = build_cluster_rollback_commands(plan, ssh_host_field="fabric-ip")
    assert [command.role for command in rollback] == [
        "rollback-ray-stop",
        "rollback-ray-stop",
    ]
    assert [command.target for command in rollback] == ["spark-02", "spark-01"]
    assert all("ray stop --force" in command.remote_command for command in rollback)


def test_invalid_topologies_are_rejected(tmp_path):
    topology = ClusterTopology(id="bad", label="bad", nodes=tuple(_nodes(1)))
    with pytest.raises(ValueError, match="at least 2 nodes"):
        build_cluster_launch_plan(
            profile=_profile(),
            topology=topology,
            node_count=1,
            runtime_id="run-123",
            transport_profile="socket",
        )

    with pytest.raises(ValueError, match="fabric_ip"):
        ClusterTopology(
            id="missing-fabric",
            label="bad",
            nodes=(ClusterNode(name="spark-01", fabric_ip="", gpus=1), ClusterNode(name="spark-02", fabric_ip="198.51.100.2", gpus=1)),
        ).select_nodes(2)

    with pytest.raises(ValueError, match="positive"):
        ClusterTopology(
            id="bad-gpu",
            label="bad",
            nodes=(ClusterNode(name="spark-01", fabric_ip="198.51.100.1", gpus=0), ClusterNode(name="spark-02", fabric_ip="198.51.100.2", gpus=1)),
        ).select_nodes(2)

    missing = tmp_path / "missing.yaml"
    with pytest.raises(FileNotFoundError):
        load_cluster_topologies(missing)


def test_load_cluster_topologies_from_public_safe_yaml(tmp_path):
    path = tmp_path / "topologies.yaml"
    path.write_text(
        """
topologies:
  example:
    label: Example public topology
    gpus_per_node: 1
    nodes:
      - name: spark-01
        fabric_ip: 198.51.100.1
      - name: spark-02
        fabric_ip: 198.51.100.2
""",
        encoding="utf-8",
    )

    topologies = load_cluster_topologies(path)
    assert topologies["example"].label == "Example public topology"
    assert topologies["example"].select_nodes(2)[1].fabric_ip == "198.51.100.2"


def test_normalize_env_assignments_rejects_malformed_values():
    with pytest.raises(ValueError, match="KEY=VALUE"):
        normalize_env_assignments(["NCCL_DEBUG"])
