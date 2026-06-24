# P2-001 DGX Spark Cluster Compatibility

## Snapshot

- Branch: `codex/p2-001-dgx-spark-cluster-compat`
- Scope: public-safe dry-run planning for 2x and larger DGX Spark clusters.
- Live execution: not implemented.
- GPU-consuming commands: not run.
- Runtime upgrade: not run; the existing vLLM `0.21.0` package pin remains.
- Default profile change: not run.

## Public-Safe Topology Contract

Real cluster inventory must stay outside tracked source files. The checked-in
`config/cluster_topologies.example.yaml` uses documentation-only addresses and
is packaged as `src/gemma4_mtp_vllm/config/cluster_topologies.example.yaml`.
Operators should copy that shape into an ignored local file such as
`config/cluster_topologies.private.yaml`.

Each topology defines:

- `label`: reader-facing topology label.
- `gpus_per_node`: default GPU count for node entries.
- `nodes`: ordered DGX Spark nodes with `name`, `fabric_ip`, and optional
  per-node `gpus`.

The first selected node is the Ray head. `node_count` must be at least `2`; the
default tensor parallel size is the total selected GPU count.

## CLI Contract

`vllm-mtp cluster-plan` prints either shell commands or deterministic JSON:

```bash
vllm-mtp cluster-plan \
  --profile tp2_2x32_fp8_gpuonly \
  --topology-file config/cluster_topologies.private.yaml \
  --topology dgx-spark-private \
  --node-count 4 \
  --runtime-id my-runtime-id \
  --transport-profile socket \
  --format shell
```

The command is dry-run-only. It has no `--execute` flag and does not generate
`ray stop`, broad process kill, service stop, or remote shell commands.

The generated command roles are:

- `ray-head`: starts the Ray head on the first selected node.
- `ray-worker`: starts one Ray worker command per remaining selected node.
- `vllm-serve`: waits for Ray node count and then prints a distributed
  `vllm serve` command with `--distributed-executor-backend ray`.

## Transport Profiles

`socket` is the default fallback/baseline transport and sets
`NCCL_IB_DISABLE=1`.

`roce-a` is opt-in and sets:

- `NCCL_IB_DISABLE=0`
- `NCCL_IB_ADDR_FAMILY=AF_INET`
- `NCCL_IB_ADDR_RANGE=<operator fabric CIDR>`
- `NCCL_IB_ROCE_VERSION_NUM=2`
- `NCCL_DEBUG=INFO`
- `NCCL_DEBUG_SUBSYS=INIT,NET,COLL,PROXY`
- `GEMMA4_MTP_RUNTIME_ID=<runtime id>`
- runtime-scoped `NCCL_DEBUG_FILE`

The planner deduplicates env assignments by key. It does not add HCA, GID
index, or dual-rail overrides automatically.

## Evidence and Gates

The JSON plan includes:

- `schema_version`
- `dry_run_only`
- `runtime_id`
- `profile`
- `topology`
- `node_count`
- `tensor_parallel_size`
- `transport_profile`
- `commands`
- `environment`
- `resolved_environment_sha256`
- `resolved_command_sha256`
- `dry_run_fingerprint`
- `expected_live_gates`

For RoCE-A, the expected gates explicitly state that model-list liveness is not
enough. Promotion still requires generation smoke, queue drain, runtime-bound
NCCL log proof, Ray node/actor/worker continuity, soak, rollback evidence, and
preserved socket fallback.

## Verification

Code-only verification for this slice:

```bash
.venv/bin/python -m pytest tests/test_cluster.py tests/test_cli.py tests/test_release_scripts.py -q
.venv/bin/python -m compileall -q src tests
.venv/bin/python -m pip check
git diff --check
```

Before publishing, run the full suite and the public-sanitization scan.
