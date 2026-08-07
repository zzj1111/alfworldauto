# Setup

## Any machine

```bash
git clone <repo-url> alfscaffold && cd alfscaffold
git checkout autoscaffold-rebuild

# Environment. Known-good resolved set (works on H100/H200 and B200 — cu128 wheels):
#   torch 2.8.0+cu128, vllm 0.11.0, ray 2.52.x, transformers 4.56.x, tensordict 0.10.x
uv venv auto && source auto/bin/activate
uv pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
uv pip install "vllm==0.11.0" "ray[default]" "tensordict>=0.8,!=0.9.0,<=0.10" \
               "transformers<=4.57.3" wandb openai pyyaml
uv pip install alfworld textworld     # the vendored env's undeclared runtime deps
uv pip install -e . --no-deps

# ALFWorld game files — MUST be node-local disk, never NFS (the ~18k-file scan on a
# network filesystem has taken a node to load 400+):
export ALFWORLD_DATA=/scratch/<you>/alfworld_data
alfworld-download

cp .autoscaffold.env.example .autoscaffold.env
$EDITOR .autoscaffold.env             # paths, GPUs, keys at the bottom

bash autoscaffold/start.sh --check    # verifies everything, launches nothing
bash autoscaffold/start.sh --exp my_run
```

Watch it:

```bash
bash autoscaffold/status.sh --watch   # one screen + anomaly warnings, no network
tail -f <exp_root>/<exp>/run_latest.log
```

## What the preflight refuses to launch without

- `json_2.1.1/` under ALFWORLD_DATA (the vendored env silently plays 0 games otherwise)
- an OpenAI credential (without it the Teacher declines every cycle and the run is a
  plain-RL control that reads as a null result)
- a python that imports `verl`, `agent_system`, `autoscaffold` from this repo

## B200 notes

- The env spec above is Blackwell-ready (cu128, vllm 0.11). Do not pin
  TORCH_CUDA_ARCH_LIST to 9.0 — B200 is sm_100.
- Do not export VLLM_ATTENTION_BACKEND=XFORMERS (the upstream example's H100-era
  choice); leave the backend to vllm 0.11's default on Blackwell.
- The container memory check reads the cgroup limit; /proc/meminfo inside a container
  reports the host and is not trusted.
- `+data.dataloader_num_workers=0` is already in the training script: the dataset is
  one batch of short prompts, and the default 8 workers only fork the trainer eight
  more times — a prior run's OOM kill took the node with it at a checkpoint save.

## Baseline (the comparison run)

`bash autoscaffold/train_alfworld.sh none <steps>` is byte-identical upstream
behavior — no injection, no recorder, no loss swap. `ARM_KEEPALIVE=plain-rl` runs it
automatically after the arm finishes.
