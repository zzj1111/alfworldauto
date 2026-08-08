#!/usr/bin/env bash
# One-shot bootstrap + launch for a fresh B200 (or any Blackwell/Hopper) machine.
#
#   bash autoscaffold/b200_launch.sh                 # bootstrap, preflight, launch
#   bash autoscaffold/b200_launch.sh --check         # bootstrap + preflight only
#   bash autoscaffold/b200_launch.sh --exp my_run --gpus 0,1,2,3 --tp 2
#
# Idempotent: every phase skips itself when its product already exists, so re-running
# after a failure continues where it stopped. Extra flags are passed through to
# start.sh untouched.
#
# Before the first launch you must put credentials in .autoscaffold.env (bottom of
# the file — OPENAI_API_KEY and, if ARM_WANDB=1, WANDB_API_KEY). This script refuses
# to launch without them but never reads them onto a command line.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
cd "$ROOT"

say() { echo "[b200] $*"; }
die() { echo "[b200] FATAL: $*" >&2; exit 1; }

# ---------- 0. site file ----------
if [[ ! -f .autoscaffold.env ]]; then
  cp .autoscaffold.env.b200 .autoscaffold.env
  say "created .autoscaffold.env from the B200 template — EDIT IT NOW:"
  say "  - adjust the /scratch/... paths to this machine's mounts"
  say "  - paste OPENAI_API_KEY (and WANDB_API_KEY) at the bottom"
  die "then re-run this script"
fi
# Pull the path settings we need for bootstrap (comment-stripping like config.py).
_site() { sed -n "s/^[[:space:]]*$1=\([^#]*\).*/\1/p" .autoscaffold.env | tail -1 | xargs; }
ALFWORLD_DATA="${ALFWORLD_DATA:-$(_site ALFWORLD_DATA)}"
ARM_DATA_DIR="${ARM_DATA_DIR:-$(_site ARM_DATA_DIR)}"
ARM_PYTHON="${ARM_PYTHON:-$(_site ARM_PYTHON)}"
[[ -n "$ALFWORLD_DATA" ]] || die "ALFWORLD_DATA missing from .autoscaffold.env"

# ---------- 1. python environment (known-good cu128 set; Blackwell-ready) ----------
VENV="${ARM_PYTHON:+$(dirname "$(dirname "$ARM_PYTHON")")}"
VENV="${VENV:-$ROOT/auto}"
if [[ ! -x "$VENV/bin/python" ]]; then
  command -v uv >/dev/null || die "uv not found — install with: curl -LsSf https://astral.sh/uv/install.sh | sh"
  say "building venv at $VENV (torch 2.8.0+cu128, vllm 0.11.0)"
  uv venv "$VENV"
  # shellcheck disable=SC1091
  source "$VENV/bin/activate"
  uv pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
  uv pip install "vllm==0.11.0" "ray[default]" "tensordict>=0.8,!=0.9.0,<=0.10" \
                 "transformers<=4.57.3" wandb openai pyyaml
  uv pip install alfworld textworld
  uv pip install -e . --no-deps
  # vllm 0.11 pulls flashinfer-python in, and on Blackwell it JIT-compiles at engine
  # init for compute_100a — which the CUDA 12.4 toolkit on these nodes cannot target,
  # so every launch died there. Env vars did not stop it (they only cover the paths
  # that read them, and the Ray runtime_env copies did not propagate); removing the
  # package is what forces the prebuilt flash-attn wheel.
  uv pip uninstall flashinfer-python 2>/dev/null || true
  rm -rf "$VENV"/lib/python*/site-packages/flashinfer*
else
  say "venv exists: $VENV — skipping install"
  # shellcheck disable=SC1091
  source "$VENV/bin/activate"
fi
export ARM_PYTHON="$VENV/bin/python"

# ---------- 2. ALFWorld game files (~18k files) ----------
if [[ ! -d "$ALFWORLD_DATA/json_2.1.1" ]]; then
  say "downloading ALFWorld data to $ALFWORLD_DATA"
  mkdir -p "$ALFWORLD_DATA"
  ALFWORLD_DATA="$ALFWORLD_DATA" alfworld-download
else
  say "ALFWorld data present — skipping download"
fi
export ALFWORLD_DATA

# ---------- 3. model weights (prefetch so the first cycle doesn't stall) ----------
MODEL="$(_site ARM_MODEL)"; MODEL="${MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
if [[ ! -d "$MODEL" ]]; then   # hub id, not a local dir -> warm the HF cache
  say "prefetching $MODEL into the HF cache"
  "$ARM_PYTHON" - "$MODEL" <<'PY'
import sys
from huggingface_hub import snapshot_download
snapshot_download(sys.argv[1])
PY
fi

# ---------- 4. credentials present? (never printed) ----------
grep -qE "^[[:space:]]*OPENAI_API_KEY=." .autoscaffold.env || [[ -n "${OPENAI_API_KEY:-}" ]] \
  || [[ -n "$(_site AUTOSCAFFOLD_OPENAI_KEY_FILE)" ]] \
  || die "no OpenAI credential in .autoscaffold.env — without it the Teacher declines every cycle and the run is a plain-RL control"

# ---------- 5. preflight + launch (start.sh owns everything from here) ----------
[[ -n "$ARM_DATA_DIR" ]] && export ARM_DATA_DIR
say "handing over to start.sh $*"
exec bash autoscaffold/start.sh "$@"
