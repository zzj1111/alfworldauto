#!/usr/bin/env bash
# Resolve every machine-specific setting, once. Source it; do not execute it.
#
# Precedence, highest first: caller environment > $ARM_ENV_FILE (default
# <repo>/.autoscaffold.env) > portable defaults. config.py implements the same rules
# and the same value parsing; test_config.py + the rehearsal keep the two in step.
# Values are read literally, one line at a time — a site file is configuration, not a
# script, and is never executed.

ARM_ROOT="${ARM_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export ARM_ROOT

ARM_ENV_FILE="${ARM_ENV_FILE:-$ARM_ROOT/.autoscaffold.env}"
export ARM_ENV_FILE
if [[ -f "$ARM_ENV_FILE" ]]; then
  while IFS= read -r _line || [[ -n "$_line" ]]; do
    _line="${_line#"${_line%%[![:space:]]*}"}"
    [[ -z "$_line" || "$_line" == \#* ]] && continue
    [[ "$_line" == export\ * ]] && _line="${_line#export }"
    [[ "$_line" != *=* ]] && continue
    _k="${_line%%=*}"; _v="${_line#*=}"
    [[ "$_k" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]] || continue
    case "$_v" in                                # trailing comments, quoted '#' kept
      \"*)  _v="${_v#\"}"; _v="${_v%%\"*}" ;;
      \'*)  _v="${_v#\'}"; _v="${_v%%\'*}" ;;
      *)    _v="${_v%%[[:space:]]#*}" ;;
    esac
    _v="${_v#"${_v%%[![:space:]]*}"}"; _v="${_v%"${_v##*[![:space:]]}"}"
    [[ -n "${!_k+x}" ]] && continue              # caller wins
    printf -v "$_k" '%s' "$_v"
    export "${_k?}"
  done < "$ARM_ENV_FILE"
  unset _line _k _v
fi

# ---------------- workspace ----------------
export ARM_EXP="${ARM_EXP:-alf_autoscaffold}"
export ARM_WORKSPACE="${ARM_WORKSPACE:-$ARM_ROOT/runs}"
export ARM_EXP_ROOT="${ARM_EXP_ROOT:-$ARM_WORKSPACE/exp}"
export ARM_CKPT_ROOT="${ARM_CKPT_ROOT:-$ARM_WORKSPACE/ckpts}"
export ARM_LOG_DIR="${ARM_LOG_DIR:-$ARM_WORKSPACE/logs}"
mkdir -p "$ARM_EXP_ROOT" "$ARM_CKPT_ROOT" "$ARM_LOG_DIR" 2>/dev/null || true

if [[ -z "${ARM_RAY_TMP:-}" ]]; then
  if [[ -d /dev/shm && -w /dev/shm ]]; then ARM_RAY_TMP="/dev/shm/zray_${ARM_EXP}"
  else ARM_RAY_TMP="$ARM_WORKSPACE/ray_tmp/${ARM_EXP}"; fi
fi
export ARM_RAY_TMP

# ---------------- interpreter ----------------
if [[ -z "${ARM_PYTHON:-}" ]]; then
  for _cand in "${VIRTUAL_ENV:+$VIRTUAL_ENV/bin/python}" \
               "$ARM_ROOT/auto/bin/python" "$ARM_ROOT/.venv/bin/python" \
               "$(command -v python3)" "$(command -v python)"; do
    if [[ -n "$_cand" && -x "$_cand" ]]; then ARM_PYTHON="$_cand"; break; fi
  done
  unset _cand
fi
export ARM_PYTHON

# ---------------- vLLM / CUDA toolchain ----------------
# vllm 0.11's default attention backend JIT-compiles at engine init (FlashInfer),
# which needs ninja + a matching toolkit and died with FileNotFoundError('ninja') on
# the very first smoke run. FLASH_ATTN is prebuilt and proven on this machine;
# ARM_VLLM_ATTN=auto leaves the choice to vllm (the right setting on Blackwell if
# flash-attn lacks sm_100 kernels).
if [[ "${ARM_VLLM_ATTN:-FLASH_ATTN}" != "auto" ]]; then
  export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-${ARM_VLLM_ATTN:-FLASH_ATTN}}"
fi
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"
export VLLM_USE_FLASHINFER="${VLLM_USE_FLASHINFER:-0}"
# These env vars only cover the paths that consult them. On the B200 sandbox
# flashinfer still JIT-compiled and failed: its CUDA 12.4 cannot target compute_100a.
# Uninstalling the package is what actually forces the flash-attn fallback —
# b200_launch.sh does that, and check_flashinfer below names it when it is still
# importable on a Blackwell card.
if [[ -n "${ARM_ROOT:-}" && -z "${ARM_SKIP_FLASHINFER_CHECK:-}" ]]; then
  _cc="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' .')"
  if [[ "$_cc" == "100" || "$_cc" == "120" ]] \
     && "${ARM_PYTHON:-python}" -c "import flashinfer" 2>/dev/null; then
    echo "  [warn] flashinfer is installed on an sm_${_cc} card — if the engine dies in JIT" >&2
    echo "         compilation, remove it: pip uninstall -y flashinfer-python (vLLM then" >&2
    echo "         falls back to the prebuilt flash-attn wheel)" >&2
  fi
  unset _cc
fi
# Detected from the cards actually present; pinning 9.0 on a B200 builds kernels the
# driver will not load, and leaving it unset makes every JIT build all archs.
if [[ -z "${TORCH_CUDA_ARCH_LIST:-}" ]]; then
  _caps="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | sort -u | paste -sd';' -)"
  [[ -n "$_caps" ]] && export TORCH_CUDA_ARCH_LIST="$_caps"
  unset _caps
fi

# ---------------- model + data ----------------
export ARM_MODEL="${ARM_MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
export ALFWORLD_DATA="${ALFWORLD_DATA:-$HOME/.cache/alfworld}"

# ---------------- placement ----------------
export ARM_GPUS="${ARM_GPUS:-0,1}"
export ARM_N_GPUS="${ARM_N_GPUS:-2}"
export ARM_TP="${ARM_TP:-2}"
export ARM_GPU_MEM="${ARM_GPU_MEM:-0.6}"
export ARM_VLLM_PORT="${ARM_VLLM_PORT:-8110}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$ARM_GPUS}"

# ---------------- wandb ----------------
export ARM_WANDB="${ARM_WANDB:-0}"
export ARM_WANDB_PROJECT="${ARM_WANDB_PROJECT:-verl_agent_alfworld}"
if [[ "$ARM_WANDB" == "1" ]]; then
  export WANDB_PROJECT="${WANDB_PROJECT:-$ARM_WANDB_PROJECT}"
  # ONE run per experiment: every trainer subprocess and the orchestrator append to
  # the same run instead of each init minting a fresh one.
  export WANDB_RUN_ID="${ARM_WANDB_RUN_ID:-${WANDB_RUN_ID:-$(printf '%s' "$ARM_EXP" | tr -c 'A-Za-z0-9_-' '_')}}"
  export WANDB_RESUME="${WANDB_RESUME:-allow}"
  [[ -n "${ARM_WANDB_ENTITY:-}" ]] && export WANDB_ENTITY="${WANDB_ENTITY:-$ARM_WANDB_ENTITY}"
  export WANDB_DIR="${WANDB_DIR:-$ARM_WORKSPACE/wandb}"
  mkdir -p "$WANDB_DIR" 2>/dev/null || true
  ARM_TRAINER_LOGGER="['console','wandb']"
else
  ARM_TRAINER_LOGGER="['console']"
fi
export ARM_TRAINER_LOGGER

autoscaffold_env_summary() {
  cat <<EOF
--- autoscaffold environment ---
  repo          $ARM_ROOT
  site file     $ARM_ENV_FILE $( [[ -f "$ARM_ENV_FILE" ]] && echo "(loaded)" || echo "(absent, defaults)")
  experiment    $ARM_EXP
  workspace     $ARM_WORKSPACE
  checkpoints   $ARM_CKPT_ROOT
  ray tmp       $ARM_RAY_TMP
  python        ${ARM_PYTHON:-NOT FOUND}
  model         $ARM_MODEL
  alfworld data $ALFWORLD_DATA $( [[ -d "$ALFWORLD_DATA/json_2.1.1" ]] && echo "(present)" || echo "(MISSING json_2.1.1 - run alfworld-download)")
  gpus          $ARM_GPUS (n=$ARM_N_GPUS tp=$ARM_TP mem=$ARM_GPU_MEM)
  wandb         $( [[ "$ARM_WANDB" == "1" ]] && echo "on -> ${ARM_WANDB_ENTITY:-<default>}/$ARM_WANDB_PROJECT run=$WANDB_RUN_ID" || echo "off" )
  openai key    $( [[ -n "${OPENAI_API_KEY:-}" ]] && echo "set" || { [[ -n "${AUTOSCAFFOLD_OPENAI_KEY_FILE:-}" ]] && echo "$AUTOSCAFFOLD_OPENAI_KEY_FILE" || echo "NOT SET - the Teacher will decline every cycle"; } )
--------------------------------
EOF
}
