#!/usr/bin/env bash
# One command to start the experiment on a new machine.
#
#   bash autoscaffold/start.sh --check          # detect + verify only, launch nothing
#   bash autoscaffold/start.sh                  # launch in the background, follow logs
#   bash autoscaffold/start.sh --fg             # run in the foreground
#
# Settings come from .autoscaffold.env (copy .autoscaffold.env.example) or exported
# variables; flags below override both. Credentials go in the site file or the
# environment, never on this command line.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export ARM_ROOT="${ARM_ROOT:-$(cd "$HERE/.." && pwd)}"
export ARM_RUN_ID="${ARM_RUN_ID:-$(date +%Y%m%d_%H%M%S)}"

CHECK_ONLY=0; FOREGROUND=0; CYCLES_FLAG=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --check)        CHECK_ONLY=1; shift ;;
    --fg)           FOREGROUND=1; shift ;;
    --exp)          export ARM_EXP="$2"; shift 2 ;;
    --gpus)         export ARM_GPUS="$2"; export ARM_N_GPUS="$(awk -F, '{print NF}' <<<"$2")"; shift 2 ;;
    --tp)           export ARM_TP="$2"; shift 2 ;;
    --model)        export ARM_MODEL="$2"; shift 2 ;;
    --alfworld-data) export ALFWORLD_DATA="$2"; shift 2 ;;
    --python)       export ARM_PYTHON="$2"; shift 2 ;;
    --workspace)    export ARM_WORKSPACE="$2"; shift 2 ;;
    --prompts)      export TRAIN_DATA_SIZE="$2"; shift 2 ;;
    --rollout-n)    export ARM_ROLLOUT_N="$2"; shift 2 ;;
    --kl)           export ARM_KL="$2"; shift 2 ;;
    --cycles)       CYCLES_FLAG="$2"; shift 2 ;;
    --target-step)  export ARM_TARGET_STEP="$2"; shift 2 ;;
    --wandb)        export ARM_WANDB=1; shift ;;
    --keepalive)    export ARM_KEEPALIVE="$2"; shift 2 ;;
    -h|--help)      sed -n '2,11p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 1 ;;
  esac
done

source "$ARM_ROOT/autoscaffold/env.sh"
# Resolved AFTER the site file loads, in the documented order: flag > file > default.
CYCLES="${CYCLES_FLAG:-${ARM_CYCLES:-30}}"

autoscaffold_env_summary
echo "preflight:"
fail=0
bad()  { echo "  [FAIL] $1"; echo "         fix: $2"; fail=1; }
note() { echo "  [warn] $1"; }
okay() { echo "  [ok]   $1"; }

[[ -x "${ARM_PYTHON:-/nonexistent}" ]] || bad "no usable python" "set ARM_PYTHON in $ARM_ENV_FILE"
"$ARM_PYTHON" -c "import verl, agent_system, autoscaffold" 2>/dev/null \
  || bad "ARM_PYTHON cannot import verl/agent_system/autoscaffold from this repo" \
         "run from the repo root; install deps per autoscaffold/SETUP.md"

# The repo itself is code 256 Ray workers import in parallel; on a network fs that
# is an RPC storm (measured: load 39 -> 381 in ten minutes, 384 D-state processes).
# Data got this check first; the repo earned it the hard way.
repo_fs=$(df --output=fstype "$ARM_ROOT" 2>/dev/null | tail -1 | tr -d ' ')
case "$repo_fs" in
  nfs*|lustre|gpfs|cifs|fuse*) bad "the REPO sits on $repo_fs (network fs)"     "clone to node-local disk and launch from there — parallel worker imports over NFS have taken this node to load 380+" ;;
  *) okay "repo on $repo_fs" ;;
esac
py_fs=$(df --output=fstype "$(dirname "$(readlink -f "$ARM_PYTHON")")" 2>/dev/null | tail -1 | tr -d ' ')
case "$py_fs" in
  nfs*|lustre|gpfs|cifs|fuse*) bad "the PYTHON ENVIRONMENT sits on $py_fs (network fs)"     "copy the env to node-local disk (cp -a works; only bin/python is invoked directly) — torch+vllm are thousands of files and 256 workers importing them over NFS is the same storm as the repo case; this exact miss relaunched the storm once" ;;
  *) okay "python env on $py_fs" ;;
esac
ws_fs=$(df --output=fstype "$ARM_WORKSPACE" 2>/dev/null | tail -1 | tr -d ' ')
case "$ws_fs" in
  nfs*|lustre|gpfs|cifs|fuse*) note "workspace on $ws_fs — acceptable only as the checkpoint root (few large writers); keep exp state and logs local" ;;
esac

if [[ ! -d "$ALFWORLD_DATA/json_2.1.1" ]]; then
  bad "ALFWORLD_DATA=$ALFWORLD_DATA has no json_2.1.1/" \
      "run 'alfworld-download' or point ALFWORLD_DATA at an existing copy — the vendored env silently plays 0 games otherwise"
else
  fstype=$(df --output=fstype "$ALFWORLD_DATA" 2>/dev/null | tail -1 | tr -d ' ')
  case "$fstype" in
    nfs*|lustre|gpfs|cifs|fuse*) bad "ALFWORLD_DATA sits on $fstype (network fs)" \
      "copy it to node-local disk — the ~18k-file scan on NFS has taken a node to load 400+" ;;
    *) okay "alfworld data on $fstype" ;;
  esac
fi

if [[ -z "${OPENAI_API_KEY:-}" && -z "${AUTOSCAFFOLD_OPENAI_KEY_FILE:-}" ]]; then
  bad "no OpenAI credential" \
      "export OPENAI_API_KEY or set AUTOSCAFFOLD_OPENAI_KEY_FILE in $ARM_ENV_FILE. Without it the Teacher declines every cycle and the run is plain RL with an empty scaffold — a null result that reads as a finding"
fi
[[ "${ARM_WANDB:-0}" == "1" && -z "${WANDB_API_KEY:-}" && ! -f "$HOME/.netrc" ]] \
  && note "ARM_WANDB=1 but no WANDB_API_KEY and no ~/.netrc — wandb may prompt or fail"

# Container memory limit, not /proc/meminfo (which reports the HOST inside a container
# and passes exactly when it should fail).
read -r _used _limit < <("$ARM_PYTHON" - <<'PYEOF'
from autoscaffold.config import container_memory
u, l = container_memory()
print(u or 0, l or 0)
PYEOF
)
_envs=$(( ${TRAIN_DATA_SIZE:-32} * ${ARM_ROLLOUT_N:-8} ))
if [[ "${_limit%.*}" -gt 0 ]]; then
  _per=$(( ${_limit%.*} / ${ARM_N_GPUS:-2} ))
  if (( _per < 24 )); then
    note "memory limit ${_limit}G over ${ARM_N_GPUS:-2} ranks = ${_per}G/rank; $_envs env actors + all ranks saving at once has OOM-killed a node — lower --prompts or use fewer GPUs"
  else
    okay "memory ${_used}G used of ${_limit}G; $_envs env actors planned"
  fi
fi

for g in $(tr ',' ' ' <<<"$ARM_GPUS"); do
  mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$g" 2>/dev/null || echo "")
  [[ -n "$mem" && "$mem" -gt 2000 ]] && note "GPU $g already holds ${mem} MiB — someone else may be on it"
done

_tgt="${ARM_TARGET_STEP:-0}"; _k="${ARM_K:-10}"; _reach=$(( CYCLES * _k ))
echo "  [info] plan: $CYCLES cycles x K=$_k steps = step $_reach"
if [[ "$_tgt" != "0" && "$_reach" -lt "$_tgt" ]]; then
  note "reaches step $_reach, short of ARM_TARGET_STEP=$_tgt — the relaunch chain will restart until the target"
fi
echo "  [info] batch: ${TRAIN_DATA_SIZE:-32} prompts x ${ARM_ROLLOUT_N:-8} rollouts = $_envs episodes/step; ppo_mini=${PPO_MINI_BS:-$_envs} micro=${MICRO_BS:-32}/gpu"

if (( fail )); then
  echo ""; echo "not launching. See .autoscaffold.env.example for every setting."; exit 1
fi
okay "ready"
(( CHECK_ONLY )) && { echo "--check: nothing launched."; exit 0; }

# ---------------- launch ----------------
STATE_DIR="$ARM_EXP_ROOT/$ARM_EXP"
mkdir -p "$STATE_DIR"
CHAIN="$STATE_DIR/.chain_${ARM_RUN_ID}.sh"
cat > "$CHAIN" <<CHAINEOF
#!/usr/bin/env bash
# Relaunch until ARM_TARGET_STEP (state.json + complete checkpoints make the arm
# resumable); three failures under 60s each = a broken environment, stop.
TARGET="\${ARM_TARGET_STEP:-0}"
fast=0
while :; do
  t0=\$(date +%s)
  cd "$ARM_ROOT" && "$ARM_PYTHON" -m autoscaffold.run_arm "$CYCLES"
  rc=\$?
  ran=\$(( \$(date +%s) - t0 ))
  step=\$("$ARM_PYTHON" -c 'import json,sys;print(json.load(open(sys.argv[1]))["step"])' \\
         "$STATE_DIR/state.json" 2>/dev/null || echo 0)
  echo "[chain] exited rc=\$rc after \${ran}s at step \$step (target \$TARGET)"
  [[ "\$TARGET" == "0" ]] && break
  (( step >= TARGET )) && { echo "[chain] target reached"; break; }
  if (( rc != 0 && ran < 60 )); then
    fast=\$(( fast + 1 )); (( fast >= 3 )) && { echo "[chain] 3 fast failures — environment broken, stopping"; break; }
  else fast=0; fi
  echo "[chain] resuming from step \$step in 60s"; sleep 60
done
if [[ "\${ARM_KEEPALIVE:-none}" == "plain-rl" ]]; then
  echo "[chain] ============================================================"
  echo "[chain] arm finished/stopped; what follows is only the keepalive baseline"
  echo "[chain] ============================================================"
  exec bash "$ARM_ROOT/autoscaffold/keepalive.sh"
fi
exit \$rc
CHAINEOF
chmod +x "$CHAIN"

RUN_LOG="$STATE_DIR/run_${ARM_RUN_ID}.log"
ln -sfn "$(basename "$RUN_LOG")" "$STATE_DIR/run_latest.log"
echo ""
echo "launching $ARM_EXP for $CYCLES cycles (target step ${ARM_TARGET_STEP:-none})"
echo "  state    $STATE_DIR"
echo "  status   bash autoscaffold/status.sh --exp $ARM_EXP"
if (( FOREGROUND )); then
  exec bash "$CHAIN" 2>&1 | tee "$RUN_LOG"
fi
nohup bash "$CHAIN" > "$RUN_LOG" 2>&1 &
pid=$!
echo "$pid" > "$STATE_DIR/run.pid"
echo "  pid      $pid   stop: kill \$(cat $STATE_DIR/run.pid)"
echo "  log      tail -f $RUN_LOG"
tail -n +1 -f --pid="$pid" "$RUN_LOG"
