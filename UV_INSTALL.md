# Installing the ALFWorld AutoScaffold / GiGPO-helper stack with `uv` on a new server

Scope: **verl-agent + `agent_system/skill_opt/autoscaffold/`** (the Teacher-driven
scaffold controller — this is the *original*; StitchCUDA's `cudascaffold/` is a port of
this to CUDA-kernel generation) **+ `agent_system/skill/`** (skill injection) **+ the
GiGPO ALFWorld actor** (`examples/gigpo_trainer/run_alfworld*.sh`). Not covered: the
`mathscaffold`, `search_orchestrator`/`search_iter_supervisor` code also living under
`agent_system/skill_opt/` — those are a different project sharing this same repo checkout.

## ⚠️ Before you do anything: this repo's `origin` is not yours to push to

```
origin  https://github.com/langfengQ/verl-agent.git
```

Unlike StitchCUDA (your own fork), this `origin` is the **upstream open-source project**.
There is no personal fork remote configured here, and `gh auth status` shows the local
GitHub token is currently invalid anyway. I have not pushed, and pushing to `origin`
directly would either fail (no write access) or — if it somehow succeeded — land your
unrelated research code in someone else's public repo. Pick one before moving code:
- fork `langfengQ/verl-agent` to your own GitHub account and I push there instead, or
- I package a tarball/rsync of the tracked+untracked files you want, no git involved.

I have **not committed anything** in this repo (unlike StitchCUDA, where you asked me to
commit+push already) — everything below is local-only until you decide.

## 0. Correction to StitchCUDA's `UV_INSTALL.md`

That doc says `/dev/shm/verl_env` is a bare tmpfs env with no rebuild script. That's only
half true: `/dev/shm/verl_env` and this box's persistent conda env at
`/mnt/data1/zha00175/miniconda/envs/verl` have **identical directory mtimes** (down to the
microsecond), and `examples/gigpo_trainer/run_alfworld_frombase.sh` spells out the
relationship directly:
```bash
# tmpfs copy of the env when it is staged (see .verl_env_done), else the NFS original.
if [ -f /dev/shm/.verl_env_done ]; then VENV=/dev/shm/verl_env; else VENV=/mnt/data1/zha00175/miniconda/envs/verl; fi
```
So `/dev/shm/verl_env` is a **speed cache** of the conda env, not the only copy — the conda
env is the durable source, and most of the ALFWorld launch scripts (`run_alfworld_baseline_3b.sh`,
`run_alfworld_p0_3b.sh`, `run_alfworld_skilltrain_*.sh`, `watchdog.py`) point at the conda
path directly, not the tmpfs one. Reboot risk is real for whatever's *only* in `/dev/shm`
at the time, but the base environment itself isn't as fragile as that doc implied.

## 1. System prerequisites

Same as StitchCUDA (same physical box, same GPUs): NVIDIA driver supporting CUDA 12.8+,
a CUDA 12.8/12.9 toolkit with working `nvcc` for the target GPU's arch (system default
`nvcc` here is CUDA 11.5 and cannot target Hopper/sm_90 — see StitchCUDA's `UV_INSTALL.md`
§1), Python 3.12 exactly (flash-attn wheel is a cp312 binary), gcc/g++, ninja, and `uv`.

## 2. Get the code onto the target

`agent_system/skill_opt/` and `agent_system/skill/` have **never been committed** to this
repo (`git log -- agent_system/skill_opt` is empty) — they, plus a batch of modified
upstream files (`agent_system/environments/env_manager.py`,
`agent_system/environments/env_package/alfworld/envs.py`,
`agent_system/multi_turn_rollout/rollout_loop.py`, `verl/trainer/ppo/ray_trainer.py`,
`verl/trainer/config/ppo_trainer.yaml`, `recipe/dapo/dapo_ray_trainer.py`,
`examples/search/retriever/retrieval_server.py`) and ~20 new `examples/gigpo_trainer/*.sh`
/ `examples/search/retriever/*.sh` scripts, are the actual project-specific work and are
all sitting as local changes. Given §0's git constraint, moving this is on you (tarball,
rsync, or your own fork) — I can prepare whichever once you've picked.

## 3. Build the environment with `uv`

Identical recipe to StitchCUDA's (same shared base stack — see that repo's
`UV_INSTALL.md` §3 for the full rationale on install order and the `--no-deps` gotcha):

```bash
cd verl-agent
uv venv --python 3.12 .venv
source .venv/bin/activate

uv pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0
uv pip install -r requirements-uv.txt

wget -nv https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.1/flash_attn-2.8.1+cu12torch2.8cxx11abiFALSE-cp312-cp312-linux_x86_64.whl
uv pip install flash_attn-2.8.1+cu12torch2.8cxx11abiFALSE-cp312-cp312-linux_x86_64.whl

# this repo IS the verl package too (same upstream-verl layout, pyproject name="verl").
# --no-deps for the same reason as StitchCUDA: setup.py's own install_requires is looser
# than the verified pins in requirements-uv.txt and will fight the resolver otherwise.
uv pip install -e . --no-deps
```

## 4. ALFWorld game data — don't copy it, regenerate it

`ALFWORLD_DATA` (every launch script sets this) points at
`/mnt/data1/zha00175/skillzero-env/alfworld_data` on this box — a directory with enough
small files that even `du -sh` over the NFS mount didn't return in 2+ minutes. Don't rsync
that. The `alfworld` pip package ships its own downloader; on the target, after the venv is
built:
```bash
export ALFWORLD_DATA=/wherever/you/want/it
alfworld-download        # official CLI from the alfworld package, fetches the game data fresh
```
Only fall back to copying the source directory if `alfworld-download` fails (e.g. no
internet egress on the target) — it's also worth noting this data directory is shared
across other, unrelated projects on this box (`skillzero-env/` has leftover files from an
earlier, separate SkillZero experiment) — copying it verbatim would drag those along too.

## 5. Environment variables

```bash
# CUDA toolchain (same reasoning as StitchCUDA: avoid the ancient system nvcc, and make
# sure the venv's own `ninja` binary is what gets found on PATH)
export PATH=/path/to/verl-agent/.venv/bin:/usr/local/cuda-12.9/bin:$PATH
export CUDA_HOME=/usr/local/cuda-12.9
export TORCH_CUDA_ARCH_LIST=9.0     # match the target GPU's compute capability

export ALFWORLD_DATA=/path/to/alfworld_data   # see §4
export VLLM_ATTENTION_BACKEND=FLASH_ATTN

# skill_opt Teacher (GPT-5.5 controller) — same pattern and same DEFAULT_KEY_FILE path as
# StitchCUDA's cudascaffold/teacher.py (agent_system/skill_opt/autoscaffold/teacher.py is
# literally what cudascaffold/teacher.py was ported from). Create your own key file on the
# target; do not copy the source machine's:
#   printf 'OPENAI_API_KEY=sk-...\n' > /path/to/your/openai.env
# and update DEFAULT_KEY_FILE in agent_system/skill_opt/autoscaffold/teacher.py accordingly.

# Skill injection (see agent_system/skill/, wired through env_manager.py)
export GIGPO_SKILL_PATH=/path/to/skills.json
export GIGPO_SKILL_MODE=full        # or none
export GIGPO_SKILL_FORCE=0
export GIGPO_SKILL_P=1.0            # or unset for per-task weighted p from the skill json
```

`agent_system/skill_opt/autoscaffold/run_arm.py` and `agent_system/skill_opt/watchdog.py`
both hardcode `/mnt/data1/zha00175/...` paths (log dirs, checkpoint dirs, `sys.path.insert`)
— same as StitchCUDA's `cudascaffold/run_arm.py` hardcoding `ROOT`. Update these to the
target's actual paths before running; they will silently read/write to the source
machine's filesystem otherwise (only obvious if the target happens to also mount
`/mnt/data1`, in which case it's worse — it'll actually succeed, against the wrong data).

## 6. Assets that are not part of the environment

- **ALFWorld game data** — see §4, regenerate rather than copy.
- **Base/checkpoint models** — e.g. `watchdog.py`'s `ACTOR_MODEL` points at
  `/mnt/data1/zha00175/ckpts_alfworld/verl_agent_alfworld/.../global_step_150/actor/huggingface`,
  a GiGPO-trained checkpoint, not a stock HF model. If you're continuing from that specific
  run, it has to travel with you (rsync from `/mnt/data1/zha00175/ckpts_alfworld/`);
  if you're starting fresh, point at a stock `Qwen2.5-*-Instruct` instead.
- **Skill files** — `agent_system/skill/skills_v0.json` is tracked in-repo (small), but
  the `skills_push98.json` and similar variants referenced by some launch scripts live
  under `/mnt/data1/zha00175/gigpo_helper_skillopt/` — bring those if you need that exact
  variant.
- **Secrets** — `/mnt/data1/zha00175/tool-agent-secrets/openai.env` — recreate by hand, see §5.

## 7. Smoke test before trusting a full run

Same shape as StitchCUDA's — this was actually run against the shared base stack already
(torch, flash_attn, vllm, verl all import cleanly from that test); the ALFWorld-specific
addition to check here:
```bash
python -c "import alfworld, textworld, gym, gymnasium; print('ok')"
python -c "from agent_system.skill_opt.autoscaffold import scaffold, adapters, gates, loop, observation, teacher, run_arm; print('ok')"
```
Not verified: an actual ALFWorld rollout (needs `ALFWORLD_DATA` populated first, see §4)
or a live training step.
