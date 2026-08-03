"""Adapters: the four side-effecting functions loop.py needs, wired to the existing
training script + vLLM rollout machinery.

Split on purpose:
  - PURE helpers (agg_per_task, assemble_measure, assemble_signals, persist_scaffold)
    turn raw rollout transcripts into the exact dicts the gates/observation consume.
    These are unit-tested with mock transcripts (no GPU).
  - GPU/subprocess adapters (train/eval/measure/signals/restore) drive
    run_alfworld_frombase.sh and a served vLLM. They need a live GPU and are marked
    NEEDS-SMOKE-TEST; the command/rollout construction is here, but nothing is launched
    from this module.

Data hygiene (locked): measurement (A/B, signals, failure trajectories) uses the TRAIN
split (is_train=True); valid_seen is only ever the eval anchor.
"""
from __future__ import annotations

import json
import os
import subprocess

# ---- config defaults (all overridable via the cfg dict) ------------------- #
ROOT = "/mnt/data1/zha00175/verl-agent"
# The interpreter. A tmpfs copy of the env is ~6-25% faster to import from under concurrency
# and removes 135k files from the NFS path; it does NOT survive a reboot, so fall back to the
# NFS original rather than failing when it is absent.
VENV_PY = ("/dev/shm/verl_env/bin/python"
           if os.path.exists("/dev/shm/.verl_env_done")
           else "/mnt/data1/zha00175/miniconda/envs/verl/bin/python")
TRAIN_SH = f"{ROOT}/examples/gigpo_trainer/run_alfworld_frombase.sh"
N_PER_TASK = 30                 # games per touched task for the A/B (fixed; not adaptive)
GROUP_N = 8                     # rollouts per game when measuring all-fail groups (= GiGPO group)


# =========================== PURE (unit-tested) =========================== #
def agg_per_task(transcripts, tasks):
    """{task: (success_rate, n)} over the given tasks. transcripts: [{task_type, success}]."""
    out = {}
    for t in tasks:
        rel = [1.0 if tr.get("success") else 0.0 for tr in transcripts if tr.get("task_type") == t]
        if rel:
            out[t] = (round(sum(rel) / len(rel), 4), len(rel))
    return out


def assemble_measure(bare_tr, current_tr, candidate_tr, tasks):
    """Three paired rollout passes -> the dict ab_gate consumes."""
    return {
        "bare": agg_per_task(bare_tr, tasks),
        "current": agg_per_task(current_tr, tasks),
        "candidate": agg_per_task(candidate_tr, tasks),
    }


def group_by_gamefile(transcripts):
    """Group flat transcripts by gamefile -> [{task, outcomes:[bool,...]}]. Several stochastic
    rollouts of the SAME game form a group; a group with no success = no GiGPO gradient there
    (an all-fail group). Robust to whether games repeat across passes (it keys on gamefile)."""
    from collections import OrderedDict
    groups = OrderedDict()
    for tr in transcripts:
        gf = tr.get("gamefile")
        if gf is None:
            continue
        g = groups.setdefault(gf, {"task": tr.get("task_type"), "outcomes": []})
        g["outcomes"].append(bool(tr.get("success")))
    return list(groups.values())


def annotate_trajectory(tr):
    """Attach computed per-trajectory signals so the Teacher does not have to re-derive them
    from raw text: how often it opened/took things, how many actions the env rejected, and
    whether it ever held an object at all (a run with takes=0 never picked anything up)."""
    steps = tr.get("steps") or []
    acts = [str(s.get("action", "")).strip().lower() for s in steps]
    opens = sum(1 for a in acts if a.startswith("open "))
    takes = sum(1 for a in acts if a.startswith("take "))
    invalid = sum(1 for s in steps if not s.get("valid", True))
    return {**tr, "computed": {"n_steps": len(steps), "opens": opens, "takes": takes,
                               "invalid_actions": invalid, "held_an_object": takes > 0}}


def assemble_signals(bare_tr, injected_tr, group_results, tasks, max_failures=40, max_successes=6):
    """Build the Teacher's train-side signals from rollouts.

    bare_tr/injected_tr: paired frozen-policy rollouts (no scaffold / current scaffold).
    group_results: [{task, outcomes: [bool,...]}] grouped rollouts (GROUP_N per game) used
                   to count all-fail groups (a group with no successful rollout = no GiGPO
                   gradient there).
    """
    bare = agg_per_task(bare_tr, tasks)
    inj = agg_per_task(injected_tr, tasks)
    per_task_gap = {}
    for t in tasks:
        b = bare.get(t, (0.0, 0))
        i = inj.get(t, (0.0, 0))
        per_task_gap[t] = {"bare": b[0], "injected": i[0], "gap": round(i[0] - b[0], 4), "n": b[1]}
    all_fail = {}
    for g in group_results:
        t = g.get("task")
        outs = g.get("outcomes", [])
        d = all_fail.setdefault(t, {"all_fail": 0, "total": 0})
        d["total"] += 1
        if outs and not any(outs):
            d["all_fail"] += 1
    # Failures are the diagnostic material; a few SUCCESSES are kept as contrast so the Teacher
    # can see what working behaviour looks like on the same tasks, not just what went wrong.
    failures = [annotate_trajectory(t) for t in bare_tr if not t.get("success")][:max_failures]
    successes = [annotate_trajectory(t) for t in bare_tr if t.get("success")][:max_successes]
    return {"per_task_gap": per_task_gap, "all_fail_groups": all_fail,
            "failures": failures, "successes": successes}



def existing_ckpt_step(ckpt_root):
    """Highest usable global_step_N already in `ckpt_root`, or 0.

    A cold start does NOT imply step 0: an arm can be seeded with a checkpoint from an earlier
    run (continuing a converged policy). Starting the counter at 0 there makes the first cycle
    ask verl to train "to step K" when it is already far past K — verl resumes, does one step,
    exits, and the loop then looks for a checkpoint dir that was never written.
    """
    import glob
    import re
    best = 0
    for d in glob.glob(os.path.join(ckpt_root, "global_step_*")):
        m = re.search(r"global_step_(\d+)$", os.path.basename(d))
        if m and ckpt_is_usable(d):
            best = max(best, int(m.group(1)))
    return best


def persist_scaffold(scaffold, path):
    """Atomic write so training's mtime hot-reload never sees a half-written file."""
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(scaffold, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)
    return path


# ==================== GPU / SUBPROCESS ADAPTERS (NEEDS SMOKE TEST) ==================== #
# Everything below drives real training / a served vLLM. Command construction is final;
# it must be smoke-tested on GPUs 0,1 before a real run. Nothing here launches on import.

def _kill_vllm():
    """Kill any lingering vLLM workers owned by this user (module-level; no side effects)."""
    subprocess.run(["bash", "-lc",
        "for p in $(ps -eo pid,cmd | grep -E 'EngineCore|VllmWorker|VLLM::|vllm serve' | grep zha00175 "
        "| grep -v grep | grep -v 'bash -lc' | awk '{print $1}'); do kill -9 $p 2>/dev/null; done"])


class StepFailed(RuntimeError):
    """A GPU subprocess did not produce the artifact the loop needs.

    Raised instead of letting the loop advance on a checkpoint that was never written.
    Without this the loop happily reports `valid_seen avg=None` and keeps proposing
    scaffold edits against a model that never trained — hours of plausible-looking garbage.
    """


# Ray's agents must register with the raylet inside this deadline or the raylet fate-shares
# and dies. The default 30s is not survivable when this conda env (on NFS /mnt/data1) has a
# cold page cache — agent startup then spends >30s just faulting in imports. Cost of a larger
# value is only a slower failure when Ray is genuinely broken.
RAY_AGENT_REGISTER_TIMEOUT_MS = "300000"
# Ray has TWO independent registration deadlines and they fail the same way. The agent one is
# above; this is the WORKER one (default 60s). On a loaded box the ALFWorld eval spawns its env
# workers faster than they can come up, the raylet reaps the late ones, and it respawns them --
# a thrash loop that pins the machine (observed: load 242 on 192 cores, 128 of our ray processes
# all under 25s old) while the eval makes no progress and never errors out.
RAY_WORKER_REGISTER_TIMEOUT_S = "600"

# vLLM startup deadline. 600s was enough on a warm page cache; it is not enough right after a
# training run evicts it, because this env lives on NFS and `import vllm` then reads cold.
VLLM_HEALTH_TIMEOUT = 2400


def _base_env():
    return {"RAY_agent_register_timeout_ms": RAY_AGENT_REGISTER_TIMEOUT_MS,
        "RAY_worker_register_timeout_seconds": RAY_WORKER_REGISTER_TIMEOUT_S}


def _run(cmd, log_path, env):
    with open(log_path, "a") as lf:
        return subprocess.run(["bash", "-lc", cmd], cwd=ROOT, stdout=lf,
                              stderr=subprocess.STDOUT,
                              env={**os.environ, **_base_env(), **env})


def ckpt_is_usable(path):
    """True only for a checkpoint that is COMPLETE, not merely started.

    Two ways to get this wrong, both of which have bitten this harness:

    1. `hf_model` alone silently restarts from base weights on resume (the verl gotcha), so the
       actor state has to be there.
    2. verl creates `global_step_N/actor/` BEFORE writing any shard and updates
       `latest_checkpointed_iteration.txt` only after the whole save finishes -- ~19 GB over NFS,
       minutes during which the directory exists and the checkpoint does not. A crash in that
       window leaves a torn dir. Treating it as usable makes train_adapter skip retraining, and
       the eval that follows resumes from the TRACKER (an older step) and reports that number as
       this step's score: no training happened, no error was raised, and the Teacher is told its
       edit moved held-out success.

    So require the full shard set: every rank's model/optim/extra for the world size actually
    written, plus the trainer's own `data.pt`, which verl writes after the actor save.
    """
    import glob
    import re
    actor = os.path.join(path, "actor")
    if not os.path.isdir(actor) or not os.path.isfile(os.path.join(path, "data.pt")):
        return False
    models = glob.glob(os.path.join(actor, "model_world_size_*_rank_*.pt"))
    if not models:
        return False
    sizes = {int(m.group(1)) for m in
             (re.search(r"model_world_size_(\d+)_rank_\d+\.pt$", os.path.basename(f)) for f in models)
             if m}
    if len(sizes) != 1:                      # shards from two different world sizes -> not trustworthy
        return False
    ws = sizes.pop()
    for kind in ("model", "optim", "extra_state"):
        for rank in range(ws):
            if not os.path.isfile(os.path.join(actor, f"{kind}_world_size_{ws}_rank_{rank}.pt")):
                return False
    return True


def train_adapter(scaffold_path, from_step, to_step, cfg):
    """Advance training from from_step to to_step with the scaffold injected per-task p
    (phase 'weighted' = mode=full, FORCE=0, per-task p from the scaffold json; resume_mode
    =auto continues the same model/optimizer). Returns the checkpoint dir for to_step.

    NOTE (verl resume gotcha): checkpoints MUST save model+optimizer+extra, not only
    hf_model, or --resume silently restarts from base. run_alfworld_frombase.sh already
    saves full state; do not narrow save_contents.
    """
    exp = cfg["exp"]
    env = {"EXP": exp, "SKILL_PATH": scaffold_path, "ALF_GPUS": cfg["gpus"],
           "ALF_MODEL": cfg["model"], "ALF_RAY_TMP": cfg["ray_tmp"],
           "EVAL_SPLIT": "eval_in_distribution", "TEST_FREQ": "99999",
           "SAVE_FREQ": str(cfg.get("steps_per_cycle", 10)), "VAL_BEFORE": "False",
           "TRAIN_DATA_SIZE": str(cfg.get("train_data_size", 16)),
           "MAX_CKPT": str(cfg.get("max_ckpt", 60)),
           # 4-GPU runs need world_size to match the checkpoint's FSDP sharding, and the
           # reference model that KL introduces eats the headroom vLLM needs to wake its pool
           # for the post-training val — hence the lower gpu_memory_utilization.
           "N_GPUS": str(cfg.get("n_gpus", 2)), "TP_SIZE": str(cfg.get("tp_size", 2)),
           "GPU_MEM": str(cfg.get("gpu_mem", 0.6)), "VAL_BS": str(cfg.get("val_bs", 128))}
    ckpt = f"/mnt/data1/zha00175/gigpo_helper_ckpts/{exp}/global_step_{to_step}"
    # Idempotent resume. Each to_step is trained exactly once, so an existing usable checkpoint
    # means that training already finished -- the loop just died before recording it (e.g. during
    # the eval). Re-running it would ask verl to advance to a step it is already at: it resumes,
    # has nothing to do, exits without writing, and the cycle fails with StepFailed. Observed on
    # the step-200 boundary; this makes an unattended restart pick up where it stopped.
    if ckpt_is_usable(ckpt):
        cfg.get("log", lambda *a: None)(
            f"[train] {from_step}->{to_step}: checkpoint already on disk, skipping retrain")
        return ckpt
    extra = cfg.get("train_extra", "")     # e.g. 'actor_rollout_ref.actor.use_kl_loss=False' for KL=0
    proc = _run(f"bash {TRAIN_SH} weighted {to_step} {extra}", cfg["train_log"], env)
    if not ckpt_is_usable(ckpt):
        raise StepFailed(
            f"training {from_step}->{to_step} wrote no usable checkpoint at {ckpt} "
            f"(subprocess rc={proc.returncode}); see {cfg['train_log']}")
    return ckpt


def eval_adapter(checkpoint, val_n, cfg):
    """Standalone eval on valid_seen (phase 'none' -> no injection), averaged over val_n
    draws with different env seeds. Returns {avg, per_task, draws}."""
    from agent_system.skill_opt.push98_loop import parse_val  # reuse the existing parser
    exp = cfg["exp"]
    to_step = checkpoint.rstrip("/").split("global_step_")[-1]
    draws, pers, cnts = [], [], []
    for d in range(val_n):
        elog = f"{cfg['log_dir']}/{exp}_eval_s{to_step}_d{d}.log"
        env = {"EXP": exp, "SKILL_PATH": cfg["scaffold_path"], "ALF_GPUS": cfg["gpus"],
               "ALF_MODEL": cfg["model"], "ALF_RAY_TMP": cfg["ray_tmp"],
               "EVAL_SPLIT": "eval_in_distribution", "VAL_ONLY": "True", "VAL_BEFORE": "True",
               "ENV_SEED": str(d),
               # N_GPUS is NOT a tuning knob here: eval loads the FSDP shards written by
               # training, whose filenames encode the world size (model_world_size_4_rank_*.pt).
               # Evaluating on a different GPU count makes verl look for shards that were never
               # written. Everything world-size-critical must match train_adapter exactly.
               "N_GPUS": str(cfg.get("n_gpus", 2)), "TP_SIZE": str(cfg.get("tp_size", 2)),
               "GPU_MEM": str(cfg.get("gpu_mem", 0.6)), "VAL_BS": str(cfg.get("val_bs", 128))}
        _run(f"bash {TRAIN_SH} none {to_step}", elog, env)
        sr, per, cnt = parse_val(elog, with_counts=True)
        if sr is not None:
            draws.append(sr)
            pers.append(per)
            cnts.append(cnt)
    if not draws:
        raise StepFailed(
            f"eval at step {to_step} parsed no val/success_rate from any of {val_n} draws; "
            f"see {cfg['log_dir']}/{exp}_eval_s{to_step}_d*.log")
    if len(draws) < val_n:
        # A draw can die on a Ray worker collision with the training run that just tore down.
        # Averaging the survivors is the right call -- the measurement is expensive and a
        # partial one still informs the Teacher -- but it must not look like a full one: the
        # eval anchors every accept/revert decision, so its sample size has to be visible.
        cfg.get("log", lambda *a: None)(
            f"[warn] eval at step {to_step}: only {len(draws)}/{val_n} draws produced a "
            f"val/success_rate; averaging the survivors. See "
            f"{cfg['log_dir']}/{exp}_eval_s{to_step}_d*.log")
    avg = round(sum(draws) / len(draws), 4)
    from .scaffold import TASKS
    per_task = ({t: round(sum(p.get(t, 0.0) for p in pers) / len(pers), 3) for t in TASKS}
                if pers else {})
    # Episodes behind each per-task rate, SUMMED over draws — the denominator a reader needs to
    # know which per-task numbers mean anything. ALFWorld's sampler draws an uneven task mix, so
    # a rare category can land a handful of episodes per draw and read 0.000 or 1.000 while
    # saying nothing. Empty for logs written before the counts metric existed.
    per_task_n = ({t: int(sum(c.get(t, 0) for c in cnts)) for t in TASKS
                   if any(t in c for c in cnts)} if cnts else {})
    return {"avg": avg, "per_task": per_task, "per_task_n": per_task_n, "draws": draws,
            "n_draws": len(draws), "n_draws_requested": val_n}


def _ensure_hf(checkpoint, cfg):
    """Merge FSDP shards -> HF and copy tokenizer/config from the base model if missing, so
    vLLM can serve `{checkpoint}/actor/huggingface`. Mirrors capture_fails.py."""
    import glob
    import shutil
    ckact = f"{checkpoint}/actor"
    hf = f"{ckact}/huggingface"
    if not os.path.exists(f"{hf}/config.json"):
        subprocess.run(["bash", "-lc",
            f"cd {ROOT} && PATH={os.path.dirname(VENV_PY)}:$PATH {VENV_PY} scripts/model_merger.py "
            f"merge --backend fsdp --local_dir {ckact} --target_dir {hf}"], check=True)
    base = cfg.get("model")
    for f in glob.glob(f"{base}/*"):
        b = os.path.basename(f)
        if (b.startswith("tokenizer") or b in ("vocab.json", "merges.txt", "special_tokens_map.json",
                                               "generation_config.json", "config.json")) \
           and not os.path.exists(f"{hf}/{b}"):
            shutil.copy(f, f"{hf}/{b}")
    return hf


def _alive(pid):
    """True only if the pid exists AND is not a zombie. os.kill(pid,0) reports zombies as alive,
    so we read the /proc state and treat 'Z' (defunct, already dead) as not-alive."""
    try:
        with open(f"/proc/{int(pid)}/stat") as f:
            state = f.read().rsplit(")", 1)[-1].split()[0]
        return state != "Z"
    except Exception:
        return False


def _safe_uid(pid):
    try:
        return os.stat(f"/proc/{pid}").st_uid
    except Exception:
        return -1


def _graceful_kill(pids, grace=20):
    """SIGTERM the pids, wait up to `grace`s for them to exit, then SIGKILL ONLY the survivors.
    Never SIGKILLs a live CUDA process outright: abruptly killing a process mid-GPU-op is what
    deadlocked the driver (see GPU01_wedged_report.md). SIGTERM lets vLLM release its context."""
    import time
    import signal
    pids = [int(p) for p in pids if str(p).strip().isdigit()]
    for p in pids:
        try:
            os.kill(p, signal.SIGTERM)
        except Exception:
            pass
    t0 = time.time()
    while time.time() - t0 < grace:
        if not any(_alive(p) for p in pids):
            return
        time.sleep(1)
    for p in pids:                            # last resort, only for stragglers
        try:
            os.kill(p, signal.SIGKILL)
        except Exception:
            pass


def _kill_on_gpus(gpus):
    """Gracefully stop leftover compute PIDs that are MINE on EXACTLY these GPUs (SIGTERM ->
    wait -> SIGKILL survivors). Scoped by GPU + uid so other GPUs (e.g. Arm B) are never touched."""
    import subprocess as _sp
    me = os.getuid()
    out = _sp.run(["bash", "-lc",
        f"nvidia-smi --query-compute-apps=pid --format=csv,noheader -i {gpus} 2>/dev/null"],
        capture_output=True, text=True).stdout
    mine = [p.strip() for p in out.split() if p.strip().isdigit() and _safe_uid(p.strip()) == me]
    _graceful_kill(mine)


def _serve_and_rollout(checkpoint, cfg, passes, seed):
    """Serve the checkpoint on vLLM (GPUs from cfg), run each pass = (label, skill_store,
    tasks, n) on freshly-built TRAIN managers, return {label: transcripts}. GPU-only."""
    from agent_system.skill_opt.actor import ActorClient
    from agent_system.skill_opt.envs import build_balanced_managers, attach_skill
    from agent_system.skill_opt.rollout import run_balanced_rollout
    import time
    hf = _ensure_hf(checkpoint, cfg)
    # NOTE: TP=2 on purpose, even when the arm owns 4 GPUs. Measured on this workload the
    # server runs `Waiting: 0` with KV-cache under 0.5% and sits fully idle in over half of
    # vLLM's own throughput samples: the rollout is lockstep (generate a batch -> step every
    # ALFWorld env in Ray -> repeat), so inference is never the bottleneck and extra GPUs or
    # replicas buy nothing. What DID bind was ActorClient's thread pool (see rollout_workers).
    vl = subprocess.Popen(["bash", "-lc",
        f"CUDA_VISIBLE_DEVICES={cfg['gpus']} VLLM_ATTENTION_BACKEND=FLASH_ATTN "
        f"PATH={os.path.dirname(VENV_PY)}:$PATH {os.path.dirname(VENV_PY)}/vllm serve {hf} "
        f"--served-model-name actor --tensor-parallel-size {int(cfg.get('tp_size', 2))} "
        f"--gpu-memory-utilization 0.85 "
        f"--max-model-len 8192 --enforce-eager --host 127.0.0.1 --port {cfg.get('vllm_port', 8110)}"],
        stdout=open(f"{cfg['log_dir']}/measure_vllm.log", "w"), stderr=subprocess.STDOUT,
        start_new_session=True)
    out = {}
    mgrs = []
    try:
        actor = ActorClient(base_url=f"http://127.0.0.1:{cfg.get('vllm_port', 8110)}/v1",
                            temperature=0.4, top_p=1.0, top_k=-1,
                            # 6 tasks x n_per_task envs step in lockstep; at the default 48
                            # each round needs ceil(envs/48) sequential waves while the GPU
                            # idles between them. Fire the whole round at once.
                            max_workers=int(cfg.get("rollout_workers", 192)))
        deadline = cfg.get("vllm_health_timeout", VLLM_HEALTH_TIMEOUT)
        t0 = time.time()
        while time.time() - t0 < deadline and not actor.healthy():
            if vl.poll() is not None:          # serve died -> stop waiting out the full deadline
                raise RuntimeError(f"vLLM serve exited early (rc={vl.returncode}); "
                                   f"see {cfg['log_dir']}/measure_vllm.log")
            time.sleep(10)
        if not actor.healthy():
            raise RuntimeError(f"vLLM did not become healthy within {deadline}s; "
                               f"see {cfg['log_dir']}/measure_vllm.log")
        # Build managers ONCE. All passes share the same tasks/n by construction, so reusing
        # them gives the SAME games across passes (paired A/B; repeated games form real groups)
        # AND avoids re-scanning the full ~8810-game train dir on every pass.
        tasks0, n0 = passes[0][2], passes[0][3]
        mgrs = build_balanced_managers(n0, seed=seed, is_train=True, only_types=tasks0)
        for label, skill_store, _tasks, _n in passes:
            for _nm, m in mgrs:
                # Rewind the game order BEFORE every arm. Without this the arms are not paired:
                # TextWorld's reset() advances to the next games in the pool, so arm 2 would play
                # a disjoint set from arm 1 and ab_gate would be comparing two independent
                # samples. At n=30/task that difference is ~0.05 while the gate accepts on any
                # margin at all, i.e. the content gate would be deciding by draw luck.
                m.envs.reseed()
                attach_skill(m, skill_store)
            out[label] = run_balanced_rollout(mgrs, actor, max_steps=50, temperature=0.4)
    finally:
        import signal
        for _n, m in mgrs:                    # 1) tear down the ALFWorld env Ray actors
            try:
                m.envs.close()
            except Exception:
                pass
        try:                                  # 2) shut down the local Ray instance we started
            import ray
            if ray.is_initialized():
                ray.shutdown()
        except Exception:
            pass
        try:                                  # 3) GRACEFULLY stop the vLLM server group:
            pgid = os.getpgid(vl.pid)         #    SIGTERM -> wait -> SIGKILL only if it hangs
            os.killpg(pgid, signal.SIGTERM)
            t0 = time.time()
            while time.time() - t0 < 20 and _alive(vl.pid):
                time.sleep(1)
            if _alive(vl.pid):
                os.killpg(pgid, signal.SIGKILL)
        except Exception:
            pass
        try:
            vl.wait(timeout=5)                # reap the vLLM child so it doesn't linger as a zombie
        except Exception:
            pass
        _kill_on_gpus(cfg["gpus"])            # 4) graceful sweep of any leftover TP workers on MY gpus
    return out


def measure_ab_adapter(checkpoint, current_scaffold, candidate_scaffold, tasks, cfg, seed):
    """Frozen-policy paired 3-way A/B on TRAIN over the touched tasks. GPU-only."""
    from agent_system.skill.skill_store import SkillStore
    n = cfg.get("n_per_task", N_PER_TASK)
    from .scaffold import injects_nothing
    passes = [("bare", SkillStore(mode="none"), tasks, n),
              ("candidate", SkillStore(**_ss_kwargs(candidate_scaffold)), tasks, n)]
    # An empty current scaffold renders to "" and splices to the identical prompt, so a
    # separate 'current' arm would just be a second, independently-noisy sample of the SAME
    # condition as 'bare'. Measured once at n=180 those two arms came out 0.917 vs 0.867 —
    # a 0.05 spread on a difference that is exactly zero by construction, which is 3x the
    # 0.016 margin the gate then ruled on. Reuse the bare arm instead of paying for that.
    same_as_bare = injects_nothing(current_scaffold)
    if not same_as_bare:
        passes.insert(1, ("current", SkillStore(**_ss_kwargs(current_scaffold)), tasks, n))
    tr = _serve_and_rollout(checkpoint, cfg, passes, seed)
    return assemble_measure(tr["bare"], tr["bare"] if same_as_bare else tr["current"],
                            tr["candidate"], tasks)


def signals_adapter(checkpoint, scaffold, cfg, seed):
    """Train-side signals for the Teacher: per_task_gap (bare vs current scaffold on train)
    and the failed bare trajectories, over all 6 tasks. GPU-only.

    all_fail_groups IS produced here (this docstring used to say it was not). The G bare passes
    run on the SAME games under the same seed, so grouping them by gamefile reconstructs real
    GiGPO-shaped groups — a game the frozen policy fails in all G attempts is a group that would
    contribute no gradient. It is measured on the frozen checkpoint rather than read out of the
    live trainer, so it describes the policy the Teacher is about to write for.
    """
    from agent_system.skill.skill_store import SkillStore
    from .scaffold import TASKS
    n = cfg.get("n_per_task", N_PER_TASK)
    G = cfg.get("group_n", GROUP_N)
    # G stochastic bare passes on the SAME train games (same seed) -> real groups per game;
    # one injected pass for the gap. Grouping keys on gamefile (robust).
    passes = [(f"bare_{k}", SkillStore(mode="none"), TASKS, n) for k in range(G)]
    passes.append(("injected", SkillStore(**_ss_kwargs(scaffold)), TASKS, n))
    tr = _serve_and_rollout(checkpoint, cfg, passes, seed)
    bare_flat = [t for k in range(G) for t in tr[f"bare_{k}"]]
    groups = group_by_gamefile(bare_flat)
    return assemble_signals(bare_flat, tr["injected"], groups, TASKS)


def _ss_kwargs(scaffold):
    return {"skills": scaffold.get("skills", {}), "general_skill": scaffold.get("general_skill", ""),
            "p_task": scaffold.get("p_task", {}), "mode": "full"}
