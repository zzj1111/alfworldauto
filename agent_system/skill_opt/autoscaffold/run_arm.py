"""Entry point: assemble the harness into one runnable arm and run the loop.

Default config = the scaffold + KL=0 arm on GPUs 0,1 (the KL ablation of Arm B),
starting from base Qwen2.5-1.5B-Instruct with an EMPTY scaffold (真空).

This module builds everything and exposes main(); it does NOT auto-launch on import.
A real launch must be gated on a load check + explicit go (server-incident rule).
Per-cycle fresh A/B games come from seed = base_seed + train_step (derived from the
checkpoint path), so no cross-cycle game reuse.
"""
from __future__ import annotations

import copy
import os

from . import adapters as A
from . import scaffold as S
from . import teacher as T
from . import loop as L


def default_cfg():
    exp = os.environ.get("ARM_EXP", "alf_scaffold_kl0")
    root = f"/mnt/data1/zha00175/exp_autoscaffold/{exp}"
    os.makedirs(root, exist_ok=True)
    return {
        "exp": exp,
        "gpus": os.environ.get("ARM_GPUS", "0,1"),
        "model": os.environ.get("ARM_MODEL", "/dev/shm/qwen15b"),  # staged on /dev/shm (anti I/O-storm)
        "ray_tmp": os.environ.get("ARM_RAY_TMP", f"/dev/shm/zray_{exp}"),  # per-exp -> no cross-run collision
        "log_dir": "/mnt/data1/zha00175/gigpo_helper_logs",
        "scaffold_path": f"{root}/scaffold.json",
        "journal_path": f"{root}/journal.json",
        "state_path": f"{root}/state.json",
        "train_log": f"{root}/train.log",
        # KL loss vs the frozen reference. The base script already sets use_kl_loss=True with
        # coef 0.01; ARM_KL overrides the coefficient, and ARM_KL=0 restores the DAPO-style
        # no-KL ablation. With bare_prompt_loss on, the reference log-probs are computed after
        # the prompt swap, so the penalty is KL(pi_theta(.|x) || pi_ref(.|x)) — same condition
        # on both sides.
        "train_extra": os.environ.get(
            "ARM_TRAIN_EXTRA",
            ("actor_rollout_ref.actor.use_kl_loss=False" if os.environ.get("ARM_KL", "0.01") == "0"
             else f"actor_rollout_ref.actor.use_kl_loss=True "
                  f"actor_rollout_ref.actor.kl_loss_coef={os.environ.get('ARM_KL', '0.01')} "
                  f"actor_rollout_ref.actor.kl_loss_type=low_var_kl "
                  f"algorithm.bare_prompt_loss.enable={os.environ.get('ARM_BARE_LOSS', 'True')} "
                  f"algorithm.bare_prompt_loss.mode={os.environ.get('ARM_BARE_MODE', 'both')}")),
        # loop knobs (all the locked decisions)
        "steps_per_cycle": int(os.environ.get("ARM_K", "10")),
        "val_n": int(os.environ.get("ARM_VAL_N", "3")),
        "n_per_task": int(os.environ.get("ARM_NPT", "30")),
        "group_n": int(os.environ.get("ARM_GROUP_N", "8")),   # rollouts/game for all-fail groups
        "train_data_size": int(os.environ.get("ARM_TRAIN_DATA_SIZE", "32")),
        "n_gpus": int(os.environ.get("ARM_N_GPUS", "2")),
        "tp_size": int(os.environ.get("ARM_TP", "2")),
        "gpu_mem": float(os.environ.get("ARM_GPU_MEM", "0.6")),
        "val_bs": int(os.environ.get("ARM_VAL_BS", "128")),
        # Calibration from earlier runs fed to the Teacher. OFF by default: with it off,
        # whatever the Teacher works out (withdrawal, all-fail leverage) it worked out from
        # the signals, which is itself a result. ARM_PRIORS=1 turns it on so the two are
        # runnable as an ablation rather than a silent config change.
        "teacher_priors": os.environ.get("ARM_PRIORS", "0") == "1",
        # How many cycles the scaffold may stay empty before the pre-check is bypassed. Measured:
        # the one arm run WITH triage declined 17 of 20 cycles and accepted nothing, while every
        # arm run without it wrote scaffolds. Set to 0 to disable and restore pure discretion.
        "intervene_floor_cycles": int(os.environ.get("ALF_FLOOR_CYCLES", "3")),
        "n_cycles": int(os.environ.get("ALF_N_CYCLES", "20")),
        "domain": S.ALF_DOMAIN,
        "base_seed": 20260722,
        "vllm_port": int(os.environ.get("ARM_VLLM_PORT", "8110")),
    }


def _step_of(checkpoint):
    try:
        return int(str(checkpoint).split("global_step_")[-1].split("/")[0])
    except Exception:
        return 0


# Everything the loop needs to resume. decision_history doubles as Teacher memory.
STATE_KEYS = ("cycle", "step", "scaffold", "sr_history", "best", "best_step",
              "decision_history", "last_eval")


def load_state(cfg):
    """Loop state from a previous run of this arm, or None for a cold start."""
    import json
    try:
        with open(cfg["state_path"]) as f:
            d = json.load(f)
    except Exception:
        return None
    return d if isinstance(d, dict) and "step" in d else None


def load_journal(cfg):
    """Prior decision history (Teacher memory) from a previous run of this arm, or []."""
    import json
    try:
        with open(cfg["journal_path"]) as f:
            d = json.load(f)
        return d if isinstance(d, list) else []
    except Exception:
        return []


def build_fns(cfg):
    """Wire the GPU/API adapters into the callables loop.run expects."""
    def train_fn(scaf, frm, to):
        A.persist_scaffold(scaf, cfg["scaffold_path"])         # training hot-reloads this file
        return A.train_adapter(cfg["scaffold_path"], frm, to, cfg)   # .../global_step_{to}

    def eval_fn(ckpt, val_n):
        return A.eval_adapter(ckpt, val_n, cfg)

    def signals_fn(ckpt, scaf):
        return A.signals_adapter(ckpt, scaf, cfg, seed=cfg["base_seed"] + _step_of(ckpt))

    def measure_ab_fn(ckpt, cur, cand, tasks):
        return A.measure_ab_adapter(ckpt, cur, cand, tasks, cfg, seed=cfg["base_seed"] + _step_of(ckpt))

    def teacher_fn(obs):
        return T.propose(obs, call_fn=T.openai_call, domain=cfg.get("domain"),
                         priors=cfg.get("teacher_priors", False))

    def triage_fn(obs):
        # Cheap intervene/decline question asked before signals_fn is paid for. Same model, a
        # much smaller prompt, and no GPU work behind it.
        return T.triage(obs, call_fn=T.openai_call)

    def persist_fn(scaf):
        A.persist_scaffold(scaf, cfg["scaffold_path"])

    def state_fn(state):
        A.persist_scaffold({k: state[k] for k in STATE_KEYS if k in state}, cfg["state_path"])

    def journal_fn(decision_history):
        # rewritten whole each cycle (small) so backfilled sr_after is captured; atomic
        A.persist_scaffold(decision_history, cfg["journal_path"])

    def log(msg):
        with open(f"{os.path.dirname(cfg['scaffold_path'])}/orch.log", "a") as f:
            f.write(msg + "\n")
        print(msg, flush=True)

    # Adapters log through cfg (they do not get the fns dict), e.g. a partial eval warning.
    cfg["log"] = log

    return {"train_fn": train_fn, "eval_fn": eval_fn, "signals_fn": signals_fn,
            "measure_ab_fn": measure_ab_fn, "teacher_fn": teacher_fn, "triage_fn": triage_fn,
            "persist_fn": persist_fn,
            "journal_fn": journal_fn, "state_fn": state_fn, "log": log}


def main(n_cycles=30):
    # Also covers ray.init() calls made IN THIS PROCESS (the measurement rollouts build
    # ALFWorld managers in-process); _run() sets it for the training/eval subprocesses.
    os.environ.setdefault("RAY_agent_register_timeout_ms", A.RAY_AGENT_REGISTER_TIMEOUT_MS)
    os.environ.setdefault("RAY_worker_register_timeout_seconds", A.RAY_WORKER_REGISTER_TIMEOUT_S)
    cfg = default_cfg()

    # Keep the IN-PROCESS ray.init() off the root filesystem. ARM_RAY_TMP only reaches the
    # training/eval subprocesses (via _run); the measurement rollouts call ray.init() in this
    # process, which defaults to /tmp/ray and leaves a 40-130 MB session dir behind every time.
    # Two measurement phases per cycle over a long arm is several GB on a partition that is
    # shared with every other user's /tmp and has repeatedly sat at 100% full — and a full root
    # disk is what killed attempt 2 of alf_scratch150_pcap at step 130.
    os.environ.setdefault("RAY_TMPDIR", cfg["ray_tmp"])
    os.makedirs(cfg["ray_tmp"], exist_ok=True)

    fns = build_fns(cfg)
    saved = load_state(cfg)
    if saved:
        # RESUME. Do not re-persist an empty scaffold here: scaffold.json on disk is the live
        # one training hot-reloads, and the resumed state already carries it.
        state = {**L.new_state(), **saved}
        # A scaffold saved before the cap existed (or by an older build) can carry p above P_MAX.
        # Clamp on load so the ceiling applies to the run in progress, not only to future edits.
        before = dict((state.get("scaffold") or {}).get("p_task") or {})
        state["scaffold"] = S.clamp_p(state["scaffold"])
        after = (state["scaffold"].get("p_task") or {})
        moved = {t: (before[t], after[t]) for t in before if before[t] != after.get(t)}
        if moved:
            fns["log"](f"[autoscaffold] clamped p to cap {S.P_MAX}: "
                       + ", ".join(f"{t} {a}->{b}" for t, (a, b) in moved.items()))
            A.persist_scaffold(state["scaffold"], cfg["scaffold_path"])
        fns["log"](f"[autoscaffold] RESUME arm={cfg['exp']} at cycle {state['cycle']} step={state['step']} "
                   f"scaffold v{state['scaffold'].get('version')} best={state['best']}@{state['best_step']} "
                   f"sr_history={state['sr_history']}")
    else:
        # Cold start, but the step counter must begin where the weights actually are: this arm
        # may be seeded with a checkpoint from a previous run (e.g. continuing a converged
        # policy). Scaffold and Teacher memory still start empty — only the step is inherited.
        ckpt_root = f"/mnt/data1/zha00175/gigpo_helper_ckpts/{cfg['exp']}"
        step0 = A.existing_ckpt_step(ckpt_root)
        state = L.new_state(step0=step0, scaffold=S.empty_scaffold())   # 真空 cold start
        state["best_step"] = step0
        if step0:
            fns["log"](f"[autoscaffold] seeded from existing checkpoint at step {step0} "
                       f"(scaffold and Teacher memory still start empty)")
        prior = load_journal(cfg)                             # Teacher memory across restarts
        if prior:
            state["decision_history"] = prior
            fns["log"](f"[autoscaffold] loaded {len(prior)} prior decisions as Teacher memory")
        A.persist_scaffold(state["scaffold"], cfg["scaffold_path"])
        fns["log"](f"[autoscaffold] arm={cfg['exp']} gpus={cfg['gpus']} KL0={'use_kl_loss=False' in cfg['train_extra']} "
                   f"empty scaffold, K={cfg['steps_per_cycle']}, VAL_N={cfg['val_n']}")

    # ARM_TARGET_STEP makes the finish line ABSOLUTE. Without it the only budget is n_cycles,
    # which counts cycles in THIS process — so a watchdog restart part-way through resumes at,
    # say, step 100 and then runs a further n_cycles, overshooting the intended total. The
    # watchdog's own TARGET_STEP only gates whether to RELAUNCH, so it notices an overshoot after
    # the fact. Set both to the same number; n_cycles then merely caps one process's work.
    target = int(os.environ.get("ARM_TARGET_STEP", "0") or 0)
    if target:
        cfg["stop_fn"] = lambda st: st.get("step", 0) >= target
        fns["log"](f"[autoscaffold] absolute target step {target} "
                   f"(n_cycles={n_cycles} caps this process only)")

    return L.run(state, fns, cfg, n_cycles)


if __name__ == "__main__":
    import sys
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 30)
