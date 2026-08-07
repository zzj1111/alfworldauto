"""Entry point: wire the adapters into the loop and run it.

    python -m autoscaffold.run_arm [n_cycles]

Resumes from state.json when present. ARM_TARGET_STEP is the absolute finish line and
survives restarts; n_cycles only caps this process.
"""
from __future__ import annotations

import json
import os
import sys

from . import config as C
from . import gate  # noqa: F401  (re-exported for tooling)
from . import loop as L
from . import monitor as M
from . import prompts as P
from . import runner as R
from . import scaffold as S
from . import signals as G
from . import teacher as T


def default_cfg():
    exp = C.exp_name()
    state_dir = os.path.join(C.exp_root(), exp)
    os.makedirs(state_dir, exist_ok=True)
    return {
        "exp": exp,
        "state_dir": state_dir,
        "ckpt_dir": os.path.join(C.ckpt_root(), exp),
        "scaffold_path": os.path.join(state_dir, "scaffold.json"),
        "state_path": os.path.join(state_dir, "state.json"),
        "journal_path": os.path.join(state_dir, "journal.json"),
        "rollout_log": os.path.join(state_dir, "rollouts.jsonl"),
        "train_log": C.stamped(os.path.join(state_dir, "train.log")),
        "python": os.environ.get("ARM_PYTHON") or sys.executable,
        "model": C.model_path(),
        "gpus": os.environ.get("ARM_GPUS", "0,1"),
        "tp": int(os.environ.get("ARM_TP", "2")),
        "steps_per_cycle": int(os.environ.get("ARM_K", "10")),
        "val_n": int(os.environ.get("ARM_VAL_N", "3")),
        "rollout_n": int(os.environ.get("ARM_ROLLOUT_N", "8")),
        "target_step": int(os.environ.get("ARM_TARGET_STEP", "0") or 0),
        "eval_split": os.environ.get("ARM_EVAL_SPLIT", "eval_in_distribution"),
        "ab_episodes": int(os.environ.get("ARM_AB_EPISODES", "180")),
        "vllm_port": int(os.environ.get("ARM_VLLM_PORT", "8110")),
        "vllm_health_timeout": int(os.environ.get("ARM_VLLM_HEALTH_TIMEOUT", "2400")),
        "base_seed": 20260722,
    }


def _wandb_run(cfg):
    if os.environ.get("ARM_WANDB", "0") != "1":
        return None
    try:
        import wandb
        return wandb.init(project=os.environ.get("WANDB_PROJECT"),
                          entity=os.environ.get("WANDB_ENTITY") or None,
                          id=os.environ.get("WANDB_RUN_ID"), resume="allow",
                          name=cfg["exp"], reinit=True)
    except Exception as e:
        print(f"[wandb] orchestrator metrics disabled: {type(e).__name__}: {e}")
        return None


def build_fns(cfg):
    orch_all = os.path.join(cfg["state_dir"], "orch.log")
    orch_run = C.stamped(orch_all)

    def log(msg):
        line = f"{msg}\n"
        for path in (orch_all, orch_run):
            try:
                with open(path, "a") as f:
                    f.write(line)
            except OSError:
                pass
        print(msg, flush=True)

    cfg["log"] = log
    wb = _wandb_run(cfg)
    system_prompt = P.render_system_prompt()

    def train_fn(scaffold, from_step, to_step):
        S.persist(scaffold, cfg["scaffold_path"])       # training hot-reloads this
        return R.train_adapter(cfg["scaffold_path"], to_step, cfg)

    def teacher_fn(obs, scaffold):
        return T.propose(system_prompt, P.render_user_prompt(obs), scaffold)

    return {
        "log": log,
        "train_fn": train_fn,
        "eval_fn": lambda ckpt: R.eval_adapter(ckpt, cfg),
        "signals_fn": lambda ckpt, scaf: _signals(ckpt, scaf, cfg),
        "teacher_fn": teacher_fn,
        "measure_ab_fn": lambda ckpt, cur, cand, tasks:
            R.measure_ab_adapter(ckpt, cur, cand, tasks, cfg),
        "persist_fn": lambda scaf: S.persist(scaf, cfg["scaffold_path"]),
        "state_fn": lambda state: S.persist(
            {k: state[k] for k in L.STATE_KEYS if k in state}, cfg["state_path"]),
        "journal_fn": lambda hist: S.persist(hist, cfg["journal_path"]),
        "snapshot_fn": lambda state: M.publish(state, cfg, wb),
    }


def _signals(ckpt, scaffold, cfg):
    sig = R.signals_adapter(ckpt, scaffold, cfg)
    cfg["_last_zero_gradient"] = sig.get("zero_gradient_groups") or {}
    return sig


def load_state(cfg):
    try:
        with open(cfg["state_path"]) as f:
            d = json.load(f)
    except (OSError, ValueError):
        return None
    return d if isinstance(d, dict) and "step" in d else None


def main(n_cycles=30):
    cfg = default_cfg()
    fns = build_fns(cfg)
    log = fns["log"]
    saved = load_state(cfg)
    if saved:
        state = {**L.new_state(), **saved}
        log(f"[resume] step={state['step']} cycle={state['cycle']} "
            f"decisions={len(state['decision_history'])}")
    else:
        step0 = 0
        # seed from an existing checkpoint when the state file is gone but training
        # artifacts survive (a wiped exp dir must not silently restart from base)
        steps = sorted(R.step_of(p) for p in
                       __import__("glob").glob(os.path.join(cfg["ckpt_dir"], "global_step_*"))
                       if R.ckpt_is_usable(p))
        if steps:
            step0 = steps[-1]
            log(f"[resume] no state.json but a usable checkpoint at step {step0}; "
                f"seeding from it (scaffold and Teacher memory start empty)")
        state = L.new_state(step0)
        S.persist(state["scaffold"], cfg["scaffold_path"])
    log(f"[autoscaffold] exp={cfg['exp']} K={cfg['steps_per_cycle']} "
        f"val_n={cfg['val_n']} rollout_n={cfg['rollout_n']} "
        f"target={cfg['target_step'] or '(n_cycles only)'} gpus={cfg['gpus']}")
    L.run(state, fns, cfg, n_cycles)
    log(f"[autoscaffold] done at step {state['step']}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 30)
