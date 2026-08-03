"""Reconstruct a W&B run for an auto-scaffold arm from its on-disk logs.

The arm trains with `trainer.logger=[console]`, so nothing was streamed to W&B live.
Everything is recovered from files the harness already writes:

  train.log                per-training-step verl metrics (`step:N - k:v - k:v ...` lines)
  orch.log                 per-cycle harness events (valid_seen, triage, teacher action, A/B)
  journal.json             the Teacher's record: diagnosis, proposed text, A/B numbers, outcome
  <log_dir>/*_eval_s*_d*.log   the standalone eval draws (per-task val rates)

NAMING (important): keys are verl's NATIVE names with NO extra prefix, and the x-axis is
`training/global_step`, so this run overlays directly on the existing runs in the project
(e.g. 0621_1310_gigpo_Qwen2.5-1.5B-Instruct_full_g8_b16_lr1e-6). In particular
`val/success_rate` is the STANDALONE (no-scaffold) eval — the same measurement those runs
report — so the curves are directly comparable.

Harness-specific series live in namespaces verl never uses (`scaffold/`, `ab/`, `teacher/`)
so they cannot collide with the native keys.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re

ENTITY = "mhong-university-of-minnesota"
PROJECT = "verl_agent_alfworld_inspect"
LOG_DIR = "/mnt/data1/zha00175/gigpo_helper_logs"

TASKS = ["pick_and_place", "pick_two_obj_and_place", "look_at_obj_in_light",
         "pick_heat_then_place_in_recep", "pick_cool_then_place_in_recep",
         "pick_clean_then_place_in_recep"]

_ANSI = re.compile(r"\x1b\[[0-9;]*m")
_STEP = re.compile(r"step:(\d+) ")
_KV = re.compile(r"([a-zA-Z][\w/]*):(-?[0-9.]+)")


def parse_train_log(path):
    """{step: {native verl metric: value}} — last write per step wins."""
    out = {}
    with open(path, errors="ignore") as f:
        for line in f:
            if "step:" not in line or "episode/success_rate" not in line:
                continue
            line = _ANSI.sub("", line)
            m = _STEP.search(line)
            if not m:
                continue
            metrics = {}
            for k, v in _KV.findall(line):
                if k == "step":
                    continue
                try:
                    metrics[k] = float(v)
                except ValueError:
                    pass
            if metrics:
                out[int(m.group(1))] = metrics
    return out


def parse_orch_log(path):
    """Per-cycle harness events, keyed by cycle number."""
    cycles = {}
    with open(path, errors="ignore") as f:
        for line in f:
            # The priming eval is logged as [baseline], not [cN]: it is the standalone number
            # for the checkpoint the arm starts from, i.e. cycle 0's anchor. Dropping it would
            # leave a hole exactly where the scaffold phase begins.
            b = re.match(r"\[baseline\] step(\d+) .*?avg=([0-9.]+) draws=\[([^\]]*)\]", line.strip())
            if b:
                c = cycles.setdefault(0, {"cycle": 0})
                c["step"] = int(b.group(1))
                c["valid_seen"] = float(b.group(2))
                c["draws"] = [float(x) for x in b.group(3).split(",") if x.strip()]
                continue
            m = re.match(r"\[c(\d+)\] (.*)", line.strip())
            if not m:
                continue
            cyc, rest = int(m.group(1)), m.group(2)
            c = cycles.setdefault(cyc, {"cycle": cyc})
            ev = re.match(r"step=(\d+) valid_seen avg=([0-9.]+) draws=\[([^\]]*)\]", rest)
            if ev:
                c["step"] = int(ev.group(1))
                c["valid_seen"] = float(ev.group(2))
                c["draws"] = [float(x) for x in ev.group(3).split(",") if x.strip()]
            elif rest.startswith("new best"):
                c["new_best"] = True
            elif rest.startswith("teacher:"):
                c["teacher_edits"] = re.findall(r"'([a-z_]+)'", rest.split("edits=")[-1].split("p=")[0])
                pm = re.search(r"p=(\{.*\})", rest)
                c["teacher_p"] = pm.group(1) if pm else "{}"
            elif rest.startswith("A/B:"):
                c["ab_reason"] = rest[4:].strip()
                n = re.search(r"candidate ([0-9.]+) [<>=]+ current ([0-9.]+) \(bare ([0-9.]+)\)", rest)
                if n:
                    c["ab_candidate"], c["ab_current"], c["ab_bare"] = (float(n.group(i)) for i in (1, 2, 3))
                c["ab_accept"] = "ACCEPT" in rest
            elif rest.startswith("triage:"):
                # cycles the Teacher declined to measure; no signals/AB were paid for
                verdict = rest[len("triage:"):].strip()
                c["triage"] = verdict
                c["triage_skipped"] = verdict.startswith("SKIP")
    return [cycles[k] for k in sorted(cycles)]


def per_task_val(exp, step, log_dir=LOG_DIR):
    """Mean per-task standalone success across this step's eval draws.

    The harness logs `val/<task>_success_rate` inside each draw's log; averaging across draws
    matches how `val/success_rate` is reported (mean over VAL_N draws)."""
    acc = {}
    for path in sorted(glob.glob(f"{log_dir}/{exp}_eval_s{step}_d*.log")):
        txt = open(path, errors="ignore").read()
        for t in TASKS:
            v = re.findall(rf"val/{t}_success_rate[:'\"\s]+\(?(?:np\.float64\()?([0-9.]+)", txt)
            if v:
                acc.setdefault(t, []).append(float(v[-1]))
    return {t: sum(v) / len(v) for t, v in acc.items() if v}


def _cfg(arm_dir):
    """verl-shaped config so this run filters/compares alongside the native runs."""
    return {
        "algorithm": {"adv_estimator": "gigpo", "gamma": 0.95,
                      "gigpo": {"mode": "mean_std_norm", "step_advantage_w": 1.0},
                      "use_kl_in_reward": False,
                      # Rollout under the scaffolded prompt g(x); log-probs (old, ref and the
                      # actor update) conditioned on the BARE prompt x, so the policy is
                      # optimised on the prompt it actually faces at evaluation.
                      "bare_prompt_loss": {"enable": True, "mode": "both"}},
        "actor_rollout_ref": {
            "model": {"path": "Qwen/Qwen2.5-1.5B-Instruct"},
            "actor": {"optim": {"lr": 1e-6}, "use_kl_loss": True, "kl_loss_coef": 0.01,
                      "kl_loss_type": "low_var_kl", "ppo_mini_batch_size": 256,
                      "use_invalid_action_penalty": True,
                      "fsdp_config": {"param_offload": True, "optimizer_offload": True}},
            "rollout": {"name": "vllm", "tensor_model_parallel_size": 2,
                        "gpu_memory_utilization": 0.35}},
        "data": {"train_batch_size": 32, "val_batch_size": 64},
        "env": {"env_name": "alfworld/AlfredTWEnv", "max_steps": 50, "seed": 0,
                "rollout": {"n": 8}, "alfworld": {"eval_dataset": "eval_in_distribution"}},
        "trainer": {"logger": ["console"], "n_gpus_per_node": 4, "save_freq": 10,
                    "experiment_name": os.path.basename(arm_dir.rstrip("/"))},
        # harness-specific (not part of verl's config)
        "autoscaffold": {
            "teacher": "gpt-5.5", "steps_per_cycle": 10, "val_n": 3, "n_per_task": 30,
            "group_n": 8, "initial_scaffold": "empty",
            "ab_rule": "strict > current, no margin, no retries",
            "revert_gate": "none (removed 2026-07-29)",
            "triage": "Teacher declines cheaply before the signals pass is paid for",
            "injection": "TRAINING prompts only; eval is always standalone"},
    }


def seed_history(run_path, max_step):
    """Steps 1..max_step pulled from an EXISTING W&B run (the pure-RL run this arm continues).

    The arm starts at a checkpoint produced by that run, so its own logs begin mid-curve. Without
    the seed the uploaded curve would start at step 151 and look like a separate experiment
    instead of a continuation. Keys come back with verl's native names, which is exactly what
    this module already logs, so the two halves line up on `training/global_step`.
    """
    import wandb
    api = wandb.Api(timeout=120)
    run = api.run(run_path)
    out = {}
    for row in run.scan_history():
        gs = row.get("training/global_step")
        if gs is None:
            continue
        gs = int(gs)
        if gs > max_step:
            continue
        out[gs] = {k: v for k, v in row.items()
                   if k and not k.startswith("_") and isinstance(v, (int, float)) and v is not None}
    return out


def merge_segments(*segments):
    """Later segments win on collision. Each segment is (dict{step: metrics}, lo, hi)."""
    out = {}
    for seg, lo, hi in segments:
        for step, m in seg.items():
            if lo <= step <= hi:
                out[step] = dict(m)
    return out


def build(arm_dir, run_id, run_name, project, entity, offline, log_dir=LOG_DIR,
          seed_run=None, seed_max=150, extra_logs=()):
    import wandb

    exp = os.path.basename(arm_dir.rstrip("/"))
    train = parse_train_log(os.path.join(arm_dir, "train.log"))
    # Prepend the run this arm continues from, then any extra local segments (e.g. the
    # empty-scaffold stretch that produced the checkpoint the scaffold phase starts at).
    # Ranges are explicit because one log can hold steps from an abandoned branch too.
    segments = []
    if seed_run:
        segments.append((seed_history(seed_run, seed_max), 1, seed_max))
    for path, lo, hi in extra_logs:
        segments.append((parse_train_log(path), lo, hi))
    segments.append((train, 1, 10**9))
    train = merge_segments(*segments)
    cycles = parse_orch_log(os.path.join(arm_dir, "orch.log"))
    journal = json.load(open(os.path.join(arm_dir, "journal.json")))
    scaffold = json.load(open(os.path.join(arm_dir, "scaffold.json")))
    by_cycle = {e.get("cycle"): e for e in journal if isinstance(e, dict)}
    cyc_at_step = {c["step"]: c for c in cycles if "step" in c}

    if offline:
        os.environ["WANDB_MODE"] = "offline"
    run = wandb.init(
        entity=entity, project=project, id=run_id, resume="allow", name=run_name,
        notes="Teacher-Student auto-scaffold, continuing the pure-RL run 0621_1310 from its "
              "global_step_150. Steps 1-150 are that run's history; 151-160 continue it with an "
              "EMPTY scaffold (the control); 161+ inject a GPT-5.5-written scaffold into TRAINING "
              "prompts only. val/success_rate is always STANDALONE (no scaffold), so it stays "
              "comparable to every other run here. scaffold/injected_frac is the measured share of "
              "training prompts that actually differ from their bare counterpart, and "
              "scaffold/phase marks the three regimes (0=pure RL, 1=inject all 6 categories, "
              "2=Teacher withdrew 4 of them). NOTE: actor/kl_loss jumps at 150->151 because the "
              "reference model is re-anchored on the step-150 policy at the restart -- it is a "
              "change of reference, not of behaviour. Reconstructed from console logs.",
        tags=["gigpo", "alfworld", "qwen2.5-1.5b", "auto-scaffold", "bare-prompt-loss",
              "from-gs150", "reconstructed"],
        config=_cfg(arm_dir))

    wandb.define_metric("training/global_step")
    wandb.define_metric("*", step_metric="training/global_step")

    for step in sorted(train):
        payload = dict(train[step])                     # native verl keys, no prefix
        payload["training/global_step"] = step
        # Which regime produced this step, so the three stretches are separable on the plot.
        payload["scaffold/injected_frac"] = (
            payload["bare_loss/n_changed"] / payload["bare_loss/n_swapped"]
            if payload.get("bare_loss/n_swapped") else 0.0)
        payload["scaffold/phase"] = (0.0 if step <= 160 else
                                     1.0 if step <= 170 else 2.0)
        c = cyc_at_step.get(step)
        if c and "valid_seen" in c:
            payload["val/success_rate"] = c["valid_seen"]          # STANDALONE — the headline
            for t, v in per_task_val(exp, step, log_dir).items():
                payload[f"val/{t}_success_rate"] = v
            for i, d in enumerate(c.get("draws", [])):
                payload[f"val/draw{i}_success_rate"] = d
        if c:
            if "ab_candidate" in c:
                payload["ab/candidate"] = c["ab_candidate"]
                payload["ab/current"] = c["ab_current"]
                payload["ab/bare"] = c["ab_bare"]
                payload["ab/gain_over_bare"] = round(c["ab_current"] - c["ab_bare"], 4)
                payload["ab/accepted"] = 1.0 if c.get("ab_accept") else 0.0
            payload["scaffold/n_text_edits"] = len(c.get("teacher_edits", []))
            payload["scaffold/cycle"] = c["cycle"]
        run.log(payload)

    tbl = wandb.Table(columns=["cycle", "global_step", "val/success_rate", "delta",
                               "text_edits", "p_edits", "ab_bare", "ab_current",
                               "ab_candidate", "verdict", "diagnosis"])
    prev = None
    for c in cycles:
        j = by_cycle.get(c["cycle"], {})
        vs = c.get("valid_seen")
        tbl.add_data(c["cycle"], c.get("step"), vs,
                     None if (vs is None or prev is None) else round(vs - prev, 4),
                     ", ".join(c.get("teacher_edits", [])) or "(none)", c.get("teacher_p", "{}"),
                     c.get("ab_bare"), c.get("ab_current"), c.get("ab_candidate"),
                     j.get("verdict") or ("accepted" if c.get("ab_accept") else ""),
                     (j.get("summary") or {}).get("diagnosis", ""))
        if vs is not None:
            prev = vs
    run.log({"teacher/decisions": tbl})

    stbl = wandb.Table(columns=["scope", "p", "chars", "text"])
    stbl.add_data("general", None, len(scaffold.get("general_skill", "")),
                  scaffold.get("general_skill", ""))
    for k, v in (scaffold.get("skills") or {}).items():
        stbl.add_data(k, float(scaffold.get("p_task", {}).get(k, 1.0)), len(v), v)
    run.log({"scaffold/current": stbl})

    measured = [c["valid_seen"] for c in cycles if "valid_seen" in c]
    run.summary.update({
        "val/success_rate": measured[-1] if measured else None,
        "val/success_rate_best": max(measured) if measured else None,
        "autoscaffold/cycles": len(cycles),
        "autoscaffold/scaffold_version": scaffold.get("version"),
    })
    url = run.get_url()
    run.finish()
    return {"cycles": len(cycles), "steps": len(train), "name": run_name, "url": url}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm-dir", default="/mnt/data1/zha00175/exp_autoscaffold/alf_scaffold_kl0")
    ap.add_argument("--run-id", default="autoscaffold_kl0_1p5b")
    ap.add_argument("--run-name",
                    default="0725_1545_gigpo_Qwen2.5-1.5B-Instruct_autoscaffold_g8_b32_lr1e-6_kl0")
    ap.add_argument("--seed-run", default=None,
                    help="entity/project/run_id whose steps 1..seed-max are prepended")
    ap.add_argument("--seed-max", type=int, default=150)
    ap.add_argument("--extra-log", action="append", default=[],
                    help="PATH:LO:HI — include steps LO..HI from another train log")
    ap.add_argument("--project", default=PROJECT)
    ap.add_argument("--entity", default=ENTITY)
    ap.add_argument("--log-dir", default=LOG_DIR)
    ap.add_argument("--offline", action="store_true")
    a = ap.parse_args()
    extra = []
    for spec in a.extra_log:
        path, lo, hi = spec.rsplit(":", 2)
        extra.append((path, int(lo), int(hi)))
    print(json.dumps(build(a.arm_dir, a.run_id, a.run_name, a.project, a.entity,
                           a.offline, a.log_dir, seed_run=a.seed_run, seed_max=a.seed_max,
                           extra_logs=extra), indent=2))


if __name__ == "__main__":
    main()
