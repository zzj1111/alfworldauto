"""Phase 1: diagnose where the (trained) Actor still fails. Many no-skill rollouts,
aggregate FAILURE PATTERNS by task type (not per-game), + save sample failed
trajectories for the Helper to turn into GENERAL skills."""
import os
import sys
import json
from collections import defaultdict

sys.path.insert(0, "/mnt/data1/zha00175/verl-agent")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("ALFWORLD_DATA", "/mnt/data1/zha00175/skillzero-env/alfworld_data")

from agent_system.skill.skill_store import SkillStore
from agent_system.skill_opt.actor import ActorClient
from agent_system.skill_opt.envs import build_balanced_managers, attach_skill
from agent_system.skill_opt.rollout import run_balanced_rollout, summarize

OUT = "/mnt/data1/zha00175/gigpo_helper_skillopt"
NPT = int(os.environ.get("DG_NPT", "40"))
DRAWS = int(os.environ.get("DG_DRAWS", "3"))
MAXS = int(os.environ.get("DG_MAXS", "50"))
TEMP = float(os.environ.get("DG_TEMP", "0.7"))
SPLIT = os.environ.get("DG_SPLIT", "train")  # diagnose on TRAIN (where we'll also train)


def fail_stats(t):
    steps = t["steps"]
    acts = [s["action"] for s in steps]
    n = len(steps)
    invalid = sum(1 for s in steps if not s.get("valid", True))
    repeats = sum(1 for i in range(1, n) if acts[i] == acts[i - 1])
    return {"n_steps": n, "invalid_frac": round(invalid / max(1, n), 2),
            "repeat_frac": round(repeats / max(1, n), 2), "timed_out": n >= MAXS}


def compact(t, max_steps=14):
    lines = []
    for s in t["steps"][:max_steps]:
        obs = " ".join(str(s["obs"]).split())[:130]
        flag = "" if s.get("valid", True) else "[INVALID]"
        lines.append(f"  {obs}\n   -> {s['action']} {flag}(r={s['reward']})")
    return "\n".join(lines)


def main():
    actor = ActorClient(temperature=TEMP, max_tokens=512, max_workers=64)
    print(f"[diagnose] per_type={NPT} draws={DRAWS} split={SPLIT} actor={actor.healthy()}", flush=True)
    is_train = (SPLIT == "train")
    eval_dataset = "eval_out_of_distribution" if SPLIT == "valid_unseen" else "eval_in_distribution"
    mgrs = build_balanced_managers(NPT, seed=0, is_train=is_train, eval_dataset=eval_dataset, history_length=2)

    all_tr = []
    for d in range(DRAWS):
        for _n, m in mgrs:
            attach_skill(m, SkillStore(mode="none"))
        tr = run_balanced_rollout(mgrs, actor, max_steps=MAXS, temperature=TEMP)
        all_tr.extend(tr)
        print(f"[draw {d+1}/{DRAWS}] {summarize(tr)['overall']:.3f} overall ({len(tr)} games)", flush=True)

    by_task = defaultdict(lambda: {"n": 0, "success": 0, "fails": []})
    for t in all_tr:
        bt = by_task[t["task_type"]]
        bt["n"] += 1
        if t["success"]:
            bt["success"] += 1
        else:
            bt["fails"].append(t)

    report = {}
    for tt, d in sorted(by_task.items()):
        fails = d["fails"]
        agg = {"n": d["n"], "success_rate": round(d["success"] / max(1, d["n"]), 3),
               "n_failed": len(fails)}
        if fails:
            import statistics
            st = [fail_stats(f) for f in fails]
            agg["avg_fail_steps"] = round(statistics.mean(s["n_steps"] for s in st), 1)
            agg["avg_invalid_frac"] = round(statistics.mean(s["invalid_frac"] for s in st), 2)
            agg["avg_repeat_frac"] = round(statistics.mean(s["repeat_frac"] for s in st), 2)
            agg["timed_out_frac"] = round(sum(s["timed_out"] for s in st) / len(st), 2)
            agg["sample_failures"] = [compact(f) for f in fails[:4]]
        report[tt] = agg
        print(f"\n=== {tt}: succ {agg['success_rate']} (n={d['n']}, failed={len(fails)}) "
              f"failstats invalid={agg.get('avg_invalid_frac')} repeat={agg.get('avg_repeat_frac')} "
              f"timeout={agg.get('timed_out_frac')} avgsteps={agg.get('avg_fail_steps')} ===", flush=True)

    overall = sum(b["success"] for b in by_task.values()) / max(1, sum(b["n"] for b in by_task.values()))
    out = {"split": SPLIT, "n_per_type": NPT, "draws": DRAWS, "overall_success": round(overall, 3),
           "per_task": report}
    json.dump(out, open(os.path.join(OUT, "diagnostic.json"), "w"), indent=2)
    print(f"\n[diagnose] overall_success={overall:.3f} -> saved diagnostic.json", flush=True)


if __name__ == "__main__":
    main()
