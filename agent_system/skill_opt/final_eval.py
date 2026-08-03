"""Clean, low-variance final evaluation of none / v0 / best_skill on valid_unseen
(and optionally valid_seen). Multiple independent rollout draws -> mean +- std,
so the headline number isn't a lucky single-draw max from the optimize loop.

Run AFTER the optimize loop finishes (shares the Actor vllm server):
  FE_N=100 FE_DRAWS=3 python -m agent_system.skill_opt.final_eval
"""
import os
import sys
import json
import statistics

sys.path.insert(0, "/mnt/data1/zha00175/verl-agent")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("ALFWORLD_DATA", "/mnt/data1/zha00175/skillzero-env/alfworld_data")

from agent_system.skill.skill_store import SkillStore
from agent_system.skill_opt.actor import ActorClient
from agent_system.skill_opt.envs import build_balanced_managers, attach_skill
from agent_system.skill_opt.rollout import run_balanced_rollout, summarize

OUT = "/mnt/data1/zha00175/gigpo_helper_skillopt"
V0 = "/mnt/data1/zha00175/verl-agent/agent_system/skill/skills_v0.json"
BEST = os.path.join(OUT, "best_skill.json")


def eval_skill(managers, actor, store, n_draws, max_steps, temp):
    for _n, m in managers:
        attach_skill(m, store)
    overalls, per_task = [], {}
    for _ in range(n_draws):
        tr = run_balanced_rollout(managers, actor, max_steps=max_steps, temperature=temp)
        s = summarize(tr)
        overalls.append(s["overall"])
        for k, v in s["per_task"].items():
            per_task.setdefault(k, []).append(v["success"])
    return {
        "overall_mean": round(sum(overalls) / len(overalls), 4),
        "overall_std": round(statistics.pstdev(overalls), 4) if len(overalls) > 1 else 0.0,
        "draws": overalls,
        "per_task_mean": {k: round(sum(v) / len(v), 3) for k, v in sorted(per_task.items())},
    }


def main():
    NPT = int(os.environ.get("FE_NPT", "8"))    # games per task type per draw (6 types -> 6*NPT)
    DRAWS = int(os.environ.get("FE_DRAWS", "3"))
    MAXS = int(os.environ.get("FE_MAXS", "50"))
    TEMP = float(os.environ.get("FE_TEMP", "0.7"))  # Qwen2.5 sampling (top_p/top_k/rep in ActorClient)
    SPLIT = os.environ.get("FE_SPLIT", "eval_out_of_distribution")  # valid_unseen
    actor = ActorClient(temperature=TEMP, max_tokens=512, max_workers=64)
    print(f"[final_eval] per_type={NPT} (total {6*NPT}) draws={DRAWS} max_steps={MAXS} "
          f"temp={TEMP} split={SPLIT} actor_healthy={actor.healthy()}", flush=True)
    managers = build_balanced_managers(NPT, seed=777, is_train=False,
                                       eval_dataset=SPLIT, history_length=2)

    configs = {"none": SkillStore(mode="none"), "v0": SkillStore.from_json(V0, mode="full")}
    if os.path.exists(BEST):
        configs["best"] = SkillStore.from_json(BEST, mode="full")

    res = {}
    for name, store in configs.items():
        r = eval_skill(managers, actor, store, DRAWS, MAXS, TEMP)
        res[name] = r
        print(f"[{name}] overall={r['overall_mean']} +/- {r['overall_std']} "
              f"draws={r['draws']}\n    per_task={r['per_task_mean']}", flush=True)
    out = {"config": {"n_per_type": NPT, "total": 6 * NPT, "draws": DRAWS, "max_steps": MAXS,
                      "temp": TEMP, "split": SPLIT},
           "results": res}
    json.dump(out, open(os.path.join(OUT, "final_eval.json"), "w"), indent=2)
    print("[final_eval] saved final_eval.json", flush=True)


if __name__ == "__main__":
    main()
