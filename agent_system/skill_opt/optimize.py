"""Inference-time scaffold optimization (autonomous) — v3: PER-TASK MERGE.

v2 stalled: monolithic full-skill rewrites can't beat an already-decent champion
(a single skill set can't be good at all 6 task types at once). v3 keeps the
better skill PER TASK TYPE and combines them, with a confirmation eval to reject
noise-driven merges. Champion still only ever improves.

  each round:
    1. eval champion on TRAIN (avg DEV_DRAWS) -> overall + per-task success + failures
    2. Helper rewrites skills from champion's failures -> candidate
    3. eval candidate on TRAIN (avg DEV_DRAWS) -> per-task success
    4. merged = champion, but for each task where candidate beats champion by
       PT_MARGIN (and enough samples) take candidate's skill for that task
    5. if merged changed: confirm with another eval; accept iff overall improves
    6. on accept / periodically: averaged valid_unseen eval -> honest best tracking
"""
import os
import sys
import json
import time
import traceback
from collections import defaultdict

sys.path.insert(0, "/mnt/data1/zha00175/verl-agent")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("ALFWORLD_DATA", "/mnt/data1/zha00175/skillzero-env/alfworld_data")

from agent_system.skill.skill_store import SkillStore
from agent_system.skill_opt.actor import ActorClient
from agent_system.skill_opt.envs import build_balanced_managers, attach_skill
from agent_system.skill_opt.rollout import run_balanced_rollout, summarize
from agent_system.skill_opt.helper import HelperClient
from agent_system.skill_opt import notify

OUT = "/mnt/data1/zha00175/gigpo_helper_skillopt"
V0 = "/mnt/data1/zha00175/verl-agent/agent_system/skill/skills_v0.json"


def _i(k, d): return int(os.environ.get(k, d))
def _f(k, d): return float(os.environ.get(k, d))
N_PER_TYPE  = _i("GIGPO_N_PER_TYPE", 8)   # games per task type per draw (6 types -> 48 balanced)
MAX_STEPS   = _i("GIGPO_MAX_STEPS", 50)
ROLL_TEMP   = _f("GIGPO_ROLL_TEMP", 0.7)  # Qwen2.5 recommended sampling (with top_p/top_k/rep in ActorClient)
EVAL_TEMP   = _f("GIGPO_EVAL_TEMP", 0.7)
DEV_DRAWS   = _i("GIGPO_DEV_DRAWS", 2)
EVAL_DRAWS  = _i("GIGPO_EVAL_DRAWS", 2)
MARGIN      = _f("GIGPO_ACCEPT_MARGIN", 0.02)
PT_MARGIN   = _f("GIGPO_PT_MARGIN", 0.1)
PT_MIN_N    = _i("GIGPO_PT_MIN_N", 4)
EVAL_EVERY  = _i("GIGPO_EVAL_EVERY", 3)
DEADLINE_H  = _f("GIGPO_DEADLINE_HOURS", 10.0)
HELPER_MODEL = os.environ.get("OPENAI_HELPER_MODEL", "gpt-5.5")
HELPER_BASE_URL = os.environ.get("GIGPO_HELPER_BASE_URL")


def _load_openai():
    env = {}
    try:
        for line in open("/mnt/data1/zha00175/tool-agent-secrets/openai.env"):
            line = line.strip()
            if "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1); env[k] = v
    except Exception:
        pass
    return env


def _clone(store):
    s = store.snapshot()
    return SkillStore(skills=s.get("skills"), general_skill=s.get("general_skill", ""),
                      p_task=s.get("p_task"), mode="full", default_p=s.get("default_p", 1.0))


def eval_pt(managers, actor, store, draws, temp):
    """managers: list of (name, manager), one per task type (balanced).
    Return (overall_mean, per_task{task:{success,n}}, first_draw_transcripts)."""
    for _n, m in managers:
        attach_skill(m, store)
    overalls, acc, first = [], defaultdict(lambda: {"s": [], "n": 0}), None
    for d in range(draws):
        tr = run_balanced_rollout(managers, actor, max_steps=MAX_STEPS, temperature=temp)
        s = summarize(tr)
        overalls.append(s["overall"])
        for k, v in s["per_task"].items():
            acc[k]["s"].append(v["success"]); acc[k]["n"] += v["n"]
        if d == 0:
            first = tr
    pt = {k: {"success": round(sum(v["s"]) / len(v["s"]), 3), "n": v["n"]} for k, v in acc.items()}
    return sum(overalls) / len(overalls), pt, first


def main():
    os.makedirs(OUT, exist_ok=True)
    log = open(os.path.join(OUT, "run.log"), "a", buffering=1)
    def P(*a):
        msg = " ".join(str(x) for x in a)
        print(msg, flush=True); log.write(msg + "\n")

    t_start = time.time()
    _dl = os.environ.get("GIGPO_DEADLINE_EPOCH")
    deadline = float(_dl) if _dl else (t_start + DEADLINE_H * 3600)
    oai = _load_openai()
    actor = ActorClient(temperature=ROLL_TEMP, max_tokens=512, max_workers=48)
    helper = HelperClient(model=HELPER_MODEL, api_key=oai.get("OPENAI_API_KEY"),
                          base_url=HELPER_BASE_URL, use_temperature=bool(HELPER_BASE_URL))
    P(f"[init v3] actor={actor.healthy()} helper={HELPER_MODEL} N_PER_TYPE={N_PER_TYPE} "
      f"(balanced 6x) roll_t={ROLL_TEMP} draws={DEV_DRAWS} PT_MARGIN={PT_MARGIN}")

    P(f"[init] building balanced env managers (6 task types x {N_PER_TYPE} games) ...")
    train_managers = build_balanced_managers(N_PER_TYPE, seed=0, is_train=True, history_length=2)
    eval_managers = build_balanced_managers(N_PER_TYPE, seed=1234, is_train=False,
                                            eval_dataset="eval_out_of_distribution", history_length=2)
    P("[init] managers built (12 total: 6 train + 6 eval)")

    none_eval, _, _ = eval_pt(eval_managers, actor, SkillStore(mode="none"), EVAL_DRAWS, EVAL_TEMP)
    v0_eval, _, _ = eval_pt(eval_managers, actor, SkillStore.from_json(V0), EVAL_DRAWS, EVAL_TEMP)
    none_eval, v0_eval = round(none_eval, 4), round(v0_eval, 4)
    P(f"[baseline] none={none_eval}  v0={v0_eval}")

    champion = SkillStore(mode="full")
    best_path = os.path.join(OUT, "best_skill.json")
    if os.path.exists(best_path):
        try:
            champion = SkillStore.from_json(best_path, mode="full")
            P(f"[resume] champion ({len(champion.skills)} skills)")
        except Exception:
            pass

    best_eval = none_eval
    best_snap = champion.snapshot()
    helper_fails = accepts = rounds = 0
    status_path = os.path.join(OUT, "status.json")
    rounds_path = os.path.join(OUT, "rounds.jsonl")

    def save_status(extra=None):
        st = {"updated": time.strftime("%Y-%m-%d %H:%M:%S"), "version": "v3",
              "round": rounds, "elapsed_h": round((time.time() - t_start) / 3600, 2),
              "baseline_none": none_eval, "baseline_v0": v0_eval,
              "best_eval_unseen": round(best_eval, 4), "accepts": accepts,
              "helper_fails": helper_fails, "helper_model": HELPER_MODEL}
        if extra:
            st.update(extra)
        notify.write_status(status_path, st)

    save_status()
    notify.send_email("[GiGPO-Helper] optimizer v3 (per-task merge) started",
                      f"none={none_eval} v0={v0_eval}; resuming champion. PT_MARGIN={PT_MARGIN}.")

    while time.time() < deadline:
        rounds += 1
        r0 = time.time()
        try:
            oc, pt_c, champ_tr = eval_pt(train_managers, actor, champion, DEV_DRAWS, ROLL_TEMP)
            s_help = summarize(champ_tr)
            proposal, hstat = helper.propose(s_help, champ_tr, champion.snapshot())
            if proposal is None:
                helper_fails += 1
                P(f"[round {rounds}] helper FAIL ({hstat}) champ={oc:.3f}")
                save_status({"last": "helper_fail"}); time.sleep(5); continue
            candidate = _clone(champion)
            candidate.update(skills=proposal["skills"], general_skill=proposal["general_skill"])
            ocand, pt_cand, _ = eval_pt(train_managers, actor, candidate, DEV_DRAWS, ROLL_TEMP)

            # per-task merge: take candidate's skill for tasks it clearly improved
            merged = _clone(champion)
            changed = []
            for t, sk in (proposal.get("skills") or {}).items():
                cs = pt_cand.get(t, {}).get("success", 0.0)
                hs = pt_c.get(t, {}).get("success", 0.0)
                cn = pt_cand.get(t, {}).get("n", 0)
                if cs > hs + PT_MARGIN and cn >= PT_MIN_N:
                    merged.skills[t] = sk
                    changed.append(t)
            gen_changed = False
            if proposal.get("general_skill") and ocand > oc + MARGIN:
                merged.general_skill = proposal["general_skill"]; gen_changed = True

            accept = False
            om = None
            if changed or gen_changed:
                om, _, _ = eval_pt(train_managers, actor, merged, DEV_DRAWS, ROLL_TEMP)  # confirm
                if om > oc + MARGIN:
                    champion = merged
                    accepts += 1
                    accept = True

            unseen = None
            if accept or rounds % EVAL_EVERY == 0:
                unseen, _, _ = eval_pt(eval_managers, actor, champion, EVAL_DRAWS, EVAL_TEMP)
                if unseen > best_eval:
                    best_eval = unseen
                    best_snap = champion.snapshot()
                    SkillStore(**{k: best_snap[k] for k in ("skills", "general_skill", "p_task")},
                               mode="full").save_json(best_path)
            rec = {"round": rounds, "t": round(time.time() - r0), "champ": round(oc, 3),
                   "cand": round(ocand, 3), "merged": (round(om, 3) if om is not None else None),
                   "changed": changed, "accept": accept,
                   "unseen": (round(unseen, 3) if unseen is not None else None),
                   "best": round(best_eval, 3), "per_task_champ": pt_c}
            with open(rounds_path, "a") as f:
                f.write(json.dumps(rec) + "\n")
            P(f"[round {rounds}] champ={oc:.3f} cand={ocand:.3f} merged={rec['merged']} "
              f"changed={changed} accept={accept} unseen={rec['unseen']} best={best_eval:.3f} ({rec['t']}s)")
            save_status({"last_round": rec})
        except Exception:
            P(f"[round {rounds}] EXCEPTION:\n{traceback.format_exc()}")
            save_status({"last": "exception"}); time.sleep(10)

    SkillStore(**{k: best_snap[k] for k in ("skills", "general_skill", "p_task")},
               mode="full").save_json(best_path)
    save_status({"last": "finished"})
    notify.send_email("[GiGPO-Helper] optimizer finished",
                      f"rounds={rounds} best valid_unseen={round(best_eval,4)} "
                      f"(none={none_eval}, v0={v0_eval}) accepts={accepts}.")
    P(f"[done] rounds={rounds} best={best_eval}")


if __name__ == "__main__":
    main()
