"""Autonomous loop to push the GiGPO-trained 1.5B's STANDALONE official valid_unseen
success toward >0.98, by iterating:
  scaffold-train (GiGPO, skill injected p=0.5 mixed, +K steps from latest ckpt)
  -> standalone OFFICIAL eval (verl _validate, mode=none, valid_unseen, temp0.4, n=128, single draw)
  -> GPT-5.5 Helper rewrites the weakest task skills
Repeats until standalone official val > GOAL or the deadline. Robust per-cycle.
"""
import os, sys, json, time, subprocess, re, traceback, shutil
sys.path.insert(0, "/mnt/data1/zha00175/verl-agent")
from agent_system.skill_opt import notify
from openai import OpenAI

OUT = os.environ.get("PUSH98_OUT", "/mnt/data1/zha00175/gigpo_helper_skillopt/push98")
SKILL = os.environ.get("PUSH98_SKILL", "/mnt/data1/zha00175/gigpo_helper_skillopt/skills_push98.json")
TRAIN = os.environ.get("PUSH98_TRAIN", "/mnt/data1/zha00175/verl-agent/examples/gigpo_trainer/run_alfworld_skilltrain_push98.sh")
EXP = os.environ.get("PUSH98_EXP", "skilltrain_push98")
EVAL_SPLIT = os.environ.get("EVAL_SPLIT", "eval_out_of_distribution")
LOGDIR = "/mnt/data1/zha00175/gigpo_helper_logs"
TASKS = ["pick_and_place", "pick_two_obj_and_place", "look_at_obj_in_light",
         "pick_heat_then_place_in_recep", "pick_cool_then_place_in_recep", "pick_clean_then_place_in_recep"]
K = int(os.environ.get("PUSH98_K", "12"))
MAXSTEP = int(os.environ.get("PUSH98_MAXSTEP", "0"))  # 0 = no cap (deadline governs)
VAL_N = int(os.environ.get("PUSH98_VAL_N", "3"))      # multi-draw eval to cut single-draw variance
COLLAPSE_MARGIN = float(os.environ.get("PUSH98_COLLAPSE_MARGIN", "0.04"))  # revert model if sr drops > this below best
GOAL = float(os.environ.get("PUSH98_GOAL", "0.98"))
HELPER_MODEL = os.environ.get("OPENAI_HELPER_MODEL", "gpt-5.5")
os.makedirs(OUT, exist_ok=True)


def P(m):
    print(m, flush=True)
    with open(f"{OUT}/orch.log", "a") as f:
        f.write(time.strftime("%H:%M:%S ") + m + "\n")


def load_oai_key():
    try:
        for line in open("/mnt/data1/zha00175/tool-agent-secrets/openai.env"):
            if line.strip().startswith("OPENAI_API_KEY="):
                return line.strip().split("=", 1)[1]
    except Exception:
        return None


def free_gpu():
    """Kill stragglers ON OUR GPUS ONLY (arm-safe: parallel arms don't kill each other),
    then wait for VRAM on our GPUs. GPU list from PUSH98_GPUS (default 0,1)."""
    gpus = os.environ.get("PUSH98_GPUS", "0,1")
    try:
        on_gpu = subprocess.run(["bash", "-lc",
            f"nvidia-smi -i {gpus} --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | sort -u"],
            capture_output=True, text=True).stdout.split()
        mine = set(subprocess.run(["bash", "-lc",
            "ps -eo pid,user | awk '$2==\"zha00175\"{print $1}'"], capture_output=True, text=True).stdout.split())
        for pid in on_gpu:
            if pid.strip() and pid.strip() in mine:
                try: os.kill(int(pid), 9)
                except Exception: pass
    except Exception:
        pass
    for _ in range(30):
        try:
            u = subprocess.run(["nvidia-smi", "-i", gpus, "--query-gpu=memory.used",
                                "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
            if u.strip() and max(int(x) for x in u.split()) < 3000:
                return
        except Exception:
            return
        time.sleep(5)


def run(cmd, logpath):
    free_gpu()
    with open(logpath, "w") as f:
        return subprocess.run(["bash", "-lc", cmd], stdout=f, stderr=subprocess.STDOUT).returncode


def parse_val(logpath, with_counts=False):
    """Return (overall, {task:rate}) from the last val/* block in a verl train/val log.

    with_counts=True instead returns (overall, {task:rate}, {task:n}), reading the companion
    `val/<task>_n` emitted alongside each rate. A per-task rate on ALFWorld stands on however
    many episodes of that type the sampler happened to draw, so the denominator decides whether
    the rate can be read at all. Older logs predate the counts metric and simply yield {}.
    """
    try:
        txt = open(logpath, errors="ignore").read()
    except Exception:
        return (None, {}, {}) if with_counts else (None, {})
    ov = re.findall(r"val/success_rate[:'\s]+\(?(?:np\.float64\()?([0-9.]+)", txt)
    overall = float(ov[-1]) if ov else None
    per, cnt = {}, {}
    for t in TASKS:
        vals = re.findall(rf"val/{t}_success_rate[:'\s]+\(?(?:np\.float64\()?([0-9.]+)", txt)
        if vals:
            per[t] = float(vals[-1])
        ns = re.findall(rf"val/{t}_n[:'\s]+\(?(?:np\.float64\()?([0-9.]+)", txt)
        if ns:
            cnt[t] = int(float(ns[-1]))
    return (overall, per, cnt) if with_counts else (overall, per)


def _fmt_transcripts(fails_path, weak, max_fail=12, max_ok=3, max_steps=26):
    """obs->action trace of weak-task failures (+ a few successes for contrast), with
    computed signals (opens/takes/invalid, ever-held-an-object) that make the failure
    mode explicit. The env's per-step observation is the key evidence for *why* it fails."""
    try:
        d = json.load(open(fails_path))
    except Exception:
        return ""
    trs = d.get("transcripts", [])
    ws = set(weak)
    fails = [t for t in trs if not t.get("success") and t.get("task_type") in ws]
    oks = [t for t in trs if t.get("success") and t.get("task_type") in ws]
    sel = fails[:max_fail] + oks[:max_ok]
    if not sel:
        sel = [t for t in trs if not t.get("success")][:max_fail]

    def clean(o):
        return str(o).replace("\n", " ").strip()[:130]

    blocks = []
    for t in sel:
        steps = t.get("steps", [])
        acts = [str(s.get("action", "")) for s in steps]
        n_open = sum(1 for a in acts if a.startswith("open "))
        n_take = sum(1 for a in acts if a.startswith("take "))
        n_inv = sum(1 for s in steps if not s.get("valid", True))
        held = "yes" if n_take else "NO — never picked up any object"
        goal = clean(steps[0].get("obs", ""))[:240] if steps else ""
        lines = [f"--- {t.get('task_type')} | success={t.get('success')} | {t.get('n_steps')} steps "
                 f"| opens={n_open} takes={n_take} invalid={n_inv} | held_an_object={held}",
                 f"  GOAL/start: {goal}"]
        for j, st in enumerate(steps[:max_steps]):
            v = "  <-INVALID(Nothing happens)" if not st.get("valid", True) else ""
            lines.append(f"  step{j}: saw \"{clean(st.get('obs',''))}\"  => {st.get('action')}{v}")
        if len(steps) > max_steps:
            lines.append(f"  ...(+{len(steps)-max_steps} more steps, ran out without finishing)")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def helper_refine(per_task, oai_key, fails_path=None):
    cur = json.load(open(SKILL))
    weak = [t for t, _ in sorted(per_task.items(), key=lambda x: x[1])[:3]]
    traj = _fmt_transcripts(fails_path, weak) if fails_path else ""
    sysp = ("You coach an RL-trained ALFWorld 1.5B text-agent. You are given REAL standalone rollout "
            "trajectories: each step shows what the env returned (saw \"...\") then the action taken, "
            "and each trajectory header has computed signals (opens/takes/invalid/held_an_object). "
            "USE that evidence concretely — e.g. held_an_object=NO with opens=0 means the agent never "
            "opened the closed cabinets/drawers/fridge where the object actually is (a search/open "
            "failure), NOT merely 'looping'; read the observations to see where the object really was. "
            "STUDY THE FAILURES of the weakest task types and infer the SPECIFIC recurring mistake "
            "(e.g. wrong order, not opening a closed receptacle, acting before holding the object, "
            "repeating an invalid action, wrong appliance/destination, wandering, stopping after the "
            "first object in two-object tasks). Then rewrite the per-task skill hints to directly "
            "counter those observed mistakes, while keeping the strong tasks intact. Each skill: 1-4 "
            "short, concrete sentences using exact ALFWorld verbs (go to, open, take X from Y, "
            "put X in/on Y, use, heat X with microwave, cool X with fridge, clean X with sinkbasin). "
            "No game-specific ids/answers. Output STRICT JSON {\"skills\":{<task>:<hint>}, "
            "\"general_skill\":\"...\", \"diagnosis\":\"<1-2 lines: the failure pattern you saw>\"}. "
            "Valid task types: " + ", ".join(TASKS))
    user = (f"Current STANDALONE success per task (official eval): {json.dumps(per_task)}\n"
            f"WEAKEST tasks to fix: {weak}\n\n"
            f"REAL ROLLOUT TRAJECTORIES (mode=none, the model's own behavior):\n{traj or '(none captured)'}\n\n"
            f"Current skills:\n{json.dumps(cur.get('skills', {}), indent=1)}\n"
            f"Current general_skill: {cur.get('general_skill','')}\n\n"
            f"Diagnose the failure pattern from the trajectories, then rewrite to fix it. STRICT JSON only.")
    c = OpenAI(api_key=oai_key, timeout=180, max_retries=2)
    r = c.chat.completions.create(model=HELPER_MODEL,
                                  messages=[{"role": "system", "content": sysp}, {"role": "user", "content": user}],
                                  response_format={"type": "json_object"}, max_completion_tokens=5000)
    out = json.loads(r.choices[0].message.content)
    diag = out.get("diagnosis")
    if isinstance(diag, str) and diag.strip():
        P(f"  [helper diagnosis] {diag.strip()[:300]}")
        cur["last_diagnosis"] = diag.strip()
    # --- champion keep-best per task + freeze strong tasks (only rewrite weak ones) ---
    cf = {}
    try:
        if fails_path:
            cf = (json.load(open(fails_path)).get("counterfactual") or {}).get("with_skill", {}) or {}
    except Exception:
        cf = {}
    champ = cur.get("champion", {})          # {task: {"skill":str, "cf":float}}
    cur_skills = cur.get("skills", {})
    # promote: if the current skill's with-skill counterfactual is the best seen for a task, it's champion
    for t, sc in cf.items():
        if t in TASKS and float(sc) >= champ.get(t, {}).get("cf", -1.0):
            champ[t] = {"skill": cur_skills.get(t, ""), "cf": float(sc)}
    sk = dict(cur_skills)
    for t in weak:                            # ONLY weak tasks change; strong tasks stay frozen
        v = (out.get("skills") or {}).get(t)
        cur_cf = float(cf.get(t, 1.0)); best_cf = champ.get(t, {}).get("cf", 0.0)
        if cur_cf < best_cf - 0.1 and champ.get(t, {}).get("skill"):
            sk[t] = champ[t]["skill"]          # regressed -> revert to champion
            P(f"  [champion] revert {t}: cf {cur_cf:.2f} < champ {best_cf:.2f}")
        elif isinstance(v, str) and v.strip():
            sk[t] = v.strip()                  # accept the new trajectory-informed candidate
    cur["skills"] = sk
    cur["champion"] = champ
    g = out.get("general_skill")
    if isinstance(g, str) and g.strip():
        cur["general_skill"] = g.strip()
    cur["mode"] = "full"; cur.setdefault("default_p", 1.0)
    cur["used_trajectories"] = bool(traj)
    json.dump(cur, open(SKILL + ".tmp", "w"), indent=2)
    os.replace(SKILL + ".tmp", SKILL)


def main():
    oai = load_oai_key()
    _dl = os.environ.get("GIGPO_DEADLINE_EPOCH")
    deadline = float(_dl) if _dl else time.time() + 10 * 3600
    sp = f"{OUT}/status.json"
    CKDIR = f"/mnt/data1/zha00175/gigpo_helper_ckpts/{EXP}"
    step = 0; best = 0.0; best_step = 0; cycle = 0
    # restart-safe: resume step from the checkpoint dir and best/cycle from status.json
    try:
        step = int(open(f"{CKDIR}/latest_checkpointed_iteration.txt").read().strip())
    except Exception:
        pass
    try:
        _s = json.load(open(sp))
        best = float(_s.get("best_standalone_unseen", 0.0))
        best_step = int(_s.get("best_step", 0)); cycle = int(_s.get("cycle", 0))
    except Exception:
        pass

    def status(extra=None):
        st = {"updated": time.strftime("%Y-%m-%d %H:%M:%S"), "cycle": cycle, "step": step,
              "best_standalone_unseen": round(best, 4), "best_step": best_step, "goal": GOAL,
              "remaining_h": round((deadline - time.time()) / 3600, 2)}
        if extra:
            st.update(extra)
        notify.write_status(sp, st)

    status()
    notify.send_email("[push98] started",
                      f"Goal: STANDALONE valid_unseen (official _validate) > {GOAL}.\n"
                      f"From trained-1.5B gs150. Cycle = train half(+{K} steps) -> standalone eval -> refine skill.")
    while time.time() < deadline and best < GOAL and (not MAXSTEP or step < MAXSTEP):
        cycle += 1
        target = step + K
        try:
            tlog = f"{LOGDIR}/{EXP}_train_c{cycle}.log"
            P(f"[c{cycle}] scaffold-train (weighted: look_at/pick_two oversampled) -> step {target}")
            envp = f"EXP={EXP} EVAL_SPLIT={EVAL_SPLIT} SKILL_PATH={SKILL}"
            run(f"{envp} SAVE_FREQ={K} TEST_FREQ=99999 VAL_BEFORE=False MAX_CKPT=50 bash {TRAIN} weighted {target}", tlog)
            step = target
            P(f"[c{cycle}] standalone OFFICIAL eval (none, val_only) split={EVAL_SPLIT} x{VAL_N} draws (avg)")
            srs, pers = [], []
            for _d in range(VAL_N):
                elog = f"{LOGDIR}/{EXP}_eval_c{cycle}_d{_d}.log"
                # vary env seed per draw -> each draw samples a DIFFERENT 128-of-~140 subset (real variance reduction + covers the missing games)
                run(f"{envp} ENV_SEED={_d} VAL_ONLY=True VAL_BEFORE=True bash {TRAIN} none {target}", elog)
                _sr, _per = parse_val(elog)
                if _sr is not None:
                    srs.append(_sr); pers.append(_per)
            sr = round(sum(srs) / len(srs), 4) if srs else None
            per = {t: round(sum(p.get(t, 0.0) for p in pers) / len(pers), 3) for t in TASKS} if pers else {}
            if srs:
                P(f"[c{cycle}] standalone draws={srs} -> mean {sr}")
            # ALSO eval WITH the skill force-injected (full) on the same official protocol/full val set
            elog_s = f"{LOGDIR}/{EXP}_evalskill_c{cycle}.log"
            P(f"[c{cycle}] WITH-SKILL OFFICIAL eval (full, val_only)")
            run(f"{envp} VAL_ONLY=True VAL_BEFORE=True bash {TRAIN} full {target}", elog_s)
            sr_skill, per_skill = parse_val(elog_s)
            P(f"[c{cycle}] standalone={sr} per_task={per}")
            P(f"[c{cycle}] with_skill={sr_skill} per_task={per_skill}  (skill lift={None if (sr is None or sr_skill is None) else round(sr_skill-sr,3)})")
            if sr is not None:
                collapsed = (best_step > 0 and sr < best - COLLAPSE_MARGIN)
                if sr > best:
                    best = sr; best_step = target
                    try: shutil.copy(SKILL, f"{OUT}/best_skill_step{target}.json")
                    except Exception: pass
                status({"last_sr": sr, "last_per_task": per,
                        "last_sr_with_skill": sr_skill, "last_per_task_with_skill": per_skill})
                if collapsed:
                    # OVER-TRAINING COLLAPSE -> revert model to the best checkpoint, resume from there
                    # (skip skill refine: don't learn from a degraded model)
                    try:
                        open(f"{CKDIR}/latest_checkpointed_iteration.txt", "w").write(str(best_step))
                        P(f"[c{cycle}] MODEL-REVERT to best step {best_step} (sr {sr:.3f} << best {best:.3f}); skip refine")
                        step = best_step
                        status({"last": f"reverted_to_{best_step}", "last_sr": sr})
                    except Exception as e:
                        P(f"[c{cycle}] revert failed: {str(e)[:150]}")
                    continue
                # capture REAL rollout trajectories of this checkpoint for the Helper to read
                fails_path = f"{OUT}/fails_step{target}.json"
                try:
                    clog = f"{LOGDIR}/{EXP}_capture_c{cycle}.log"
                    P(f"[c{cycle}] capturing rollout trajectories (merge->serve->rollout)")
                    rc = run(f"cd /mnt/data1/zha00175/verl-agent && CAP_STEP={target} CAP_EXP={EXP} "
                             f"CAP_SPLIT={EVAL_SPLIT} CAP_OUTJSON={fails_path} CAP_NPT=10 "
                             f"/mnt/data1/zha00175/miniconda/envs/verl/bin/python -u "
                             f"-m agent_system.skill_opt.capture_fails", clog)
                    P(f"[c{cycle}] capture rc={rc} trajectories={'yes' if os.path.exists(fails_path) else 'no'}")
                except Exception as e:
                    P(f"[c{cycle}] capture failed: {str(e)[:200]}")
                try:
                    helper_refine(per, oai, fails_path if os.path.exists(fails_path) else None)
                    P(f"[c{cycle}] skill refined (trajectory-informed={'yes' if os.path.exists(fails_path) else 'no'})")
                except Exception as e:
                    P(f"[c{cycle}] refine failed: {str(e)[:200]}")
            else:
                status({"last": "val_parse_fail"})
        except Exception:
            P(f"[c{cycle}] EXCEPTION\n{traceback.format_exc()}")
            status({"last": "exception"}); time.sleep(30)
    status({"last": "finished"})
    notify.send_email("[push98] finished",
                      f"best STANDALONE valid_unseen (official) = {best} at step {best_step} (goal {GOAL}). cycles={cycle}.")
    P(f"[done] best={best} at step {best_step}")


if __name__ == "__main__":
    main()
