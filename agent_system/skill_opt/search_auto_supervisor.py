"""Supervisor for the AUTO-SCAFFOLD Search-R1 experiment.

Difference from the hand-tuned iter supervisor: there is NO hard-coded withdrawal
`scaffold_p()` and NO fix-list prompt. At a FIXED cadence (every new checkpoint) the
GPT-5.5 controller is handed the full measured state and decides all four scaffold
actions (content / injection p / bucket partition / revert). The controller gets ZERO
outcome priors; it learns any policy online from its own decision->outcome history.

Cold start is BLIND: before training, the controller produces an initial scaffold with
no measurements at all.

Model revert is auto-executed (AUTO_ALLOW_REVERT=1) but made SAFE via generations: each
revert restarts training into a fresh `gen{k}` checkpoint dir, so it never overwrites the
pre-revert checkpoints and the (gen, step) key keeps the refine trigger correct after the
step counter resets. A registry maps checkpoint ids -> hf paths across generations; the
controller may only revert to an id we showed it (state.available_checkpoints). Guarded by
MAX_REVERTS so a revert loop can't thrash a multi-day run.
"""
import os, sys, time, json, subprocess, socket, traceback, glob
sys.path.insert(0, "/mnt/data1/zha00175/verl-agent")
from agent_system.skill_opt import notify, controller

ROOT = "/mnt/data1/zha00175/verl-agent"
VERL = "/mnt/data1/zha00175/miniconda/envs/verl/bin/python3"
RETR_SH = f"{ROOT}/examples/search/retriever/run_retriever_gpu.sh"
TRAIN_SH = f"{ROOT}/examples/gigpo_trainer/run_search_3b_auto.sh"
CAPTURE = f"{ROOT}/agent_system/skill_opt/capture_search.py"
OUT = "/mnt/data1/zha00175/searchR1_data/orch_auto"
TRAIN_LOG = f"{OUT}/train.log"
ORCH_LOG = f"{OUT}/orch.log"
RUN_DIR = "/mnt/data1/zha00175/ckpts_search_3b_auto"   # parent; generations live in gen{k}/
SCAFFOLD = "/mnt/data1/zha00175/searchR1_data/search_skills_auto.json"
CAP_DIR = f"{OUT}/captures"
HIST = f"{OUT}/history.json"
GEN_STATE = f"{OUT}/gen_state.json"
REGISTRY = f"{OUT}/ckpt_registry.json"
DEADLINE_FILE = f"{OUT}/deadline.txt"
KEY_FILE = "/mnt/data1/zha00175/.openai_key"
PORT = 8010
TRAIN_TAG = "gigpo_3b_auto"
WANDB_RUN = "gigpo_3b_auto"
PROJECT = "mhong-university-of-minnesota/verl_agent_search"
HOURS = float(os.environ.get("AUTO_HOURS", "72"))
N_GPUS = 2
TOTAL_STEPS = int(os.environ.get("AUTO_TOTAL_STEPS", "400"))   # 400 = de-risk pilot; 662 = full/comparable
FIRST_REFINE = 50               # fixed cadence: every new ckpt (save_freq=50)
CAPTURE_GPU = os.environ.get("AUTO_CAPTURE_GPU", "3")
ALLOW_REVERT = os.environ.get("AUTO_ALLOW_REVERT", "0") == "1"
MAX_REVERTS = int(os.environ.get("AUTO_MAX_REVERTS", "5"))
DSETS = ["nq", "triviaqa", "popqa", "hotpotqa", "2wikimultihopqa", "musique", "bamboogle"]
os.makedirs(OUT, exist_ok=True); os.makedirs(CAP_DIR, exist_ok=True)


def P(m):
    print(m, flush=True)
    try:
        open(ORCH_LOG, "a").write(time.strftime("%m-%d %H:%M:%S ") + m + "\n")
    except Exception:
        pass


def email(subj, body):
    try:
        notify.send_email(subj, body)
    except Exception:
        P("[email-fail] " + subj)


def sh(c):
    return subprocess.run(["bash", "-lc", c], capture_output=True, text=True).stdout


def pids(pat):
    return [p for p in sh(f"ps -eo pid,cmd | grep -E '{pat}' | grep zha00175 | grep -v grep | grep -v 'bash -lc' | awk '{{print $1}}'").split() if p]


def _read_json(path, default):
    try:
        return json.load(open(path))
    except Exception:
        return default


def _write_json(path, obj):
    tmp = path + ".tmp"
    json.dump(obj, open(tmp, "w"), indent=2, ensure_ascii=False)
    os.replace(tmp, path)


# ---------- generations + checkpoint registry ----------
def gen_state():
    s = _read_json(GEN_STATE, None)
    if s is None:
        s = {"gen": 0, "active_dir": f"{RUN_DIR}/gen0", "reverts": 0}
        _write_json(GEN_STATE, s)
    return s


def active_dir():
    return gen_state()["active_dir"]


def register_ckpt(ckpt_id, hf):
    r = _read_json(REGISTRY, {})
    r[ckpt_id] = hf
    _write_json(REGISTRY, r)


def ckpt_id_of(gen, step):
    return f"gen{gen}_step{step}"


# ---------- retriever / training process ----------
def port_up():
    try:
        s = socket.create_connection(("127.0.0.1", PORT), timeout=3); s.close(); return True
    except Exception:
        return False


def retriever_ok():
    r = sh("curl -s -m 25 -X POST http://127.0.0.1:%d/retrieve -H 'Content-Type: application/json' "
           "-d '{\"query\":\"who wrote hamlet\",\"topk\":3,\"return_scores\":false}'" % PORT)
    return ('"result"' in r) and ("contents" in r or "document" in r)


def launch_retriever():
    for p in pids("retrieval_server"):
        try: os.kill(int(p), 9)
        except Exception: pass
    for _ in range(20):
        if not port_up(): break
        time.sleep(1)
    subprocess.Popen(["bash", "-lc", f"bash {RETR_SH}"], stdout=open(f"{OUT}/retriever.log", "a"),
                     stderr=subprocess.STDOUT, start_new_session=True)
    P("[retriever] launched")


def training_alive():
    return len(pids(f"main_ppo.*({TRAIN_TAG}|ckpts_search_3b_auto)")) > 0


def ckpt_world_size_ok(ckpt_dir):
    shards = glob.glob(f"{ckpt_dir}/global_step_*/actor/model_world_size_*_rank_0.pt")
    bad = [x for x in shards if f"world_size_{N_GPUS}_" not in x]
    if bad:
        P(f"[guard] ckpt world_size MISMATCH (need {N_GPUS}): {bad[:2]}")
        email("[auto] ckpt world_size 不匹配 - 需人工", f"{ckpt_dir} 有非 world_size={N_GPUS} 分片:\n{bad[:5]}")
        return False
    return True


def launch_training(revert_model=None, revert_mode="auto", ckpt_dir=None):
    ckpt_dir = ckpt_dir or active_dir()
    if not ckpt_world_size_ok(ckpt_dir):
        P("[guard] refusing to launch until ckpt/world_size resolved"); return
    env = f"AUTO_CKPT_DIR={ckpt_dir} AUTO_ALLOW_REVERT={'1' if ALLOW_REVERT else '0'} AUTO_TOTAL_STEPS={TOTAL_STEPS} "
    if revert_model:
        env += f"AUTO_RESUME_MODEL={revert_model} AUTO_RESUME_MODE={revert_mode} "
    subprocess.Popen(["bash", "-lc", f"{env}bash {TRAIN_SH}"], stdout=open(TRAIN_LOG, "a"),
                     stderr=subprocess.STDOUT, start_new_session=True)
    P(f"[training] launched -> {ckpt_dir}{' [REVERT '+revert_model+']' if revert_model else ''}")


def train_step():
    st = sh(f"grep -aoE '[0-9]+/{TOTAL_STEPS}' {TRAIN_LOG} 2>/dev/null | tail -1").strip()
    sp = sh(f"grep -aoE '[0-9.]+s/it' {TRAIN_LOG} 2>/dev/null | tail -1").strip()
    return f"{st or '?'} ({sp}/step)"


def latest_ckpt():
    """Highest global_step_N with an hf_model in the CURRENT generation dir."""
    best = -1
    for d in glob.glob(f"{active_dir()}/global_step_*/actor/huggingface"):
        try:
            n = int(d.split("global_step_")[1].split("/")[0])
            if os.path.exists(os.path.join(d, "config.json")):
                best = max(best, n)
        except Exception:
            pass
    return best


def get_train_curve():
    try:
        import wandb
        r = wandb.Api().runs(PROJECT, filters={"display_name": WANDB_RUN}, order="-created_at")[0]
        cand = ["critic/rewards/mean", "critic/score/mean", "actor/kl_loss", "actor/kl"]
        got = {k: [] for k in cand}
        for h in r.scan_history(keys=["_step"] + cand):
            for k in cand:
                if h.get(k) is not None:
                    got[k].append(round(float(h[k]), 5))
        return {k.split("/")[-1] + "_recent": v[-6:] for k, v in got.items() if v}
    except Exception:
        return {}


def standalone_val():
    try:
        import wandb
        r = wandb.Api().runs(PROJECT, filters={"display_name": WANDB_RUN}, order="-created_at")[0]
        ev = {}
        for h in r.scan_history(keys=["_step"] + [f"val/{d}_success_rate" for d in DSETS]):
            vs = [h.get(f"val/{d}_success_rate") for d in DSETS]
            if all(v is not None for v in vs):
                ev[h.get("_step")] = sum(vs) / 7 * 100
        return ev
    except Exception:
        return {}


# ---------- history ----------
def load_history():
    return _read_json(HIST, [])


def save_history(h):
    _write_json(HIST, h)


def action_summary(res):
    a = res.get("action")
    if not a:
        return {"applied": False, "reason": res.get("reason")}
    bs = {}
    for name, b in a.get("buckets", {}).items():
        bs[name] = {"members": [str(m).lower() for m in b.get("members", [])],
                    "p": b.get("p", a.get("default_p")),
                    "skill_preview": (b.get("skill", "") or "")[:140]}
    return {"applied": res.get("ok", False), "buckets": bs,
            "default_p": a.get("default_p"), "revert_to": a.get("revert_to")}


def available_checkpoints():
    """Checkpoint ids the controller may revert to (evaluated ones), newest first."""
    out = []
    for h in load_history():
        if h.get("step", 0) > 0 and h.get("standalone_overall") is not None:
            out.append({"id": ckpt_id_of(h.get("gen", 0), h["step"]),
                        "step": h["step"], "standalone_overall": h["standalone_overall"]})
    return out[-12:]


# ---------- controller tick ----------
def capture(n):
    hf = f"{active_dir()}/global_step_{n}/actor/huggingface"
    cap_path = f"{CAP_DIR}/cap_gen{gen_state()['gen']}_step{n}.json"
    cf = f"--counterfactual {SCAFFOLD}" if os.path.exists(SCAFFOLD) else ""
    env = (f"CUDA_VISIBLE_DEVICES={CAPTURE_GPU} VLLM_ENABLE_V1_MULTIPROCESSING=0 VLLM_ATTENTION_BACKEND=FLASH_ATTN "
           f"PATH=/mnt/data1/zha00175/miniconda/envs/verl/bin:/usr/bin:/bin")
    P(f"[capture] step {n} standalone+counterfactual on GPU{CAPTURE_GPU}...")
    rc = subprocess.run(["bash", "-lc", f"{env} {VERL} {CAPTURE} --ckpt {hf} --out {cap_path} --n 256 --max_fail 84 {cf}"],
                        capture_output=True, text=True, timeout=3600)
    P("[capture] " + (rc.stdout.strip().splitlines()[-1] if rc.stdout.strip() else rc.stderr.strip()[-200:]))
    return json.load(open(cap_path)) if os.path.exists(cap_path) else None


def do_revert(ckpt_id):
    """Auto-execute a model revert SAFELY: bump to a new generation dir so nothing is
    overwritten. Guarded by AUTO_ALLOW_REVERT + MAX_REVERTS + a valid registered id."""
    hf = _read_json(REGISTRY, {}).get(ckpt_id)
    if not hf or not os.path.exists(os.path.join(hf, "config.json")):
        P(f"[revert] unknown/invalid ckpt id {ckpt_id} -> ignored")
        email("[auto] revert ignored", f"controller asked revert_to {ckpt_id}, not a valid registered ckpt")
        return False
    s = gen_state()
    if s["reverts"] >= MAX_REVERTS:
        P(f"[revert] MAX_REVERTS={MAX_REVERTS} reached -> ignoring {ckpt_id}")
        email("[auto] revert cap hit", f"controller asked revert_to {ckpt_id} but MAX_REVERTS reached")
        return False
    if not ALLOW_REVERT:
        P(f"[revert] RECORDED but NOT executed (AUTO_ALLOW_REVERT off): {ckpt_id}")
        email("[auto] revert requested (not executed)", f"controller asked revert_to {ckpt_id}; auto-exec OFF")
        return False
    g = s["gen"] + 1
    new_dir = f"{RUN_DIR}/gen{g}"
    os.makedirs(new_dir, exist_ok=True)
    P(f"[revert] executing rollback to {ckpt_id} -> new generation gen{g}")
    for p in pids(f"main_ppo.*({TRAIN_TAG}|ckpts_search_3b_auto)"):
        try: os.kill(int(p), 9)
        except Exception: pass
    time.sleep(30)
    _write_json(GEN_STATE, {"gen": g, "active_dir": new_dir, "reverts": s["reverts"] + 1})
    launch_training(revert_model=hf, revert_mode="disable", ckpt_dir=new_dir)
    email("[auto] MODEL REVERT executed",
          f"controller reverted to {ckpt_id} ({hf}).\nNew generation gen{g} -> {new_dir} (pre-revert ckpts untouched).")
    return True


def controller_on_ckpt(n):
    g = gen_state()["gen"]
    register_ckpt(ckpt_id_of(g, n), f"{active_dir()}/global_step_{n}/actor/huggingface")
    cap = capture(n)
    if cap is None:
        P(f"[tick] no capture at step {n} -> skip"); return
    overall = cap.get("acc")
    hist = load_history()
    # backfill: previous decision's realized outcome = standalone(now) - standalone(then)
    if hist and hist[-1].get("delta_standalone") is None and hist[-1].get("standalone_overall") is not None \
            and overall is not None:
        hist[-1]["delta_standalone"] = round(overall - hist[-1]["standalone_overall"], 4)
    current = controller.load_scaffold(SCAFFOLD)
    res = controller.tick(cap, current, hist[-8:], get_train_curve(), n, SCAFFOLD, KEY_FILE,
                          available_checkpoints=available_checkpoints())
    hist.append({"step": n, "gen": g, "standalone_overall": overall,
                 "per_source": {k: v.get("acc") for k, v in cap.get("per_source", {}).items()},
                 "counterfactual": cap.get("counterfactual", {}),
                 "action": action_summary(res), "reason": res.get("reason"), "delta_standalone": None})
    save_history(hist)
    per_src = {k: v.get("acc") for k, v in cap.get("per_source", {}).items()}
    act_json = json.dumps(action_summary(res), ensure_ascii=False, indent=2)
    P(f"[tick] gen{g} step {n}: standalone={overall} applied={res.get('ok')} reason={str(res.get('reason'))[:80]}")
    email(f"[auto] controller tick @ gen{g} step {n}",
          f"standalone overall acc={overall}\nper_source={per_src}\n"
          f"counterfactual={cap.get('counterfactual')}\napplied={res.get('ok')} ({res.get('reason')})\n"
          f"action:\n{act_json}")
    if res.get("ok") and res["action"] and res["action"].get("revert_to"):
        do_revert(res["action"]["revert_to"])


def cold_start():
    cur = controller.load_scaffold(SCAFFOLD)
    if cur and cur.get("buckets"):
        P("[cold] scaffold already present -> skip blind tick"); return
    P("[cold] blind tick-0: controller writes initial scaffold with NO measurements")
    res = controller.tick(None, None, [], {}, 0, SCAFFOLD, KEY_FILE)
    if res.get("ok"):
        save_history([{"step": 0, "gen": 0, "standalone_overall": None, "per_source": {},
                       "action": action_summary(res), "reason": "cold_start", "delta_standalone": None}])
        email("[auto] cold-start (blind) scaffold",
              f"Blind tick-0 initial config:\n{json.dumps(action_summary(res), ensure_ascii=False, indent=2)}")
    else:
        P(f"[cold] blind tick failed ({res.get('reason')}) -> safe default single-bucket scaffold")
        default = {"mode": "full", "default_p": 0.5,
                   "buckets": {"all": {"members": controller.KNOWN_SOURCES, "skill": "", "p": 0.5}}}
        _write_json(SCAFFOLD, default)
        email("[auto] cold-start FELL BACK to default", f"blind GPT tick failed: {res.get('reason')}")


def status_text(deadline, note=""):
    ev = standalone_val()
    champ = max(ev.items(), key=lambda kv: kv[1]) if ev else (None, 0)
    curve = " ".join(f"s{k}={v:.1f}" for k, v in sorted(ev.items()))
    s = gen_state()
    nref = len([h for h in load_history() if h.get("step", 0) > 0])
    rem = round((deadline - time.time()) / 3600, 2)
    return (f"{note}\n=== AUTO-SCAFFOLD (base 3B, controller 全权; 零先验) ===\n"
            f"训练={'UP' if training_alive() else 'DOWN'} | 检索器={'UP' if (port_up() and retriever_ok()) else 'DOWN'} | 剩余={rem}h\n"
            f"进度: {train_step()} (共 {TOTAL_STEPS} 步) | gen={s['gen']} reverts={s['reverts']}\n"
            f"目标=STANDALONE 裸测 avg7 曲线: {curve or '(no eval yet)'}\n"
            f"** champion: step {champ[0]} = {champ[1]:.1f}  [baseline=42.9, 手调iter2=45.9] **\n"
            f"controller tick 次数={nref} | revert 自动执行={'ON' if ALLOW_REVERT else 'OFF'} (cap {MAX_REVERTS})\n"
            f"active ckpt dir: {s['active_dir']}\n"
            f"wandb run {WANDB_RUN}")


def refined_keys():
    return {ckpt_id_of(h.get("gen", 0), h["step"]) for h in load_history() if h.get("step", 0) > 0}


def _main():
    deadline = float(open(DEADLINE_FILE).read()) if os.path.exists(DEADLINE_FILE) else time.time() + HOURS * 3600
    if not os.path.exists(DEADLINE_FILE):
        open(DEADLINE_FILE, "w").write(str(deadline))
    email("[auto] supervisor online", status_text(deadline, "AUTO-SCAFFOLD 启动 (controller 全权决策, 零先验)"))
    if not (port_up() and retriever_ok()):
        launch_retriever()
    cold_start()                       # blind tick-0 writes the initial scaffold BEFORE training
    if not training_alive():
        launch_training(); time.sleep(60)
    P("[auto] supervising")
    done = refined_keys()
    last_email = 0.0
    crashes = []
    while time.time() < deadline:
        try:
            if not (port_up() and retriever_ok()):
                P("[heal] retriever down -> relaunch"); launch_retriever(); time.sleep(120)
            if not training_alive():
                crashes = [t for t in crashes if time.time() - t < 1800] + [time.time()]
                P(f"[heal] training down (crashes/30min={len(crashes)}) -> relaunch")
                if len(crashes) < 5:
                    launch_training(); time.sleep(120)
                else:
                    email("[auto] training crash-loop", status_text(deadline, "训练30min崩>=5次, 暂停30min")); time.sleep(1800); crashes = []
            n = latest_ckpt()
            g = gen_state()["gen"]
            key = ckpt_id_of(g, n)
            if n >= FIRST_REFINE and key not in done:
                try:
                    controller_on_ckpt(n)
                except Exception:
                    P("[tick] EXCEPTION\n" + traceback.format_exc())
                done = refined_keys()
            if time.time() - last_email >= 1800:
                email("[auto] 30-min 审视", status_text(deadline)); last_email = time.time()
            P(f"[ok] step={train_step()} gen={g} ckpt={n} refined={len(done)}")
        except Exception:
            P("[loop] EXCEPTION\n" + traceback.format_exc())
        time.sleep(300)
    email("[auto] window done", status_text(deadline, "窗口到, 停止自动决策"))
    P("[done] deadline reached")


def main():
    try:
        _main()
    except Exception:
        tb = traceback.format_exc(); P("[FATAL]\n" + tb); email("[auto] SUPERVISOR FATAL", tb[:1500]); raise


if __name__ == "__main__":
    main()
