"""Autonomous 24h orchestrator for the ITERATIVE Teacher-Student scaffold experiment.
Continuous training (base 3B, scaffold injected on train, STANDALONE val on subset).
Every new checkpoint (~50 steps): capture standalone failure trajectories -> GPT-5.5
rewrites the scaffold -> write scaffold JSON (training hot-reloads it). Tracks the
champion by STANDALONE val (the objective = pure model, no scaffold). 30-min email
reviews, self-heals training/retriever, runs until the 24h deadline. Fixes bugs by
wrapping every step in try/except and emailing on trouble.
"""
import os, sys, time, json, subprocess, socket, traceback, glob
sys.path.insert(0, "/mnt/data1/zha00175/verl-agent")
from agent_system.skill_opt import notify

ROOT = "/mnt/data1/zha00175/verl-agent"
VERL = "/mnt/data1/zha00175/miniconda/envs/verl/bin/python3"
RETR_SH = f"{ROOT}/examples/search/retriever/run_retriever_gpu.sh"
TRAIN_SH = f"{ROOT}/examples/gigpo_trainer/run_search_3b_iter.sh"
CAPTURE = f"{ROOT}/agent_system/skill_opt/capture_search.py"
OUT = "/mnt/data1/zha00175/searchR1_data/orch_iter2"
TRAIN_LOG = f"{OUT}/train.log"
ORCH_LOG = f"{OUT}/orch.log"
CKPT_DIR = "/mnt/data1/zha00175/ckpts_search_3b_iter2"
SCAFFOLD = "/mnt/data1/zha00175/searchR1_data/search_skills_iter.json"
CAP_DIR = f"{OUT}/captures"
DEADLINE_FILE = f"{OUT}/deadline.txt"
KEY_FILE = "/mnt/data1/zha00175/.openai_key"
PORT = 8010
TRAIN_TAG = "gigpo_3b_iter2"
WANDB_RUN = "gigpo_3b_iter2"
HOURS = float(os.environ.get("ITER_HOURS", "24"))
STEP_OFFSET = 150   # new run starts from the step-150 weights
N_GPUS = 2
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
    return len(pids(f"main_ppo.*({TRAIN_TAG}|ckpts_search_3b_iter)")) > 0


def ckpt_world_size_ok():
    """FSDP ckpts are world_size-sharded. Resuming a ckpt saved with a different
    world_size raises FileNotFoundError (this killed the previous run). Guard it."""
    shards = glob.glob(f"{CKPT_DIR}/global_step_*/actor/model_world_size_*_rank_0.pt")
    bad = [x for x in shards if f"world_size_{N_GPUS}_" not in x]
    if bad:
        P(f"[guard] ckpt world_size MISMATCH (need {N_GPUS}): {bad[:2]}")
        email("[iter2] ckpt world_size 不匹配 - 需人工处理",
              f"ckpt dir {CKPT_DIR} 里有非 world_size={N_GPUS} 的分片, resume 会崩:\n{bad[:5]}")
        return False
    return True


def launch_training():
    if not ckpt_world_size_ok():
        P("[guard] refusing to launch until ckpt/world_size resolved"); return
    subprocess.Popen(["bash", "-lc", f"bash {TRAIN_SH}"], stdout=open(TRAIN_LOG, "a"),
                     stderr=subprocess.STDOUT, start_new_session=True)
    P("[training] launched (base 3B, GPU1, scaffold-train / standalone-val)")


def train_step():
    st = sh(f"grep -aoE '[0-9]+/662' {TRAIN_LOG} 2>/dev/null | tail -1").strip()
    sp = sh(f"grep -aoE '[0-9.]+s/it' {TRAIN_LOG} 2>/dev/null | tail -1").strip()
    return f"{st or '?'} ({sp}/step)"


def latest_ckpt():
    """Highest global_step_N with an hf_model (huggingface) dir ready."""
    best = -1
    for d in glob.glob(f"{CKPT_DIR}/global_step_*/actor/huggingface"):
        try:
            n = int(d.split("global_step_")[1].split("/")[0])
            if os.path.exists(os.path.join(d, "config.json")):
                best = max(best, n)
        except Exception:
            pass
    return best


def scaffold_p(step):
    """Withdrawal curriculum: scaffold guides early, model must practice the BARE prompt
    later so that STANDALONE (no-scaffold) eval -- the objective -- does not suffer
    train/eval distribution shift."""
    if step < 100: return 1.0
    if step < 200: return 0.6
    if step < 300: return 0.4
    return 0.3


def cur_step():
    """Effective step = raw step + STEP_OFFSET (we resumed from step-150 weights)."""
    st = sh(f"grep -aoE '[0-9]+/662' {TRAIN_LOG} 2>/dev/null | tail -1").strip()
    try: return int(st.split("/")[0]) + STEP_OFFSET
    except Exception: return STEP_OFFSET


def set_scaffold_p(p):
    """Write default_p into the scaffold JSON (training hot-reloads it)."""
    try:
        d = json.load(open(SCAFFOLD)); 
        if abs(float(d.get("default_p", -1)) - p) < 1e-6: return False
        d["default_p"] = p
        tmp = SCAFFOLD + ".tmp"; json.dump(d, open(tmp,"w"), indent=2, ensure_ascii=False); os.replace(tmp, SCAFFOLD)
        P(f"[withdrawal] scaffold 注入概率 p -> {p}")
        return True
    except Exception:
        P("[withdrawal] failed\n" + traceback.format_exc()); return False


def standalone_val():
    """Return {step: avg7} dict from wandb (val is standalone via the is_train guard)."""
    try:
        import wandb
        r = wandb.Api().runs("mhong-university-of-minnesota/verl_agent_search",
                             filters={"display_name": WANDB_RUN}, order="-created_at")[0]
        ev = {}
        for h in r.scan_history(keys=["_step"] + [f"val/{d}_success_rate" for d in DSETS]):
            vs = [h.get(f"val/{d}_success_rate") for d in DSETS]
            if all(v is not None for v in vs):
                ev[h.get("_step")] = sum(vs) / 7 * 100
        return ev
    except Exception as e:
        P(f"[val-query-fail] {str(e)[:80]}")
        return {}


def gpt_refine(data):
    """GPT-5.5 rewrites the scaffold from MANY standalone failure trajectories. Writes SCAFFOLD."""
    key = open(KEY_FILE).read().strip()
    cur = json.load(open(SCAFFOLD))
    from openai import OpenAI
    cli = OpenAI(api_key=key, timeout=600, max_retries=2)
    failures = data.get("failures", [])
    oks = data.get("successes_for_contrast", [])
    modes = data.get("failure_modes", {})
    acc = data.get("acc", 0)
    # GPT-5.5 has a large context: send many full trajectories
    fail_txt = json.dumps(failures[:60], ensure_ascii=False)[:150000]
    ok_txt = json.dumps(oks[:6], ensure_ascii=False)[:12000]
    n_run = data.get("n_run", 0); n_correct = data.get("n_correct", 0); n_fail = len(failures[:60])
    prompt = f"""You are improving a text "scaffold" (strategy hints) that is injected into a WEAK Qwen2.5-3B agent's prompt DURING RL training on Search-R1 multi-hop QA. The agent has at most 4 turns; each turn it writes <think>..</think> then EITHER <search>query</search> (gets top-3 wiki passages) OR <answer>..</answer>. The scaffold is a TRAINING crutch; the goal is that the model, trained WITH the scaffold, gets better when evaluated WITHOUT it.

CURRENT scaffold:
single_hop: {cur['skills'].get('single_hop','')}
multi_hop: {cur['skills'].get('multi_hop','')}

Diagnostics of the CURRENT model, evaluated STANDALONE (no scaffold) on {n_run} training questions:
- standalone accuracy (cover-EM): {acc:.1%}  ({n_correct}/{n_run} correct)
- failure-mode counts: {modes}

Below are {n_fail} FULL FAILURE trajectories (question, gold, each turn's model output + the search query it issued + what was retrieved, and its final answer). Study the failure MODES across the WHOLE set — do not overfit to one example:
{fail_txt}

For contrast, here are a few SUCCESS trajectories (what working behavior looks like):
{ok_txt}

Rewrite BOTH scaffolds (single_hop, multi_hop) to fix the observed failure modes. Be concrete and imperative. Hard requirements based on known issues: (a) RESPECT the 4-turn budget — commit to an <answer> by turn 3, never exhaust the budget without answering; (b) do NOT repeat the same search; (c) if a search is unhelpful, reformulate with different keywords rather than re-searching; (d) answers should be the exact minimal entity/phrase (not a whole sentence, not a single vague word). Keep each scaffold under ~120 words, starting with a one-line header. Return ONLY JSON: {{"single_hop":"...","multi_hop":"..."}}."""
    r = cli.chat.completions.create(model="gpt-5.5", messages=[{"role": "user", "content": prompt}],
                                    max_completion_tokens=4000, response_format={"type": "json_object"})
    new = json.loads(r.choices[0].message.content)
    out = {"skills": {"single_hop": new["single_hop"], "multi_hop": new["multi_hop"]},
           "general_skill": "", "p_task": {}, "mode": "full",
           "default_p": float(cur.get("default_p", 1.0))}
    tmp = SCAFFOLD + ".tmp"
    json.dump(out, open(tmp, "w"), indent=2, ensure_ascii=False)
    os.replace(tmp, SCAFFOLD)  # atomic -> training hot-reloads via mtime
    return new


def refine_on_ckpt(n, deadline):
    """Capture standalone failures from ckpt n, then GPT-5.5 refines the scaffold."""
    hf = f"{CKPT_DIR}/global_step_{n}/actor/huggingface"
    cap = f"{CAP_DIR}/fail_step{n}.json"
    P(f"[refine] capturing standalone failures from step {n} (GPU3)...")
    env = (f"CUDA_VISIBLE_DEVICES=3 VLLM_ENABLE_V1_MULTIPROCESSING=0 VLLM_ATTENTION_BACKEND=FLASH_ATTN "
           f"PATH=/mnt/data1/zha00175/miniconda/envs/verl/bin:/usr/bin:/bin")
    rc = subprocess.run(["bash", "-lc", f"{env} {VERL} {CAPTURE} --ckpt {hf} --out {cap} --n 256 --max_fail 80"],
                        capture_output=True, text=True, timeout=2400)
    P("[refine] capture: " + (rc.stdout.strip().splitlines()[-1] if rc.stdout.strip() else rc.stderr.strip()[-200:]))
    if not os.path.exists(cap):
        P("[refine] no capture output -> skip refine this cycle"); return None
    data = json.load(open(cap))
    fails = data.get("failures", [])
    if len(fails) < 5:
        P(f"[refine] only {len(fails)} failures (model doing well) -> keep scaffold"); return data
    new = gpt_refine(data)
    P(f"[refine] step {n}: {len(fails)} failures -> GPT-5.5 rewrote scaffold. multi_hop[:80]={new['multi_hop'][:80]}")
    email(f"[iter] scaffold refined @ step {n}",
          f"Captured {len(fails)} standalone failures (of {data.get('n_run')} run, acc={data.get('acc')}). modes={data.get('failure_modes')}\n"
          f"GPT-5.5 rewrote the scaffold. New multi_hop:\n{new['multi_hop']}\n\nNew single_hop:\n{new['single_hop']}")
    return data


def status_text(deadline, note=""):
    ev = standalone_val()
    champ = max(ev.items(), key=lambda kv: kv[1]) if ev else (None, 0)
    curve = " ".join(f"s{k}={v:.1f}" for k, v in sorted(ev.items()))
    try:
        ver = json.load(open(SCAFFOLD)).get("skills", {}).get("multi_hop", "")[:70]
    except Exception:
        ver = "?"
    nrefine = len(glob.glob(f"{CAP_DIR}/fail_step*.json"))
    ck = sh(f"ls -d {CKPT_DIR}/global_step_* 2>/dev/null | grep -oE 'global_step_[0-9]+' | tr '\\n' ' '").strip()
    rem = round((deadline - time.time()) / 3600, 2)
    return (f"{note}\n=== 迭代 Teacher-Student scaffold (base 3B, GPU1) ===\n"
            f"训练={'UP' if training_alive() else 'DOWN'} | 检索器={'UP' if (port_up() and retriever_ok()) else 'DOWN'} | 剩余={rem}h\n"
            f"进度: raw {train_step()} | 实际step≈{cur_step()} (从step150权重续)\n"
            f"目标=STANDALONE裸测 avg7(success) 曲线: {curve or '(no eval yet)'}\n"
            f"** 当前 champion(裸测最优): step {champ[0]} = {champ[1]:.1f}  [baseline无scaffold=42.9] **\n"
            f"scaffold 已被 GPT-5.5 改进 {nrefine} 次 | 注入概率p={json.load(open(SCAFFOLD)).get('default_p')} (withdrawal) | multi_hop[:70]: {ver}\n"
            f"checkpoints: {ck or '(none)'}\n"
            f"GPU1,6(训练)/GPU3(capture): {sh('nvidia-smi -i 1,6,3 --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader').strip()}\n"
            f"wandb: https://wandb.ai/mhong-university-of-minnesota/verl_agent_search (run {WANDB_RUN})")


def _main():
    deadline = float(open(DEADLINE_FILE).read()) if os.path.exists(DEADLINE_FILE) else time.time() + HOURS * 3600
    if not os.path.exists(DEADLINE_FILE):
        open(DEADLINE_FILE, "w").write(str(deadline))
    email("[iter] autonomous supervisor online", status_text(deadline, "迭代 scaffold 实验启动 (24h, 每30min汇报)"))
    if not (port_up() and retriever_ok()):
        launch_retriever()
    if not training_alive():
        launch_training(); time.sleep(60)
    P("[iter] supervising")
    last_refined = -1
    last_email = 0.0
    crashes = []
    while time.time() < deadline:
        try:
            if not (port_up() and retriever_ok()):
                P("[heal] retriever down -> relaunch"); launch_retriever(); time.sleep(120)
            if not training_alive():
                crashes = [t for t in crashes if time.time() - t < 1800] + [time.time()]
                P(f"[heal] training down (crashes/30min={len(crashes)}) -> relaunch (resume)")
                if len(crashes) < 5:
                    launch_training(); time.sleep(120)
                else:
                    email("[iter] training crash-loop", status_text(deadline, "训练30min内崩>=5次, 暂停重启30min")); time.sleep(1800); crashes = []
            set_scaffold_p(scaffold_p(cur_step()))
            n = latest_ckpt()
            if n > last_refined and n >= 50:
                try:
                    refine_on_ckpt(n, deadline)
                except Exception:
                    P("[refine] EXCEPTION\n" + traceback.format_exc())
                last_refined = n
            if time.time() - last_email >= 1800:
                email("[iter] 30-min 审视", status_text(deadline)); last_email = time.time()
            P(f"[ok] step={train_step()} ckpt={n} refined_upto={last_refined}")
        except Exception:
            P("[loop] EXCEPTION\n" + traceback.format_exc())
        time.sleep(300)
    email("[iter] 24h window done", status_text(deadline, "24h到, 训练可继续但停止自动改scaffold"))
    P("[done] deadline reached")


def main():
    try:
        _main()
    except Exception:
        tb = traceback.format_exc(); P("[FATAL]\n" + tb); email("[iter] SUPERVISOR FATAL", tb[:1500]); raise


if __name__ == "__main__":
    main()
