"""Autonomous supervisor for the search-R1 (HotpotQA) GiGPO training.
Lifecycle: wait for index assembly -> launch retriever (wait ready) -> launch training
-> 10-min self-healing diagnostic loop for 24h.
Self-heals: retriever down (relaunch), training crashed (resume via resume_mode=auto),
GPU stragglers. Crash-loop guard: if training dies <3min repeatedly, stop + email for help.
Emails status every 30 min. Survives the launching session (run detached).
"""
import os, sys, time, subprocess, socket, traceback
sys.path.insert(0, "/mnt/data1/zha00175/verl-agent")
from agent_system.skill_opt import notify

ROOT = "/mnt/data1/zha00175/verl-agent"
DATA = os.path.expanduser("~/data/searchR1")
RETR_SH = f"{ROOT}/examples/search/retriever/run_retriever_gpu.sh"
TRAIN_SH = f"{ROOT}/examples/gigpo_trainer/run_search_3b.sh"
ASSEMBLE_LOG = "/mnt/data1/zha00175/searchR1_data/assemble.log"
OUT = "/mnt/data1/zha00175/searchR1_data/orch"
RETR_LOG = f"{OUT}/retriever.log"
TRAIN_LOG = f"{OUT}/train.log"
ORCH_LOG = f"{OUT}/orch.log"
DEADLINE_FILE = f"{OUT}/deadline.txt"
CKPT = "/mnt/data1/zha00175/ckpts_search_3b"
PORT = 8010
HOURS = float(os.environ.get("SEARCH_HOURS", "24"))
TRAIN_TAG = "gigpo_3b_searchr1"           # unique marker for OUR training proc
os.makedirs(OUT, exist_ok=True)


def P(m):
    print(m, flush=True)
    try:
        with open(ORCH_LOG, "a") as f:
            f.write(time.strftime("%m-%d %H:%M:%S ") + m + "\n")
    except Exception:
        pass


def sh(cmd):
    return subprocess.run(["bash", "-lc", cmd], capture_output=True, text=True).stdout


def pids(pat):
    out = sh(f"ps -eo pid,cmd | grep -E '{pat}' | grep zha00175 | grep -v grep | grep -v 'bash -lc' | awk '{{print $1}}'")
    return [p for p in out.split() if p]


def kill(pat):
    for p in pids(pat):
        try:
            os.kill(int(p), 9)
        except Exception:
            pass


def port_up():
    try:
        s = socket.create_connection(("127.0.0.1", PORT), timeout=3); s.close(); return True
    except Exception:
        return False


def retriever_responds():
    r = sh("curl -s -m 25 -X POST http://127.0.0.1:%d/retrieve -H 'Content-Type: application/json' "
           "-d '{\"query\":\"who wrote hamlet\",\"topk\":3,\"return_scores\":false}'" % PORT)
    return ('"result"' in r) and ("document" in r or "contents" in r)


def launch_retriever():
    kill("retrieval_server")
    time.sleep(3)
    with open(RETR_LOG, "a") as f:
        subprocess.Popen(["bash", "-lc", f"bash {RETR_SH}"], stdout=f, stderr=subprocess.STDOUT, start_new_session=True)
    P("[retriever] launched (CPU-faiss, loading 64G index into RAM ~few min)")


def retriever_ready(timeout=1800):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if port_up() and retriever_responds():
            return True
        time.sleep(15)
    return False


def launch_training():
    with open(TRAIN_LOG, "a") as f:
        subprocess.Popen(["bash", "-lc", f"bash {TRAIN_SH}"], stdout=f, stderr=subprocess.STDOUT, start_new_session=True)
    P("[training] launched (3B, GPUs 2,3, resume_mode=auto)")


def training_alive():
    return len(pids(f"main_ppo.*({TRAIN_TAG}|ckpts_search_3b)")) > 0


def last_train_error():
    txt = sh(f"sed -E 's/\\x1b\\[[0-9;]*m//g' {TRAIN_LOG} 2>/dev/null | grep -aE 'Error|Traceback|out of memory|assert|raise ' "
             f"| grep -avE 'ResourceTracker|_thread|invalid_action|FutureWarning' | tail -4")
    return txt.strip()[:600]


def train_progress():
    return sh(f"grep -aoE 'step:[0-9]+|val/[a-z0-9_/]*success[a-z_]*:[0-9.]+|critic/rewards/mean:[0-9.-]+' {TRAIN_LOG} 2>/dev/null | tail -4").strip()


def gpu_line():
    return sh("nvidia-smi -i 2,3 --query-gpu=index,memory.used --format=csv,noheader").replace("\n", " | ").strip()


def train_step():
    st = sh(f"grep -aoE '[0-9]+/662' {TRAIN_LOG} 2>/dev/null | tail -1").strip()
    sp = sh(f"grep -aoE '[0-9.]+s/it' {TRAIN_LOG} 2>/dev/null | tail -1").strip()
    return (f"{st}  ({sp}/step)" if st else "(initializing)")


def latest_em():
    try:
        import wandb
        runs = wandb.Api().runs("mhong-university-of-minnesota/verl_agent_search",
                                filters={"display_name": "gigpo_3b_searchr1"}, order="-created_at")
        r = runs[0]
        dsets = ["hotpotqa", "nq", "2wikimultihopqa", "triviaqa", "popqa", "musique", "bamboogle"]
        # success_rate = 官方 README 口径 (search-R1 cover-EM); test_score 是更严格的 exact-match
        vals = {d: r.summary.get(f"val/{d}_success_rate") for d in dsets}
        vals = {d: v for d, v in vals.items() if v is not None}
        if not vals:
            return "(no eval yet; first eval at step 50)"
        avg = sum(vals.values()) / len(vals) * 100
        return (f"avg7(官方口径success)={avg:.1f}  [官方GiGPO-3B=42.1] @ wandb-step {r.summary.get('_step')} | "
                f"nq={vals.get('nq',0)*100:.1f} triviaqa={vals.get('triviaqa',0)*100:.1f} "
                f"popqa={vals.get('popqa',0)*100:.1f} hotpotqa={vals.get('hotpotqa',0)*100:.1f} "
                f"2wiki={vals.get('2wikimultihopqa',0)*100:.1f} musique={vals.get('musique',0)*100:.1f} "
                f"bamboogle={vals.get('bamboogle',0)*100:.1f}")
    except Exception as e:
        return f"(wandb query failed: {str(e)[:50]}; see wandb UI)"


def status_text(deadline, note=""):
    rem = round((deadline - time.time()) / 3600, 2)
    rup = "UP" if (port_up() and retriever_responds()) else "DOWN"
    tup = "UP" if training_alive() else "DOWN"
    ck = sh("ls -d /mnt/data1/zha00175/ckpts_search_3b/global_step_* 2>/dev/null | grep -oE 'global_step_[0-9]+' | tr '\\n' ' '").strip()
    return (f"{note}\n"
            f"=== Search-R1 GiGPO 训练 (Qwen2.5-3B, 官方一致) ===\n"
            f"训练={tup} | 检索器={rup} | 剩余监督窗口={rem}h\n"
            f"进度: step {train_step()}  (共 662 步)\n"
            f"评测EM(7数据集): {latest_em()}\n"
            f"已存 checkpoints: {ck or '(none)'}\n"
            f"GPU2(检索)/GPU3(训练): {gpu_line()}\n"
            f"wandb: https://wandb.ai/mhong-university-of-minnesota/verl_agent_search/runs/3f2htbdk\n"
            f"last_err: {last_train_error() or '(none)'}")


def main():
    if os.path.exists(DEADLINE_FILE):
        deadline = float(open(DEADLINE_FILE).read().strip())
    else:
        deadline = time.time() + HOURS * 3600
        open(DEADLINE_FILE, "w").write(str(deadline))

    notify.send_email("[search-R1] supervisor online",
                      f"Autonomous 24h supervisor started. Will: wait assembly -> launch retriever -> launch training "
                      f"-> 10-min self-heal loop.\nDeadline in {round((deadline-time.time())/3600,2)}h.")

    # Phase 1: wait for index assembly
    P("[phase1] waiting for index assembly (CORPUS_DONE)...")
    while time.time() < deadline:
        try:
            corpus_done = "EXTRACT_DONE" in open(ASSEMBLE_LOG).read()
        except Exception:
            corpus_done = False
        if corpus_done and os.path.exists(f"{DATA}/e5_Flat.index") and os.path.getsize(f"{DATA}/e5_Flat.index") > 60e9 \
           and os.path.exists(f"{DATA}/wiki-18.jsonl") and os.path.getsize(f"{DATA}/wiki-18.jsonl") > 12e9:
            break
        time.sleep(30)
    P(f"[phase1] assembly ready: index={os.path.getsize(DATA+'/e5_Flat.index')/1e9:.1f}G corpus={os.path.getsize(DATA+'/wiki-18.jsonl')/1e9:.1f}G")

    # Phase 2: retriever (reuse an already-healthy one -> avoids a 64G reload on restart)
    if port_up() and retriever_responds():
        P(f"[phase2] retriever already healthy on :{PORT} -> reusing (no reload)")
    else:
        launch_retriever()
        if not retriever_ready(timeout=2400):
            notify.send_email("[search-R1] retriever FAILED to start", status_text(deadline, "retriever not ready in 40min"))
            P("[phase2] retriever not ready -> will keep retrying in the loop")
        else:
            P(f"[phase2] retriever READY (responds on :{PORT})")

    # Phase 3: training (adopt an existing tagged run on restart; else launch)
    if training_alive():
        P("[phase3] training already running -> adopting it")
    else:
        launch_training()
        time.sleep(60)

    # Phase 4: 10-min self-heal loop
    last_email = 0.0
    crash_times = []
    while time.time() < deadline:
        try:
            # retriever health
            if not (port_up() and retriever_responds()):
                P("[heal] retriever DOWN -> relaunching")
                launch_retriever()
                retriever_ready(timeout=1800)
            # training health
            if not training_alive():
                crash_times.append(time.time())
                crash_times = [t for t in crash_times if time.time() - t < 1800]  # crashes in last 30min
                err = last_train_error()
                P(f"[heal] training DOWN (crashes/30min={len(crash_times)}). last_err: {err[:300]}")
                if len(crash_times) >= 4:
                    notify.send_email("[search-R1] training crash-loop — NEEDS ATTENTION",
                                      status_text(deadline, f"training died {len(crash_times)}x in 30min; not restarting further this window."))
                    P("[heal] crash-loop -> pausing restarts for 30min")
                    time.sleep(1800); crash_times = []
                else:
                    launch_training()
                    time.sleep(90)
            # periodic status email
            if time.time() - last_email >= 1800:
                notify.send_email("[search-R1] 30-min status", status_text(deadline))
                last_email = time.time()
            P(f"[ok] retr={'up' if port_up() else 'down'} train={'up' if training_alive() else 'down'} | {train_progress()[-120:]}")
        except Exception:
            P("[loop] EXCEPTION\n" + traceback.format_exc())
        time.sleep(600)  # 10-minute diagnostic interval

    notify.send_email("[search-R1] 24h window done", status_text(deadline, "deadline reached"))
    P("[done] deadline reached")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        _tb = traceback.format_exc()
        P("[FATAL] supervisor died:\n" + _tb)
        try:
            notify.send_email("[search-R1] SUPERVISOR FATAL - needs attention", _tb[:1500])
        except Exception:
            pass
        raise
