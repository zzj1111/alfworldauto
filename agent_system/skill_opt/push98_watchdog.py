"""Watchdog for the push98 autonomous training loop.
- Persists a 10h deadline (survives restarts).
- (Re)starts push98_loop if it dies, until the deadline.
- Emails a status summary every 30 min (status.json + recent log + train progress).
- GPUs 0,1 only (the training script pins them).
"""
import os, sys, time, json, subprocess, glob
sys.path.insert(0, "/mnt/data1/zha00175/verl-agent")
from agent_system.skill_opt import notify

ROOT = "/mnt/data1/zha00175/verl-agent"
OUT = os.environ.get("PUSH98_OUT", "/mnt/data1/zha00175/gigpo_helper_skillopt/push98")
EXP = os.environ.get("PUSH98_EXP", "skilltrain_push98")
LOGDIR = "/mnt/data1/zha00175/gigpo_helper_logs"
VENV_PY = "/mnt/data1/zha00175/miniconda/envs/verl/bin/python"
DEADLINE_FILE = f"{OUT}/deadline.txt"
EMAIL_EVERY = 30 * 60
HOURS = float(os.environ.get("PUSH98_HOURS", "10"))
os.makedirs(OUT, exist_ok=True)


def now():
    return time.time()


def get_deadline():
    if os.path.exists(DEADLINE_FILE):
        try:
            return float(open(DEADLINE_FILE).read().strip())
        except Exception:
            pass
    dl = now() + HOURS * 3600
    open(DEADLINE_FILE, "w").write(str(dl))
    return dl


def loop_pid():
    out = subprocess.run(["bash", "-lc",
        "ps -eo pid,cmd | grep 'skill_opt.push98_loop' | grep -v grep | grep -v 'bash -lc' | awk '{print $1}'"],
        capture_output=True, text=True).stdout.strip()
    return out.split()[0] if out else None


def start_loop(deadline):
    env = dict(os.environ)
    env["GIGPO_DEADLINE_EPOCH"] = str(deadline)
    env.setdefault("PUSH98_K", "12")
    env.setdefault("PUSH98_GOAL", "0.98")
    env.setdefault("OPENAI_HELPER_MODEL", "gpt-5.5")
    lf = open(f"{OUT}/push98_loop.log", "a")
    subprocess.Popen([VENV_PY, "-u", "-m", "agent_system.skill_opt.push98_loop"],
                     cwd=ROOT, stdout=lf, stderr=subprocess.STDOUT, env=env, start_new_session=True)


def train_progress():
    logs = sorted(glob.glob(f"{LOGDIR}/{EXP}_train_c*.log") + glob.glob(f"{LOGDIR}/{EXP}_eval_c*.log"),
                  key=os.path.getmtime)
    if not logs:
        return "(no train log yet)"
    f = logs[-1]
    try:
        lines = subprocess.run(["bash", "-lc",
            f"grep -aE 'step:[0-9]+|global_step|val/success_rate:|Error|Traceback' {f} | tail -3"],
            capture_output=True, text=True).stdout.strip()
    except Exception:
        lines = ""
    return f"[{os.path.basename(f)}]\n{lines or '(starting...)'}"


def status_text(deadline):
    st = {}
    try:
        st = json.load(open(f"{OUT}/status.json"))
    except Exception:
        pass
    try:
        orch = subprocess.run(["bash", "-lc", f"tail -6 {OUT}/orch.log"], capture_output=True, text=True).stdout
    except Exception:
        orch = ""
    rem = round((deadline - now()) / 3600, 2)
    alive = "UP" if loop_pid() else "DOWN"
    return (f"loop={alive}  remaining={rem}h\n"
            f"status.json: {json.dumps(st, ensure_ascii=False)}\n\n"
            f"--- train progress ---\n{train_progress()}\n\n"
            f"--- orch.log tail ---\n{orch}")


def main():
    deadline = get_deadline()
    notify.send_email("[push98 watchdog] online",
                      f"Supervising autonomous train<->skill loop toward standalone valid_unseen >0.98.\n"
                      f"Deadline in {round((deadline-now())/3600,2)}h. Emailing status every 30 min.")
    last_email = 0.0
    while now() < deadline:
        if not loop_pid():
            # don't restart if the loop already declared success/finish
            done = False
            try:
                done = json.load(open(f"{OUT}/status.json")).get("best_standalone_unseen", 0) >= float(os.environ.get("PUSH98_GOAL", "0.98"))
            except Exception:
                pass
            if done:
                break
            notify.send_email("[push98 watchdog] restarting loop", status_text(deadline))
            start_loop(deadline)
            time.sleep(45)
        if now() - last_email >= EMAIL_EVERY:
            notify.send_email("[push98] 30-min status", status_text(deadline))
            last_email = now()
        time.sleep(60)
    notify.send_email("[push98 watchdog] deadline reached", status_text(deadline))


if __name__ == "__main__":
    main()
