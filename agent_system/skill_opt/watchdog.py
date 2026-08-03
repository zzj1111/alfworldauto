"""Supervisor: keeps the Actor server + optimize loop alive, emails status every
30 min, restarts what dies, until a persisted global deadline (~10h)."""
import os
import sys
import json
import time
import subprocess
import urllib.request

sys.path.insert(0, "/mnt/data1/zha00175/verl-agent")
from agent_system.skill_opt import notify

OUT = "/mnt/data1/zha00175/gigpo_helper_skillopt"
LOGDIR = "/mnt/data1/zha00175/gigpo_helper_logs"
VENV = "/mnt/data1/zha00175/miniconda/envs/verl"
ACTOR_MODEL = "/mnt/data1/zha00175/ckpts_alfworld/verl_agent_alfworld/0621_1310_gigpo_Qwen2.5-1.5B-Instruct_full_g8_b16_lr1e-6/global_step_150/actor/huggingface"  # GiGPO-trained 1.5B (upper-bound expt)
TOTAL_HOURS = float(os.environ.get("GIGPO_TOTAL_HOURS", "9.5"))
EMAIL_EVERY = int(os.environ.get("GIGPO_EMAIL_EVERY", "1800"))


def _sh(cmd):
    subprocess.run(cmd, shell=True, executable="/bin/bash")


def vllm_up():
    try:
        urllib.request.urlopen("http://127.0.0.1:8100/v1/models", timeout=10)
        return True
    except Exception:
        return False


def opt_alive():
    r = subprocess.run(["pgrep", "-f", "skill_opt.optimize"], capture_output=True, text=True)
    return bool(r.stdout.strip())


def start_vllm():
    _sh(f"CUDA_VISIBLE_DEVICES=0,1 VLLM_ATTENTION_BACKEND=FLASH_ATTN "
        f"PATH={VENV}/bin:$PATH CUDA_HOME={VENV} TORCH_CUDA_ARCH_LIST=9.0 "
        f"nohup {VENV}/bin/vllm serve {ACTOR_MODEL} --served-model-name actor "
        f"--tensor-parallel-size 2 --gpu-memory-utilization 0.85 --max-model-len 8192 "
        f"--host 127.0.0.1 --port 8100 >> {LOGDIR}/vllm_actor_wd.log 2>&1 &")


def wait_vllm(timeout=900):
    """Poll until vllm /v1/models responds (big models load slowly), then a small buffer."""
    t0 = time.time()
    while time.time() - t0 < timeout:
        if vllm_up():
            time.sleep(15)  # buffer for cudagraph capture
            return True
        time.sleep(10)
    return False


def start_opt(deadline):
    _sh(f"cd /mnt/data1/zha00175/verl-agent && CUDA_VISIBLE_DEVICES= "
        f"ALFWORLD_DATA=/mnt/data1/zha00175/skillzero-env/alfworld_data "
        f"GIGPO_DEADLINE_EPOCH={deadline} "
        f"nohup {VENV}/bin/python -u -m agent_system.skill_opt.optimize >> {LOGDIR}/optimize_wd.log 2>&1 &")


def main():
    os.makedirs(OUT, exist_ok=True)
    dlfile = os.path.join(OUT, "deadline.txt")
    if os.path.exists(dlfile):
        deadline = float(open(dlfile).read().strip())
    else:
        deadline = time.time() + TOTAL_HOURS * 3600
        open(dlfile, "w").write(str(deadline))

    last_email = 0.0
    if not vllm_up():
        start_vllm()
        wait_vllm()
    if not opt_alive():
        start_opt(deadline)

    while time.time() < deadline:
        try:
            healed = []
            if not vllm_up():
                start_vllm(); healed.append("vllm"); wait_vllm()
            if not opt_alive():
                start_opt(deadline); healed.append("optimize")
            if time.time() - last_email >= EMAIL_EVERY:
                st = {}
                try:
                    st = json.load(open(os.path.join(OUT, "status.json")))
                except Exception:
                    pass
                rem = (deadline - time.time()) / 3600
                hf = st.get("helper_fails", 0)
                subj = (f"[GiGPO-Helper] r{st.get('round','?')} "
                        f"best_unseen={st.get('best_eval_unseen','?')} ({rem:.1f}h left)")
                body = (
                    f"time {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
                    f"round={st.get('round')} elapsed_h={st.get('elapsed_h')} remaining_h={rem:.1f}\n"
                    f"valid_unseen baselines: none={st.get('baseline_none')} v0={st.get('baseline_v0')}\n"
                    f"BEST optimized valid_unseen = {st.get('best_eval_unseen')}\n"
                    f"accepts={st.get('accepts')}  helper_fails={hf}  helper={st.get('helper_model')}\n"
                    f"last_round={json.dumps(st.get('last_round', {}))[:700]}\n"
                    f"health: vllm={'UP' if vllm_up() else 'DOWN'} "
                    f"optimize={'UP' if opt_alive() else 'DOWN'} heals={healed}\n"
                    + ("WARNING: many helper failures — OpenAI quota may be exhausted.\n" if hf > 8 else ""))
                notify.send_email(subj, body)
                last_email = time.time()
            time.sleep(60)
        except Exception:
            time.sleep(60)

    notify.send_email("[GiGPO-Helper] watchdog: 10h window done",
                      f"Window elapsed. See {OUT}/best_skill.json, rounds.jsonl, status.json.")


if __name__ == "__main__":
    main()
