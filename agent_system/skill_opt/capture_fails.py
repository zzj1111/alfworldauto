"""Roll out the CURRENT checkpoint (standalone, mode=none, official sampling) and dump
fail+success trajectories, so GPT-5.5 can read WHY the model fails when it rewrites skills.
Flow: merge FSDP->HF (if needed) -> serve vllm on GPU 0,1 -> balanced rollout -> save JSON -> kill vllm.
Robust: on any failure it still exits and frees the GPU; the loop falls back to numbers-only refine.
"""
import os, sys, json, subprocess, time, signal, shutil, glob
sys.path.insert(0, "/mnt/data1/zha00175/verl-agent")
os.environ["ALFWORLD_DATA"] = "/mnt/data1/zha00175/skillzero-env/alfworld_data"
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # this driver process touches no GPU; vllm subprocess does

STEP = os.environ["CAP_STEP"]
EXP = os.environ.get("CAP_EXP", "skilltrain_push98_seen")
SPLIT = os.environ.get("CAP_SPLIT", "eval_in_distribution")
NPT = int(os.environ.get("CAP_NPT", "10"))           # games/type for the balanced fallback
NPT_WEAK = int(os.environ.get("CAP_NPT_WEAK", "30")) # games for EACH weak task (targeted)
WEAK_K = int(os.environ.get("CAP_WEAK_K", "3"))      # how many weakest tasks to target
WEAK_THRESH = float(os.environ.get("CAP_WEAK_THRESH", "0.95"))
OUTJSON = os.environ["CAP_OUTJSON"]


def weak_tasks_from_status():
    """Read the latest per-task success (status.json beside OUTJSON) and return the
    weakest tasks to concentrate rollouts on. None -> caller uses balanced fallback."""
    try:
        st = json.load(open(f"{os.path.dirname(OUTJSON)}/status.json"))
        per = st.get("last_per_task") or {}
        if not per:
            return None
        ranked = sorted(per.items(), key=lambda kv: kv[1])
        weak = [t for t, v in ranked if v < WEAK_THRESH][:WEAK_K]
        if not weak:                       # everything strong -> still take the bottom 2
            weak = [t for t, _ in ranked[:2]]
        return weak
    except Exception:
        return None
VENV = "/mnt/data1/zha00175/miniconda/envs/verl"
BASE_HF = os.environ.get("CAP_BASE_HF", "/mnt/data1/zha00175/ckpts_alfworld/verl_agent_alfworld/0621_1310_gigpo_Qwen2.5-1.5B-Instruct_full_g8_b16_lr1e-6/global_step_150/actor/huggingface")
CKACT = f"/mnt/data1/zha00175/gigpo_helper_ckpts/{EXP}/global_step_{STEP}/actor"
HFDIR = f"{CKACT}/huggingface"


def sh(cmd, **kw):
    return subprocess.run(["bash", "-lc", cmd], **kw)


def kill_vllm():
    sh("for p in $(ps -eo pid,cmd | grep -E 'EngineCore|VllmWorker|VLLM::|vllm serve' | grep zha00175 "
       "| grep -v grep | grep -v 'bash -lc' | awk '{print $1}'); do kill -9 $p 2>/dev/null; done")


def main():
    # 1. merge FSDP -> HF if not present
    if not os.path.exists(f"{HFDIR}/config.json"):
        print("[capture] merging FSDP->HF ...", flush=True)
        sh(f"cd /mnt/data1/zha00175/verl-agent && PATH={VENV}/bin:$PATH {VENV}/bin/python "
           f"scripts/model_merger.py merge --backend fsdp --local_dir {CKACT} --target_dir {HFDIR}", check=True)
    # ensure tokenizer / config files exist (copy from base if the merge omitted them)
    for f in glob.glob(f"{BASE_HF}/*"):
        b = os.path.basename(f)
        if (b.startswith("tokenizer") or b in ("vocab.json", "merges.txt", "special_tokens_map.json",
                                               "generation_config.json", "config.json")) \
           and not os.path.exists(f"{HFDIR}/{b}"):
            shutil.copy(f, f"{HFDIR}/{b}")
    # 2. serve vllm
    print("[capture] serving vllm ...", flush=True)
    vlog = f"{os.path.dirname(OUTJSON)}/capture_vllm_step{STEP}.log"
    CAP_GPUS = os.environ.get('CAP_GPUS', '0,1')
    vl = subprocess.Popen(["bash", "-lc",
        f"CUDA_VISIBLE_DEVICES={CAP_GPUS} VLLM_ATTENTION_BACKEND=FLASH_ATTN PATH={VENV}/bin:$PATH CUDA_HOME={VENV} "
        f"TORCH_CUDA_ARCH_LIST=9.0 {VENV}/bin/vllm serve {HFDIR} --served-model-name actor "
        f"--tensor-parallel-size 2 --gpu-memory-utilization 0.85 --max-model-len 8192 "
        f"--host 127.0.0.1 --port 8100"],
        stdout=open(vlog, "w"), stderr=subprocess.STDOUT, start_new_session=True)
    try:
        from agent_system.skill_opt.actor import ActorClient
        # OFFICIAL sampling (match the eval GPT is optimizing): temp0.4 / top_p1.0 / top_k-1 / no rep
        actor = ActorClient(temperature=0.4, top_p=1.0, top_k=-1, repetition_penalty=1.0, max_tokens=512)
        t0 = time.time()
        while time.time() - t0 < 600:
            if actor.healthy():
                break
            time.sleep(10)
        else:
            raise RuntimeError("vllm not ready in 600s")
        from agent_system.skill_opt.envs import build_balanced_managers, attach_skill
        from agent_system.skill_opt.rollout import run_balanced_rollout, summarize
        from agent_system.skill.skill_store import SkillStore
        # TARGETED: concentrate rollouts on the weakest tasks so real failures are dense.
        weak = weak_tasks_from_status()
        if weak:
            print(f"[capture] targeting weak tasks {weak} x{NPT_WEAK} games each", flush=True)
            mgrs = build_balanced_managers(NPT_WEAK, seed=12345, is_train=False,
                                           eval_dataset=SPLIT, only_types=weak)
        else:
            print(f"[capture] no per-task info -> balanced {NPT}/type", flush=True)
            mgrs = build_balanced_managers(NPT, seed=12345, is_train=False, eval_dataset=SPLIT)
        # (A) standalone (mode=none) = the official condition; these failures are what GPT reads.
        #     This is also a DENSER per-task signal than the 14-game eval (less noise on weak tasks).
        for _n, m in mgrs:
            attach_skill(m, SkillStore(mode="none"))
        tr = run_balanced_rollout(mgrs, actor, max_steps=50, temperature=0.4)
        s = summarize(tr)
        nfail = sum(1 for t in tr if not t.get("success"))
        # (B) COUNTERFACTUAL: re-roll the SAME managers WITH the skill always injected.
        #     Answers the decisive question: does the model even USE the skill on the weak tasks?
        cf = None
        try:
            skill_path = (os.environ.get("PUSH98_SKILL") or os.environ.get("GIGPO_SKILL_PATH")
                          or f"/mnt/data1/zha00175/gigpo_helper_skillopt/skills_{EXP.replace('skilltrain_','')}.json")
            for _n, m in mgrs:
                attach_skill(m, SkillStore.from_json(skill_path, mode="full"))
            s2 = summarize(run_balanced_rollout(mgrs, actor, max_steps=50, temperature=0.4))
            cf = {"skill_path": skill_path, "overall_none": s["overall"], "overall_with_skill": s2["overall"],
                  "none": {k: v["success"] for k, v in s["per_task"].items()},
                  "with_skill": {k: v["success"] for k, v in s2["per_task"].items()}}
            print("[capture] COUNTERFACTUAL — does the skill help on the weak tasks?", flush=True)
            for k in s["per_task"]:
                a = s["per_task"][k]["success"]; b = s2["per_task"].get(k, {}).get("success", 0.0)
                print(f"    {k}: none={a:.3f}  with_skill={b:.3f}  delta={b - a:+.3f}", flush=True)
        except Exception as e:
            print("[capture] counterfactual failed:", str(e)[:200], flush=True)
        json.dump({"step": STEP, "split": SPLIT, "weak_tasks": weak, "overall": s["overall"],
                   "per_task": s["per_task"], "counterfactual": cf, "transcripts": tr},
                  open(OUTJSON, "w"), default=str, indent=1)
        print(f"[capture] saved {len(tr)} transcripts ({nfail} failures) overall={s['overall']:.3f} -> {OUTJSON}", flush=True)
    finally:
        try:
            os.killpg(os.getpgid(vl.pid), signal.SIGKILL)
        except Exception:
            pass
        kill_vllm()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        print("[capture] FAILED:", traceback.format_exc(), flush=True)
        kill_vllm()
        sys.exit(1)
