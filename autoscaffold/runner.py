"""The side-effecting adapters loop.py needs: train, eval, signals, and the A/B.

Everything heavy lives here; loop.py stays pure. GPU-touching functions are marked;
none of them is executed by the test suite.
"""
from __future__ import annotations

import glob
import json
import os
import re
import signal
import socket
import subprocess
import time

import yaml

from . import config as C
from . import scaffold as S
from . import signals as G

TRAIN_SH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "train_alfworld.sh")

# slug -> vendored alfworld task-type id (alfred_tw_env.TASK_TYPES)
TASK_ID = {"pick_and_place": 1, "look_at_obj_in_light": 2,
           "pick_clean_then_place_in_recep": 3, "pick_heat_then_place_in_recep": 4,
           "pick_cool_then_place_in_recep": 5, "pick_two_obj_and_place": 6}


class StepFailed(RuntimeError):
    pass


# ---------------- checkpoints ----------------

def ckpt_is_usable(path):
    """Complete, not merely started: every rank's model/optim/extra shard plus the
    trainer's data.pt. verl creates the directory before writing shards and updates
    latest_checkpointed_iteration.txt only afterwards, so a crash mid-save leaves a
    torn directory that must read as absent."""
    actor = os.path.join(path, "actor")
    if not os.path.isdir(actor) or not os.path.isfile(os.path.join(path, "data.pt")):
        return False
    models = glob.glob(os.path.join(actor, "model_world_size_*_rank_*.pt"))
    if not models:
        return False
    m = re.search(r"model_world_size_(\d+)_rank_", os.path.basename(models[0]))
    world = int(m.group(1)) if m else 0
    for r in range(world):
        for kind in ("model", "optim", "extra_state"):
            if not os.path.isfile(os.path.join(actor, f"{kind}_world_size_{world}_rank_{r}.pt")):
                return False
    return world > 0


def step_of(ckpt):
    try:
        return int(str(ckpt).rstrip("/").split("global_step_")[-1].split("/")[0])
    except (ValueError, IndexError):
        return 0


# ---------------- subprocess plumbing ----------------

def _run(cmd, log_path, env=None, timeout=None):
    os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
    with open(log_path, "a") as f:
        f.write(f"\n===== {time.strftime('%F %T')} $ {cmd}\n")
        f.flush()
        return subprocess.run(cmd, shell=True, stdout=f, stderr=subprocess.STDOUT,
                              env=env or os.environ.copy(), timeout=timeout)


def parse_val(log_path):
    """(overall, per_task, found) from the LAST val block of a trainer log. Parsed by
    key presence — val metrics may share a line with training metrics."""
    overall, per_task = None, {}
    try:
        with open(log_path, errors="replace") as f:
            for line in f:
                if "val/success_rate:" not in line:
                    continue
                m = re.search(r"val/success_rate:([0-9.]+)", line)
                if m:
                    overall = float(m.group(1))
                    per_task = {}
                for cat in S.CATEGORIES:
                    m = re.search(rf"val/{cat}_success_rate:([0-9.]+)", line)
                    if m:
                        per_task[cat] = float(m.group(1))
    except OSError:
        return None, {}, False
    return overall, per_task, overall is not None


# ---------------- the adapters ----------------

def train_adapter(state_scaffold_path, to_step, cfg):
    """GPU. Train (resume) to `to_step` with the current scaffold hot-loaded. Skips
    retraining when the target checkpoint is already complete (a restart mid-cycle).
    Reads back this cycle's rollout rows for the free signals."""
    ckpt = os.path.join(cfg["ckpt_dir"], f"global_step_{to_step}")
    rl_path = cfg["rollout_log"]
    offset = G.log_offset(rl_path)
    if ckpt_is_usable(ckpt):
        cfg["log"](f"[train] step {to_step} checkpoint already complete; skipping retrain")
        cfg["_rollout_rows"] = []
        return ckpt
    env = os.environ.copy()
    # ARM_PYTHON travels explicitly: the orchestrator knows its interpreter, and the
    # train script falling back to `command -v python3` silently runs another
    # environment (observed: the system python with a broken user-site pandas).
    env.update(SAVE_FREQ=str(cfg["steps_per_cycle"]), TEST_FREQ="99999",
               VAL_BEFORE="False", AUTOSCAFFOLD_ROLLOUT_LOG=rl_path,
               ARM_PYTHON=cfg["python"])
    proc = _run(f"bash {TRAIN_SH} scaffold {to_step}", cfg["train_log"], env)
    if not ckpt_is_usable(ckpt):
        raise StepFailed(f"training to step {to_step} left no usable checkpoint at {ckpt} "
                         f"(rc={proc.returncode}); see {cfg['train_log']}")
    rows = G.read_rows(rl_path, offset)
    cfg["_rollout_rows"] = rows
    by = {}
    for r in rows:
        if not r.get("injected") and r.get("task_type") in S.CATEGORIES:
            d = by.setdefault(r["task_type"], {"n_correct": 0, "n": 0})
            d["n"] += 1
            d["n_correct"] += 1 if float(r.get("success") or 0) > 0 else 0
    cfg["_last_train_rollouts"] = by
    if not rows:
        cfg["log"](f"[train] no rollout rows appeared in {rl_path}; free signals will be "
                   f"empty this cycle (is the scaffold manager active?)")
    return ckpt


def eval_adapter(ckpt, cfg):
    """GPU. VAL_N independent draws of the trainer's own validation (bare by
    construction: the val manager is the vanilla class). Each draw resumes the latest
    checkpoint, trains zero steps, and runs val_before_train."""
    draws, per_task_acc = [], {}
    tracker = os.path.join(os.path.dirname(ckpt), "latest_checkpointed_iteration.txt")

    def _tracker():
        try:
            return open(tracker).read().strip()
        except OSError:
            return ""

    before = _tracker()
    for d in range(cfg["val_n"]):
        log = C.stamped(os.path.join(cfg["state_dir"], f"eval_s{step_of(ckpt)}_d{d}.log"))
        env = os.environ.copy()
        # VAL_ONLY is what makes this an EVAL. Without it, verl's fit() resumed at
        # total_training_steps still advances one batch past the target AND saves it
        # (is_last_step forces the save): the testrun's evals silently trained and
        # checkpointed steps 3 and 5, and three draws would each build on the last
        # draw's stray update — not even measuring the same weights.
        # A different env seed per draw. vLLM v1 defaults its engine seed to 0, so
        # identical launches reproduce generations token for token — cycle 1's three
        # "independent" draws came back 0.180/0.180/0.180. Varying the env seed makes
        # each draw a different 128-game sample of the held-out split, so the spread
        # measures what it claims to (game-sampling variance, the dominant term).
        env.update(VAL_BEFORE="True", VAL_ONLY="True", TEST_FREQ="99999",
                   SAVE_FREQ="99999", ARM_WANDB="0", ARM_PYTHON=cfg["python"],
                   ENV_SEED=str(d))
        proc = _run(f"bash {TRAIN_SH} none {step_of(ckpt)}", log, env)
        overall, per_task, found = parse_val(log)
        if not found:
            cfg["log"](f"[eval] draw {d}: no val block in {log} (rc={proc.returncode})")
            continue
        draws.append(overall)
        for k, v in per_task.items():
            per_task_acc.setdefault(k, []).append(v)
    after = _tracker()
    if after != before:
        # a moved tracker means an eval WROTE a checkpoint — the exact defect
        # VAL_ONLY exists to prevent; refuse to continue on corrupted accounting
        raise StepFailed(f"eval moved the checkpoint tracker {before!r} -> {after!r}; "
                         f"an eval must never train or save")
    if not draws:
        return {"avg": None, "per_task": {}, "draws": []}
    return {"avg": round(sum(draws) / len(draws), 4),
            "per_task": {k: round(sum(v) / len(v), 4) for k, v in per_task_acc.items()},
            "draws": [round(x, 4) for x in draws]}


def signals_adapter(ckpt, scaffold, cfg):
    """Free: the rows train_adapter already read back."""
    rows = cfg.get("_rollout_rows") or []
    return G.signals_from_rows(rows, rollout_n=cfg["rollout_n"], scaffold=scaffold)


# ---------------- the A/B harness (GPU) ----------------

def free_port(start, tries=50):
    """First free port at or above `start`. Two runs sharing a machine both read the
    same default; handing out a busy port would point this run's client at the OTHER
    run's server. The caller still re-checks its own serve process is alive after the
    health wait — binding to test then releasing leaves a race."""
    for port in range(int(start), int(start) + tries):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    return int(start)


def ensure_hf(ckpt, cfg):
    """FSDP shards -> a servable HF model dir, cached beside the checkpoint."""
    target = os.path.join(ckpt, "hf")
    if os.path.isfile(os.path.join(target, "config.json")):
        return target
    log = C.stamped(os.path.join(cfg["state_dir"], f"merge_s{step_of(ckpt)}.log"))
    # `merge` is a subcommand — without it argparse rejects the call (found live;
    # the review lens assigned to check this CLI died on the subagent quota)
    proc = _run(f"{cfg['python']} {os.path.join(C.repo_root(), 'scripts', 'model_merger.py')} "
                f"merge --backend fsdp --local_dir {os.path.join(ckpt, 'actor')} "
                f"--target_dir {target}", log)
    if not os.path.isfile(os.path.join(target, "config.json")):
        raise StepFailed(f"FSDP->HF merge failed (rc={proc.returncode}); see {log}")
    # the merger does not copy tokenizer files; serve needs them
    import shutil
    base = cfg["model"]
    if os.path.isdir(base):
        for name in ("tokenizer.json", "tokenizer_config.json", "vocab.json",
                     "merges.txt", "generation_config.json", "special_tokens_map.json"):
            src = os.path.join(base, name)
            if os.path.isfile(src) and not os.path.isfile(os.path.join(target, name)):
                shutil.copy(src, target)
    return target


def _cat_config(category, tmp_dir):
    """A per-category copy of the vendored alfworld config: the env scans only that
    task type, which is how the A/B spends its whole budget on touched categories."""
    src = os.path.join(C.repo_root(), "agent_system", "environments", "env_package",
                       "alfworld", "configs", "config_tw.yaml")
    with open(src) as f:
        conf = yaml.safe_load(f)
    conf["env"]["task_types"] = [TASK_ID[category]]
    out = os.path.join(tmp_dir, f"config_{category}.yaml")
    os.makedirs(tmp_dir, exist_ok=True)
    with open(out, "w") as f:
        yaml.safe_dump(conf, f)
    return out


class _Actor:
    """Minimal OpenAI-compatible client against the served checkpoint."""

    def __init__(self, base_url, model="actor", temperature=0.4, max_workers=64):
        from concurrent.futures import ThreadPoolExecutor
        import openai
        self.cli = openai.OpenAI(base_url=base_url, api_key="EMPTY", timeout=180,
                                 max_retries=2)
        self.model = model
        self.temperature = temperature
        self.pool = ThreadPoolExecutor(max_workers=max_workers)

    def healthy(self):
        try:
            self.cli.models.list()
            return True
        except Exception:
            return False

    def _one(self, prompt):
        try:
            r = self.cli.chat.completions.create(
                model=self.model, temperature=self.temperature, max_tokens=512,
                messages=[{"role": "user", "content": prompt}])
            return r.choices[0].message.content or ""
        except Exception:
            return ""

    def generate(self, prompts):
        return list(self.pool.map(self._one, prompts))


def _rollout_once(manager, actor, max_steps=50):
    """One episode per env. Returns [(gamefile, success)]."""
    import numpy as np
    obs, _ = manager.reset({})
    n = len(obs["text"])
    done = np.zeros(n, dtype=bool)
    won = np.zeros(n, dtype=float)
    for _ in range(max_steps):
        replies = actor.generate(obs["text"])
        obs, rewards, dones, infos = manager.step(replies)
        for i in range(n):
            if not done[i] and dones[i]:
                done[i] = True
                won[i] = float(infos[i].get("won") or 0.0)
        if done.all():
            break
    return [(manager.gamefile[i], float(won[i])) for i in range(n)]


def measure_ab_adapter(ckpt, current, candidate, tasks, cfg):
    """GPU. Frozen-policy paired A/B on HELD-OUT games of the touched categories.

    A fixed episode budget per condition (ARM_AB_EPISODES, default 180) is split
    across the touched categories, so the measurement's resolution does not shrink
    when the Teacher narrows after a rejection. Conditions are paired: the env stack
    for one category is rebuilt per condition with the SAME seed, so worker i replays
    the same game sequence. The held-out pools are small (28-43 games per category on
    valid_seen); the replay factor is logged, not capped.
    """
    import types

    from autoscaffold.scaffold_env_manager import ScaffoldAlfWorldEnvironmentManager

    log = cfg["log"]
    episodes = int(cfg.get("ab_episodes") or os.environ.get("ARM_AB_EPISODES", "180"))
    n_cats = max(1, len(tasks))
    n_per = max(1, episodes // n_cats)

    hf = ensure_hf(ckpt, cfg)
    port = free_port(int(cfg.get("vllm_port") or 8110))
    vlog = C.stamped(os.path.join(cfg["state_dir"], f"ab_vllm_s{step_of(ckpt)}.log"))
    serve_env = os.environ.copy()
    serve_env["CUDA_VISIBLE_DEVICES"] = cfg["gpus"]
    vl = subprocess.Popen(
        f"{cfg['python']} -m vllm.entrypoints.openai.api_server --model {hf} "
        f"--served-model-name actor --tensor-parallel-size {cfg['tp']} "
        f"--gpu-memory-utilization 0.85 --max-model-len 4096 --enforce-eager "
        f"--host 127.0.0.1 --port {port}",
        shell=True, stdout=open(vlog, "w"), stderr=subprocess.STDOUT,
        env=serve_env, start_new_session=True)

    conditions = [("bare", S.empty_scaffold()), ("candidate", candidate)]
    if not S.injects_nothing(current):
        conditions.insert(1, ("current", current))
    result = {name: {} for name, _ in conditions}
    managers = []
    try:
        actor = _Actor(f"http://127.0.0.1:{port}/v1",
                       max_workers=int(cfg.get("ab_workers") or 96))
        deadline = time.time() + int(cfg.get("vllm_health_timeout") or 2400)
        while time.time() < deadline and not actor.healthy():
            if vl.poll() is not None:
                raise StepFailed(f"vLLM serve exited early (rc={vl.returncode}); see {vlog}")
            time.sleep(10)
        # A healthy endpoint alone proves only that SOMEONE serves on this port; our
        # own process must still be alive or we are measuring another run's model.
        if vl.poll() is not None:
            raise StepFailed(f"vLLM serve died while port {port} still answers — another "
                             f"server holds it; see {vlog}")
        if not actor.healthy():
            raise StepFailed(f"vLLM not healthy before timeout; see {vlog}")

        from agent_system.environments.env_package.alfworld import (build_alfworld_envs,
                                                                    alfworld_projection)
        seed = int(cfg.get("base_seed") or 20260722) + step_of(ckpt)
        env_cfg = types.SimpleNamespace(env=types.SimpleNamespace(
            rollout=types.SimpleNamespace(n=1), history_length=2))
        for cat in tasks:
            pool = _held_out_pool_size(cat, cfg)
            if pool:
                log(f"[ab] {cat}: {n_per} episodes over {pool} distinct held-out games"
                    + (f" (x{n_per / pool:.1f} replay)" if n_per > pool else ""))
            conf = _cat_config(cat, os.path.join(cfg["state_dir"], "ab_configs"))
            for name, scaf in conditions:
                envs = build_alfworld_envs(
                    conf, seed, n_per, 1,
                    resources_per_worker={"num_cpus": 0.1}, is_train=False,
                    env_kwargs={"eval_dataset": cfg.get("eval_split",
                                                        "eval_in_distribution")})
                mgr = ScaffoldAlfWorldEnvironmentManager(envs, alfworld_projection, env_cfg)
                mgr._scaffold_override = scaf
                mgr._force_all = True         # the sanctioned exception: forced onto held-out
                mgr._record_path = ""         # the recorder is for training only
                managers.append(mgr)
                outcomes = _rollout_once(mgr, actor)
                wins = float(sum(w for _, w in outcomes))
                # plain python numbers: numpy scalars here poison every JSON sink
                # downstream (journal, status.json) with a TypeError
                result[name][cat] = (round(wins / len(outcomes), 4), int(len(outcomes)))
                log(f"[ab] {cat} {name}: {wins:.0f}/{len(outcomes)}")
    finally:
        for mgr in managers:
            try:
                mgr.close()
            except Exception:
                pass
        try:
            import ray
            if ray.is_initialized():
                ray.shutdown()
        except Exception:
            pass
        try:
            pgid = os.getpgid(vl.pid)
            os.killpg(pgid, signal.SIGTERM)
            t0 = time.time()
            while time.time() - t0 < 20 and vl.poll() is None:
                time.sleep(1)
            if vl.poll() is None:
                os.killpg(pgid, signal.SIGKILL)
        except Exception:
            pass
        try:
            vl.wait(timeout=5)
        except Exception:
            pass
    if "current" not in result:
        result["current"] = dict(result["bare"])   # empty current == bare by construction
    return result


def _held_out_pool_size(category, cfg):
    split = "valid_unseen" if cfg.get("eval_split") == "eval_out_of_distribution" \
        else "valid_seen"
    d = os.path.join(C.alfworld_data(), "json_2.1.1", split)
    prefix = {"pick_and_place": "pick_and_place_simple"}.get(category, category)
    try:
        return sum(1 for name in os.listdir(d) if name.startswith(prefix))
    except OSError:
        return 0
