"""Drive the frozen Actor through ALFWorld and collect transcripts + per-task success."""
import re
import numpy as np
from collections import defaultdict

from agent_system.skill.skill_store import detect_task_type

_ACTION_RE = re.compile(r"<action>(.*?)</action>", re.DOTALL)


def _to_bool(x, default=False):
    try:
        return bool(np.asarray(x).item())
    except Exception:
        try:
            return bool(x)
        except Exception:
            return default


def _extract_action(resp):
    m = _ACTION_RE.search(resp or "")
    return (m.group(1).strip() if m else (resp or "").strip())[:160]


def normalize_tags(resp):
    """The env parser requires <think>..</think> AND <action>..</action> (angle brackets).
    Models (esp. 7B) deviate: square brackets [action], OR an opening tag with NO closing
    tag (e.g. '[action] go to cabinet 1'), OR missing <think>. Repair these so we measure
    the model's DECISION, not its formatting compliance."""
    if not resp:
        return resp
    # 1) square -> angle brackets for both tags
    for tag in ("think", "action"):
        resp = re.sub(rf"\[\s*{tag}\s*\]", f"<{tag}>", resp, flags=re.I)
        resp = re.sub(rf"\[\s*/\s*{tag}\s*\]", f"</{tag}>", resp, flags=re.I)
    # 2) close <action> if the model omitted </action> (close at end of that line / string)
    if "<action>" in resp and "</action>" not in resp:
        idx = resp.find("<action>")
        nl = resp.find("\n", idx)
        resp = (resp + "</action>") if nl == -1 else (resp[:nl] + "</action>" + resp[nl:])
    # 3) ensure <think>..</think> exists (parser rejects outputs lacking it)
    if "<think>" in resp and "</think>" not in resp:
        a = resp.find("<action>")
        resp = (resp[:a] + "</think>\n" + resp[a:]) if a != -1 else (resp + "</think>")
    if "<think>" not in resp and "<action>" in resp:
        resp = "<think>reasoning</think>\n" + resp
    return resp


def run_rollout(manager, actor, max_steps=50, temperature=None, max_transcript_steps=60):
    """Returns a list of per-episode transcripts with success flags."""
    obs, infos = manager.reset(None)
    texts, anchors = obs["text"], obs["anchor"]
    N = len(texts)
    active = [True] * N
    won = [0.0] * N
    gamefiles = [str(g) for g in manager.gamefile]
    transcripts = [{
        "idx": i,
        "task_type": detect_task_type(gamefiles[i]),
        "gamefile": gamefiles[i],
        "skill_injected": bool(manager.skill_inject[i]) if manager.skill_inject else False,
        "steps": [],
    } for i in range(N)]

    for _ in range(max_steps):
        gen_idx = [i for i in range(N) if active[i]]
        if not gen_idx:
            break
        sub = actor.generate([texts[i] for i in gen_idx], temperature=temperature)
        responses = [""] * N
        for k, i in enumerate(gen_idx):
            responses[i] = normalize_tags(sub[k])
        cur_anchor = anchors
        next_obs, rewards, dones, infos = manager.step(responses)
        for i in range(N):
            if not active[i]:
                continue
            info = infos[i]
            won[i] = max(won[i], float(info.get("won", 0.0) or 0.0))
            if len(transcripts[i]["steps"]) < max_transcript_steps:
                transcripts[i]["steps"].append({
                    "obs": str(cur_anchor[i])[:500],
                    "action": _extract_action(responses[i]),
                    "valid": _to_bool(info.get("is_action_valid", True), True),
                    "reward": round(float(rewards[i]), 3),
                })
            if _to_bool(dones[i]):
                active[i] = False
        texts, anchors = next_obs["text"], next_obs["anchor"]

    for i in range(N):
        transcripts[i]["success"] = bool(won[i] > 0.5)
        transcripts[i]["n_steps"] = len(transcripts[i]["steps"])
    return transcripts


def run_balanced_rollout(managers, actor, max_steps=50, temperature=None, max_transcript_steps=60):
    """managers: list of (task_name, manager), one per task type (balanced n_per_type each).
    Lockstep over all managers: ONE batched generate call per step across all active envs
    (keeps vllm batching high), then each manager steps its envs (Ray-parallel)."""
    states = []
    for _name, mgr in managers:
        obs, _ = mgr.reset(None)
        n = len(obs["text"])
        gamefiles = [str(g) for g in mgr.gamefile]
        states.append({
            "mgr": mgr, "text": obs["text"], "anchor": obs["anchor"],
            "active": [True] * n, "won": [0.0] * n,
            "tr": [{"idx": i, "task_type": detect_task_type(gamefiles[i]), "gamefile": gamefiles[i],
                    "skill_injected": bool(mgr.skill_inject[i]) if mgr.skill_inject else False,
                    "steps": []} for i in range(n)],
        })

    for _ in range(max_steps):
        flat_prompts, ref = [], []
        for si, st in enumerate(states):
            for i in range(len(st["active"])):
                if st["active"][i]:
                    flat_prompts.append(st["text"][i]); ref.append((si, i))
        if not flat_prompts:
            break
        flat_resps = actor.generate(flat_prompts, temperature=temperature)  # one batched call
        resp = {k: normalize_tags(v) for k, v in zip(ref, flat_resps)}
        for si, st in enumerate(states):
            n = len(st["active"])
            responses = [resp.get((si, i), "") for i in range(n)]
            cur_anchor = st["anchor"]
            next_obs, rewards, dones, infos = st["mgr"].step(responses)
            for i in range(n):
                if not st["active"][i]:
                    continue
                info = infos[i]
                st["won"][i] = max(st["won"][i], float(info.get("won", 0.0) or 0.0))
                if len(st["tr"][i]["steps"]) < max_transcript_steps:
                    st["tr"][i]["steps"].append({
                        "obs": str(cur_anchor[i])[:500], "action": _extract_action(responses[i]),
                        "valid": _to_bool(info.get("is_action_valid", True), True),
                        "reward": round(float(rewards[i]), 3)})
                if _to_bool(dones[i]):
                    st["active"][i] = False
            st["text"], st["anchor"] = next_obs["text"], next_obs["anchor"]

    all_tr = []
    for st in states:
        for i in range(len(st["active"])):
            st["tr"][i]["success"] = bool(st["won"][i] > 0.5)
            st["tr"][i]["n_steps"] = len(st["tr"][i]["steps"])
            all_tr.append(st["tr"][i])
    return all_tr


def summarize(transcripts):
    agg = defaultdict(list)
    for t in transcripts:
        agg[t["task_type"]].append(1.0 if t["success"] else 0.0)
    per_task = {k: {"success": round(sum(v) / len(v), 3), "n": len(v)}
                for k, v in sorted(agg.items())}
    overall = round(sum(1.0 for t in transcripts if t["success"]) / max(1, len(transcripts)), 3)
    return {"overall": overall, "n": len(transcripts), "per_task": per_task}
