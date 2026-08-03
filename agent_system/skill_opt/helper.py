"""GPT-5.5 Helper: reads rollouts + per-task success, rewrites the skill scaffold.

Supports an OpenAI backend (default, GPT-5.5) and a local OpenAI-compatible
backend (vllm teacher) used as fallback when the OpenAI quota is exhausted.
"""
import json
import os
import re
import random
from openai import OpenAI

from agent_system.skill.skill_store import ALFWORLD_TASK_TYPES

_SYS = """You are an expert coach for an ALFWorld text-agent. You write concise, reusable SKILL HINTS that get prepended to a FROZEN agent's prompt so it solves more tasks. The agent cannot be retrained, so the hints must do the work.

You receive, per task type: the current skill, the recent success rate, and sample trajectories (mostly FAILURES, each a sequence of observation -> action). Diagnose the failure modes and rewrite the hints to fix them.

Rules:
- Each skill: 1-4 short, concrete sentences. Use exact ALFWorld verbs: "go to", "open", "take X from Y", "put X in/on Y", "use X", "heat X with microwave", "cool X with fridge", "clean X with sinkbasin".
- Target the ACTUAL failure patterns you observe (e.g. not opening closed receptacles, putting before arriving, not holding the object, repeating invalid actions, hallucinating actions not in the admissible list, wrong appliance).
- For task types CURRENTLY AT 0% success, the existing recipe is NOT working for this weak agent — propose a GENUINELY DIFFERENT, simpler, more explicit strategy (e.g., spell out the exact action order, name the most common locations to search first, warn against the specific invalid actions seen).
- For task types ALREADY succeeding (>=20%), keep their skill essentially as-is; do not risk breaking what works.
- Be general: NO specific object/receptacle numbers that won't transfer.
- Output STRICT JSON only, schema:
  {"general_skill": "<text>", "skills": {"<task_type>": "<text>", ...}, "rationale": "<short>"}
  Only include task_type keys you want to change. Valid task types: """ + ", ".join(ALFWORLD_TASK_TYPES) + "."


def _compact_traj(t, max_steps=12):
    lines = []
    for s in t["steps"][:max_steps]:
        obs = re.sub(r"\s+", " ", s["obs"])[:160]
        flag = "" if s.get("valid", True) else " [INVALID]"
        lines.append(f"  obs: {obs}\n  -> {s['action']}{flag} (r={s['reward']})")
    return "\n".join(lines)


def _build_user_msg(summary, transcripts, current, per_task_samples=3):
    by_task = {}
    for t in transcripts:
        by_task.setdefault(t["task_type"], []).append(t)
    parts = [f"Overall success: {summary['overall']:.3f} over {summary['n']} games.\n"]
    for tt in ALFWORLD_TASK_TYPES:
        st = summary["per_task"].get(tt)
        if not st:
            continue
        cur = current.get("skills", {}).get(tt, "(none)")
        parts.append(f"\n=== TASK: {tt} | success {st['success']:.3f} (n={st['n']}) ===")
        parts.append(f"current skill: {cur}")
        pool = by_task.get(tt, [])
        fails = [t for t in pool if not t["success"]]
        succ = [t for t in pool if t["success"]]
        random.shuffle(fails); random.shuffle(succ)
        chosen = fails[:per_task_samples] + succ[:1]
        for t in chosen:
            tag = "SUCCESS" if t["success"] else "FAIL"
            parts.append(f"[{tag}, {t['n_steps']} steps]\n{_compact_traj(t)}")
    parts.append(f"\nCurrent general_skill: {current.get('general_skill','(none)')}")
    parts.append("\nRewrite the skills to raise success. Output STRICT JSON only.")
    return "\n".join(parts)


def _parse_json(text):
    text = (text or "").strip()
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                return None
    return None


class HelperClient:
    def __init__(self, model="gpt-5.5", api_key=None, base_url=None,
                 max_completion_tokens=6000, use_temperature=False):
        kw = {}
        if api_key:
            kw["api_key"] = api_key
        if base_url:
            kw["base_url"] = base_url
        self.client = OpenAI(timeout=240, max_retries=2, **kw)
        self.model = model
        self.max_completion_tokens = max_completion_tokens
        self.use_temperature = use_temperature  # reasoning models reject custom temp

    def propose(self, summary, transcripts, current):
        user = _build_user_msg(summary, transcripts, current)
        kw = dict(model=self.model,
                  messages=[{"role": "system", "content": _SYS},
                            {"role": "user", "content": user}],
                  max_completion_tokens=self.max_completion_tokens)
        try:
            kw["response_format"] = {"type": "json_object"}
            if self.use_temperature:
                kw["temperature"] = 0.7
            r = self.client.chat.completions.create(**kw)
        except Exception:
            kw.pop("response_format", None)
            r = self.client.chat.completions.create(**kw)
        out = _parse_json(r.choices[0].message.content)
        if not isinstance(out, dict):
            return None, "parse_failed"
        skills = {k: v for k, v in (out.get("skills") or {}).items()
                  if k in ALFWORLD_TASK_TYPES and isinstance(v, str) and v.strip()}
        gen = out.get("general_skill")
        gen = gen.strip() if isinstance(gen, str) and gen.strip() else None
        return {"skills": skills, "general_skill": gen,
                "rationale": str(out.get("rationale", ""))[:500]}, "ok"
