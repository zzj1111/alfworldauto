"""Phase 2: from the diagnostic, have GPT-5.5 propose GENERAL strategic skills
(principles that transfer), NOT game-specific patches. Saves skills_general.json."""
import os
import sys
import json
import re

sys.path.insert(0, "/mnt/data1/zha00175/verl-agent")
from openai import OpenAI
from agent_system.skill.skill_store import ALFWORLD_TASK_TYPES

OUT = "/mnt/data1/zha00175/gigpo_helper_skillopt"

_SYS = """You are an expert coach for an ALFWorld text agent that is ALREADY WELL-TRAINED (~88% success) and makes essentially NO invalid or repeated actions. Its ONLY residual failures are TIMEOUTS: on hard episodes it uses all 50 steps without completing the task, due to STRATEGY errors (wrong order of sub-steps, placing in the wrong receptacle, not completing all required sub-goals, inefficient search).

Write GENERAL strategic skill hints that fix these strategy errors and will be injected during further training. HARD RULES:
- GENERAL ONLY. State transferable PRINCIPLES. NEVER reference specific object names, object/receptacle ID numbers, room layouts, or particular locations. A good skill reads like a rule of thumb that applies to EVERY game of that task type.
- It must NOT be a "patch" for a specific game. If a hint only helps one layout, it is wrong.
- Target the OBSERVED failure modes: (1) re-read the goal and identify the exact TARGET object and TARGET receptacle; place ONLY into the target receptacle; (2) do multi-step tasks in the correct ORDER (e.g., you must be HOLDING the object before using a lamp / heating / cooling / cleaning); (3) complete ALL sub-goals (e.g., "two objects" means place a SECOND identical object too); (4) search efficiently — check the most likely locations first, open closed receptacles, and don't revisit ones already searched empty, to finish within the step budget.
- Keep each skill 1-4 concise sentences.
- Output STRICT JSON: {"general_skill": "<text>", "skills": {"<task_type>": "<text>", ...}, "rationale": "<short>"}. Valid task types: """ + ", ".join(ALFWORLD_TASK_TYPES) + "."


def _load_key():
    for line in open("/mnt/data1/zha00175/tool-agent-secrets/openai.env"):
        if line.startswith("OPENAI_API_KEY="):
            return line.strip().split("=", 1)[1]


def main():
    diag = json.load(open(os.path.join(OUT, "diagnostic.json")))
    parts = [f"Overall success {diag['overall_success']} (split={diag['split']}). Per-task residual failures:\n"]
    for tt in ALFWORLD_TASK_TYPES:
        a = diag["per_task"].get(tt, {})
        if not a:
            continue
        parts.append(f"\n=== {tt}: success {a['success_rate']} ({a['n_failed']} failed; all timeouts at 50 steps, "
                     f"invalid_frac~{a.get('avg_invalid_frac')}, repeat_frac~{a.get('avg_repeat_frac')}) ===")
        for s in a.get("sample_failures", [])[:2]:
            parts.append("[FAILED trajectory: obs -> action]\n" + s)
    parts.append("\nPropose GENERAL skills (no specific objects/IDs/locations) that fix these strategy errors. STRICT JSON only.")
    user = "\n".join(parts)

    c = OpenAI(api_key=_load_key())
    model = os.environ.get("OPENAI_HELPER_MODEL", "gpt-5.5")
    r = c.chat.completions.create(model=model,
        messages=[{"role": "system", "content": _SYS}, {"role": "user", "content": user}],
        response_format={"type": "json_object"}, max_completion_tokens=4000)
    txt = r.choices[0].message.content
    try:
        d = json.loads(txt)
    except Exception:
        m = re.search(r"\{.*\}", txt, re.S); d = json.loads(m.group(0))
    skills = {k: v for k, v in (d.get("skills") or {}).items() if k in ALFWORLD_TASK_TYPES}
    out = {"mode": "full", "default_p": 1.0, "general_skill": d.get("general_skill", ""),
           "skills": skills, "p_task": {k: 1.0 for k in ALFWORLD_TASK_TYPES}}
    json.dump(out, open(os.path.join(OUT, "skills_general.json"), "w"), indent=2)
    print("RATIONALE:", d.get("rationale", ""))
    print("\nGENERAL:", out["general_skill"])
    for k, v in skills.items():
        print(f"\n[{k}] {v}")
    print("\nsaved skills_general.json")


if __name__ == "__main__":
    main()
