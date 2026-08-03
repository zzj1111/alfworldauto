"""Auto-scaffold CONTROLLER (GPT-5.5) for the Search-R1 Teacher-Student experiment.

Replaces the hand-tuned harness (fixed withdrawal `scaffold_p()` + a fix-list-laden
rewrite prompt). The controller is fed the full measured STATE and decides FOUR actions
with ZERO outcome priors — it is told the environment and the control interface, never
which values work:

  1. content    -- per-bucket skill text
  2. injection  -- per-bucket p + default_p
  3. granularity-- the bucket partition over data_sources (free re-bucketing)
  4. stability  -- revert_to an earlier checkpoint (or null)

Cadence is NOT an action (fixed by the supervisor). The controller discovers any
injection/bucketing policy online, from per-source standalone accuracy, the with/without-
scaffold counterfactual, failure trajectories, and its OWN decision->outcome history.
"""
import json
import os

# The data_source universe the controller must partition (benchmark-given, not a prior).
KNOWN_SOURCES = ["nq", "triviaqa", "popqa", "hotpotqa", "2wikimultihopqa", "musique", "bamboogle"]
MODEL = "gpt-5.5"
DEFAULT_KEY_FILE = "/mnt/data1/zha00175/.openai_key"

_FAIL_CHARS = 150000   # cap on serialized failure trajectories fed to the controller


def load_scaffold(path):
    """Current scaffold config (buckets form). Returns None if absent/unreadable."""
    try:
        return json.load(open(path))
    except Exception:
        return None


def assemble_state(cap, current, history, train_curve, step, available_checkpoints=None):
    """Build the controller's OBSERVATION. `cap` is capture_search.py output (or None on the
    blind cold start). Only measurements — no advice."""
    if cap is None:
        return {
            "step": step,
            "objective": "maximize standalone (no-scaffold) cover-EM accuracy, per data_source",
            "data_sources": KNOWN_SOURCES,
            "note": "COLD START: no evaluation yet. Produce an initial configuration blind.",
            "current_scaffold": current,
            "history": history or [],
        }
    return {
        "step": step,
        "objective": "maximize standalone (no-scaffold) cover-EM accuracy, per data_source",
        "data_sources": KNOWN_SOURCES,
        "standalone": {"overall_acc": cap.get("acc"), "per_source": cap.get("per_source", {})},
        "counterfactual_with_vs_without_scaffold": cap.get("counterfactual", {}),
        "failure_modes": cap.get("failure_modes", {}),
        "failure_trajectories": cap.get("failures", []),
        "successes_for_contrast": cap.get("successes_for_contrast", []),
        "current_scaffold": current,
        "decision_history": history or [],   # [{step, action_summary, delta_standalone}]
        "available_checkpoints": available_checkpoints or [],  # revert_to must be one of these ids
        "train_curve": train_curve or {},
    }


_PROMPT_HEAD = """You are the CONTROLLER of an automated RL training experiment. A weak Qwen2.5-3B agent is being trained with reinforcement learning (GiGPO) on Search-R1, a multi-hop QA benchmark.

ENVIRONMENT (facts about the task, NOT advice):
- Each episode the agent answers one question in at most 4 turns. Each turn it emits <think>...</think> then EITHER <search>query</search> (returns 3 retrieved wiki passages) OR <answer>...</answer> (ends the episode). Correct = the gold string is covered by the answer (cover-EM).
- The data_sources are: {sources}.

CONTROL INTERFACE (what you decide):
- You partition the data_sources into BUCKETS. Every data_source goes into EXACTLY ONE bucket.
- Each bucket carries a "skill": a text hint that, DURING TRAINING ONLY, is spliced into the agent's prompt with probability p (the bucket's injection probability). Empty skill or p=0 => nothing injected for that bucket.
- The agent is EVALUATED WITHOUT any skill text. The ONLY objective is the standalone (no-scaffold) per-source accuracy. Nothing else is scored.
- You may set "revert_to": one of the checkpoint ids listed in state.available_checkpoints to roll the model back to that earlier checkpoint, or null to keep training forward. Do NOT invent ids.

You are given the measured STATE below, INCLUDING your own past decisions and the standalone-accuracy change each produced. You are given NO guidance about which values work — infer everything from the state and your history.

STATE:
{state}

Return ONLY JSON with this exact shape:
{{"diagnosis": "<your reasoning, logged not executed>",
  "buckets": {{"<bucket_name>": {{"members": ["<data_source>", ...], "skill": "<hint text>", "p": <float 0..1>}}}},
  "default_p": <float 0..1>,
  "revert_to": null}}"""


def render_prompt(state):
    body = json.dumps(state, ensure_ascii=False)
    budget = _FAIL_CHARS + 20000
    if len(body) > budget and state.get("failure_trajectories"):
        # trim the heaviest field (failure trajectories) until under budget; keep the rest intact
        s = dict(state)
        fails = list(state["failure_trajectories"])
        while fails and len(json.dumps({**s, "failure_trajectories": fails}, ensure_ascii=False)) > budget:
            fails = fails[:-1]
        s["failure_trajectories"] = fails
        body = json.dumps(s, ensure_ascii=False)
    return _PROMPT_HEAD.format(sources=KNOWN_SOURCES, state=body)


def validate(action):
    """Enforce PHYSICAL well-formedness only (no policy). Returns (ok, reason)."""
    if not isinstance(action, dict):
        return False, "action is not an object"
    buckets = action.get("buckets")
    if not isinstance(buckets, dict) or not buckets:
        return False, "buckets missing or empty"
    seen, members_all = set(), []
    for name, b in buckets.items():
        if not isinstance(b, dict):
            return False, f"bucket {name} is not an object"
        mem = b.get("members")
        if not isinstance(mem, list) or not mem:
            return False, f"bucket {name} has no members"
        for m in mem:
            k = str(m).strip().lower()
            if k not in KNOWN_SOURCES:
                return False, f"unknown data_source '{m}' in bucket {name}"
            if k in seen:
                return False, f"data_source '{k}' assigned to more than one bucket"
            seen.add(k); members_all.append(k)
        p = b.get("p", action.get("default_p"))
        if p is not None and not (0.0 <= float(p) <= 1.0):
            return False, f"bucket {name} p={p} out of [0,1]"
        if not isinstance(b.get("skill", ""), str):
            return False, f"bucket {name} skill is not a string"
    missing = set(KNOWN_SOURCES) - seen
    if missing:
        return False, f"data_sources not assigned to any bucket: {sorted(missing)}"
    dp = action.get("default_p", 0.5)
    if not (0.0 <= float(dp) <= 1.0):
        return False, f"default_p={dp} out of [0,1]"
    rv = action.get("revert_to")
    if rv is not None and not isinstance(rv, str):
        return False, "revert_to must be null or a checkpoint id string"
    return True, "ok"


def to_scaffold_json(action):
    """Normalize a validated action into the scaffold JSON the injection path reads."""
    dp = float(action.get("default_p", 0.5))
    buckets = {}
    for name, b in action["buckets"].items():
        p = b.get("p", dp)
        buckets[name] = {
            "members": [str(m).strip().lower() for m in b["members"]],
            "skill": b.get("skill", "") or "",
            "p": float(p) if p is not None else dp,
        }
    return {"mode": "full", "default_p": dp, "buckets": buckets}


def write_scaffold(action, path):
    out = to_scaffold_json(action)
    tmp = path + ".tmp"
    json.dump(out, open(tmp, "w"), indent=2, ensure_ascii=False)
    os.replace(tmp, path)   # atomic -> training hot-reloads via mtime
    return out


def call_gpt(prompt, key_file=DEFAULT_KEY_FILE, max_tokens=6000):
    from openai import OpenAI
    cli = OpenAI(api_key=open(key_file).read().strip(), timeout=600, max_retries=2)
    r = cli.chat.completions.create(model=MODEL, messages=[{"role": "user", "content": prompt}],
                                    max_completion_tokens=max_tokens, response_format={"type": "json_object"})
    return json.loads(r.choices[0].message.content)


def tick(cap, current, history, train_curve, step, scaffold_path, key_file=DEFAULT_KEY_FILE,
         available_checkpoints=None):
    """One control tick: assemble state -> GPT -> validate -> write scaffold.
    Returns {ok, action, reason}. On invalid output the scaffold is left unchanged."""
    state = assemble_state(cap, current, history, train_curve, step, available_checkpoints)
    prompt = render_prompt(state)
    try:
        action = call_gpt(prompt, key_file)
    except Exception as e:
        return {"ok": False, "action": None, "reason": f"gpt call failed: {str(e)[:200]}"}
    ok, reason = validate(action)
    if not ok:
        return {"ok": False, "action": action, "reason": f"invalid action: {reason}"}
    write_scaffold(action, scaffold_path)
    return {"ok": True, "action": action, "reason": "applied"}
