"""Domain-agnostic AUTO-SCAFFOLD controller (GPT-5.5). ONE brain for Search-R1 + ALFWorld.

The controller is fed the full measured STATE and emits a list of EDIT OPERATIONS with ZERO
outcome priors. The op language makes scaffold modification FLEXIBLE and auditable:
  content   : rewrite_bucket / append / patch / merge         (B1 edit-ops)
  level     : set_level  (T1 strategy | T2 instance-diagnosis | T3 partial-solution)   (A1 dose)
  injection : set_inject (p, mode=fixed|learnable, band=[lb,ub], schedule)             (A2 + A3)
  structure : rebucket / set_global                            (granularity + B2 layered global)
  stability : revert_scaffold(version) / revert_model(ckpt)    (C1 + model rollback)
A `Domain` supplies the item universe and a neutral ENVIRONMENT description; rest is generic.
"""
import json
import os

MODEL = "gpt-5.5"
DEFAULT_KEY_FILE = "/mnt/data1/zha00175/.openai_key"
_FAIL_CHARS = 150000
LEVELS = ("T1", "T2", "T3")


class Domain:
    def __init__(self, name, items, env_desc, instance_word="instance", t3_desc=""):
        self.name = name
        self.items = [str(x).strip().lower() for x in items]
        self.env_desc = env_desc
        self.instance_word = instance_word
        self.t3_desc = t3_desc


SEARCH_DOMAIN = Domain(
    "search_r1",
    ["nq", "triviaqa", "popqa", "hotpotqa", "2wikimultihopqa", "musique", "bamboogle"],
    ("A weak Qwen2.5-3B agent is RL-trained (GiGPO) on Search-R1 multi-hop QA. Each episode it "
     "answers one question in <=4 turns; each turn <think>..</think> then <search>query</search> "
     "(returns 3 wiki passages) OR <answer>..</answer>. Correct = gold covered by answer (cover-EM)."),
    instance_word="question",
    t3_desc="(no ground-truth solution traces exist for Search-R1, so T3 is not recommended here)")

ALF_DOMAIN = Domain(
    "alfworld",
    ["pick_and_place", "pick_two_obj_and_place", "look_at_obj_in_light",
     "pick_heat_then_place_in_recep", "pick_cool_then_place_in_recep", "pick_clean_then_place_in_recep"],
    ("A weak Qwen2.5-1.5B agent is RL-trained (GiGPO) on ALFWorld embodied text tasks. Each episode "
     "completes one household task in <=50 steps using verbs: go to, open, take X from Y, put X in/on Y, "
     "use, heat X with microwave, cool X with fridge, clean X with sinkbasin. Correct = goal satisfied."),
    instance_word="game",
    t3_desc="ALFWorld HAS a ground-truth expert plan per game (oracle PDDL subgoals); a T3 hint reveals the "
            "first K subgoals of that real plan (mechanical truncation, not fabricated).")


def load_scaffold(path):
    try:
        return json.load(open(path))
    except Exception:
        return None


def empty_scaffold():
    return {"mode": "full", "global_skill": "", "default_p": 0.5,
            "buckets": {}, "instances": {}, "version": 0}


def _norm(x):
    return str(x).strip().lower()


def apply_ops(scaffold, ops):
    """Apply edit ops -> NEW scaffold. Assumes validate() already passed."""
    s = json.loads(json.dumps(scaffold or empty_scaffold()))
    s.setdefault("buckets", {}); s.setdefault("instances", {}); s.setdefault("global_skill", "")
    for op in ops:
        t = op.get("op")
        if t == "rebucket":
            new = {}
            for name, b in op["buckets"].items():
                prev = s["buckets"].get(name, {})
                new[name] = {"members": [_norm(m) for m in b["members"]],
                             "skill": b.get("skill", prev.get("skill", "")),
                             "level": b.get("level", prev.get("level", "T1")),
                             "inject": b.get("inject", prev.get("inject", {"p": s.get("default_p", 0.5), "mode": "fixed"}))}
            s["buckets"] = new
        elif t == "rewrite_bucket":
            b = s["buckets"].setdefault(op["bucket"], {"members": [], "level": "T1", "inject": {"p": 0.5, "mode": "fixed"}})
            b["skill"] = op.get("skill", "")
            if "level" in op: b["level"] = op["level"]
        elif t == "append":
            b = s["buckets"].get(op["bucket"])
            if b is not None:
                b["skill"] = (b.get("skill", "").rstrip() + " " + op.get("text", "")).strip()
        elif t == "patch":
            b = s["buckets"].get(op["bucket"])
            if b is not None and op.get("find"):
                b["skill"] = b.get("skill", "").replace(op["find"], op.get("replace", ""), 1)
        elif t == "merge":
            src, dst = op.get("from"), op.get("into")
            if src in s["buckets"] and dst in s["buckets"]:
                s["buckets"][dst]["members"] = list(dict.fromkeys(
                    s["buckets"][dst]["members"] + s["buckets"][src]["members"]))
                if op.get("skill"): s["buckets"][dst]["skill"] = op["skill"]
                del s["buckets"][src]
        elif t == "set_level":
            b = s["buckets"].get(op["bucket"])
            if b is not None and op.get("level") in LEVELS: b["level"] = op["level"]
        elif t == "set_inject":
            b = s["buckets"].get(op["bucket"])
            if b is not None:
                inj = {k: op[k] for k in ("p", "mode", "band", "schedule") if k in op}
                b["inject"] = {**b.get("inject", {}), **inj}
        elif t == "set_global":
            s["global_skill"] = op.get("text", "")
        elif t == "set_instance":
            s["instances"][str(op["key"])] = {"level": op.get("level", "T2"),
                                              "hint": op.get("hint", ""), "p": op.get("p", s.get("default_p", 0.5))}
        elif t == "set_default_p":
            s["default_p"] = float(op["p"])
    s["version"] = int(s.get("version", 0)) + 1
    return s


def validate(action, domain, scaffold=None):
    if not isinstance(action, dict):
        return False, "action is not an object"
    ops = action.get("ops")
    if not isinstance(ops, list) or not ops:
        return False, "ops missing or empty"
    items = set(domain.items)
    for op in ops:
        if not isinstance(op, dict) or "op" not in op:
            return False, f"malformed op: {op}"
        t = op["op"]
        if t in ("rewrite_bucket", "append", "patch", "set_level", "set_inject") and "bucket" not in op:
            return False, f"{t} needs a bucket"
        if t == "set_level" and op.get("level") not in LEVELS:
            return False, f"set_level bad level {op.get('level')}"
        if t == "set_inject" and "p" in op and not (0.0 <= float(op["p"]) <= 1.0):
            return False, f"inject p out of [0,1]: {op.get('p')}"
        if t == "set_inject" and op.get("mode") not in (None, "fixed", "learnable"):
            return False, f"inject mode invalid: {op.get('mode')}"
        if t == "rebucket":
            bs = op.get("buckets")
            if not isinstance(bs, dict) or not bs:
                return False, "rebucket needs buckets"
            seen = set()
            for name, b in bs.items():
                for m in (b.get("members") or []):
                    k = _norm(m)
                    if k not in items:
                        return False, f"unknown item '{m}' in bucket {name}"
                    if k in seen:
                        return False, f"item '{k}' in >1 bucket"
                    seen.add(k)
            if seen != items:
                return False, f"rebucket must partition ALL items; missing {sorted(items - seen)}"
        if t == "revert_model" and not isinstance(op.get("checkpoint"), str):
            return False, "revert_model needs a checkpoint id string"
    return True, "ok"


def assemble_state(cap, current, history, train_curve, step, domain, available_checkpoints=None,
                   available_versions=None):
    base = {"step": step, "domain": domain.name,
            "objective": f"maximize standalone (no-scaffold) success, per {domain.instance_word}-type",
            "items_to_partition": domain.items, "current_scaffold": current,
            "decision_history": history or [],
            "available_checkpoints": available_checkpoints or [],
            "available_scaffold_versions": available_versions or []}
    if cap is None:
        base["note"] = "COLD START: no evaluation yet. Emit an initial scaffold blind."
        return base
    base.update({
        "standalone": {"overall": cap.get("acc"), "per_item": cap.get("per_source", cap.get("per_task", {}))},
        "counterfactual_with_vs_without_scaffold": cap.get("counterfactual", {}),
        "learnable_band_per_item": cap.get("learnable", {}),
        "failure_modes": cap.get("failure_modes", {}),
        "failure_trajectories": cap.get("failures", []),
        "successes_for_contrast": cap.get("successes_for_contrast", []),
        "train_curve": train_curve or {}})
    return base


_HEAD = """You are the CONTROLLER of an automated RL-training experiment (domain: {domain}). You shape a text "scaffold" spliced into the agent's prompt DURING TRAINING ONLY; the agent is EVALUATED WITHOUT it. The ONLY objective is standalone (no-scaffold) per-item success. You get NO guidance about which values work — infer everything from the measured STATE and your decision->outcome history.

ENVIRONMENT (facts, NOT advice):
{env}
The item-types to partition are: {items}.
{t3}

WHAT YOU CONTROL — emit a list of EDIT OPERATIONS (surgical, not full rewrites):
- rewrite_bucket {{bucket, skill, level?}} / append {{bucket, text}} / patch {{bucket, find, replace}} / merge {{from, into, skill?}}
- set_level {{bucket, level}}  level T1 (task strategy) | T2 (per-instance failure diagnosis, no solution) | T3 (per-instance PARTIAL SOLUTION)
- set_inject {{bucket, p, mode, band, schedule}}  mode=fixed (constant p) OR learnable (inject only when that item success is inside band=[lb,ub]); schedule=[[step,p],...] optional withdrawal
- rebucket {{buckets:{{name:{{members:[...]}}}}}}  (repartition; every item in exactly one bucket)
- set_global {{text}}  (global skill layered under all buckets)
- set_instance {{key, level, hint, p}}  (T2/T3 per-instance)
- revert_scaffold {{version}}  (roll scaffold back to a version in available_scaffold_versions)
- revert_model {{checkpoint}}  (roll MODEL back to a checkpoint in available_checkpoints; do not invent ids)

STATE:
{state}

Return ONLY JSON: {{"diagnosis":"<reasoning, logged not executed>", "ops":[ ... ]}}"""


def render_prompt(state, domain):
    body = json.dumps(state, ensure_ascii=False)
    budget = _FAIL_CHARS + 20000
    if len(body) > budget and state.get("failure_trajectories"):
        s = dict(state); fails = list(state["failure_trajectories"])
        while fails and len(json.dumps({**s, "failure_trajectories": fails}, ensure_ascii=False)) > budget:
            fails = fails[:-1]
        s["failure_trajectories"] = fails
        body = json.dumps(s, ensure_ascii=False)
    return _HEAD.format(domain=domain.name, env=domain.env_desc, items=domain.items,
                        t3=domain.t3_desc, state=body)


def call_gpt(prompt, key_file=DEFAULT_KEY_FILE, max_tokens=6000):
    from openai import OpenAI
    cli = OpenAI(api_key=open(key_file).read().strip(), timeout=600, max_retries=2)
    r = cli.chat.completions.create(model=MODEL, messages=[{"role": "user", "content": prompt}],
                                    max_completion_tokens=max_tokens, response_format={"type": "json_object"})
    return json.loads(r.choices[0].message.content)


def tick(cap, current, history, train_curve, step, scaffold_path, domain,
         key_file=DEFAULT_KEY_FILE, available_checkpoints=None, available_versions=None):
    state = assemble_state(cap, current, history, train_curve, step, domain,
                           available_checkpoints, available_versions)
    prompt = render_prompt(state, domain)
    try:
        action = call_gpt(prompt, key_file)
    except Exception as e:
        return {"ok": False, "action": None, "reason": f"gpt call failed: {str(e)[:200]}"}
    ok, reason = validate(action, domain, current)
    if not ok:
        return {"ok": False, "action": action, "reason": f"invalid: {reason}"}
    new_scaffold = apply_ops(current or empty_scaffold(), action["ops"])
    tmp = scaffold_path + ".tmp"
    json.dump(new_scaffold, open(tmp, "w"), indent=2, ensure_ascii=False)
    os.replace(tmp, scaffold_path)
    revert = next((o for o in action["ops"] if o.get("op") == "revert_model"), None)
    return {"ok": True, "action": action, "reason": "applied", "scaffold": new_scaffold,
            "revert_model": revert.get("checkpoint") if revert else None}
