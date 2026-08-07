"""The Teacher's prompts and observation packet.

One rule governs this file: every mechanism the prompt describes must be a mechanism
the code has, and every field the packet carries must be described. The previous
implementation was corrected three times for drift here (a smoothing that did not
exist, another arm's assignment mechanism, retired fields still described);
tests/test_alignment.py holds the line structurally.
"""
from __future__ import annotations

import json
import os

from . import scaffold as S
from . import signals as G

PROMPT_BUDGET = int(os.environ.get("AUTOSCAFFOLD_PROMPT_BUDGET", "160000"))
HISTORY_KEPT = 40   # covers a whole 30-cycle run; entries are small


def render_system_prompt():
    return f"""You are the Teacher for an RL run training a small language model to solve
ALFWorld household tasks (text environment, up to 50 steps per episode, categories:
{', '.join(S.CATEGORIES)}).

WHAT YOU CONTROL. A scaffold: short text items you write, each with an id, in scope
'general' (reaches every category) or one category's scope. During TRAINING, for each
game's group of rollouts, a coin with probability p_task[category] decides once PER
GROUP whether the scaffold text is spliced into that group's prompts. Evaluation and
validation always use the bare prompt — the run's objective is standalone success
WITHOUT your text, and your text can only help by changing what gets learned.

WHY THIS CAN WORK. The trainer is GRPO-family: the advantage is reward minus the
group's own mean, so a group whose rollouts all score the same — all fail OR all
succeed — contributes NO gradient. signals.zero_gradient_groups counts these. all_fail
and all_succeed have opposite remedies: all-fail means the task is out of reach and
text may buy a foothold; all-succeed means it is already solved and text buys nothing.

WHAT THE SIGNALS MEAN (descriptions, not instructions):
- signals.per_task_gap[cat]: success on the TRAINING rollouts of THE LAST CYCLE ONLY,
  split by whether the coin fired ('bare' vs 'injected'). Which side a group lands on
  is random and independent of difficulty, so the sides are comparable. n_bare and
  n_injected are RAW episode counts from that one cycle; nothing is carried over and
  nothing is smoothed — a gap that moves may be real or may be sampling noise at that
  n, so read the n before the gap. A category whose injected side is empty carries
  no_injection_reason naming which of three causes produced it.
- signals.zero_gradient_groups[cat]: {{total, zero_gradient, all_fail, all_succeed}}
  over complete groups of rollout_n episodes.
- signals.contrastive_traces[cat]: up to 3 FAILED trajectories, drawn from all-fail
  groups first and topped up from the lowest-success groups when fewer than 3 exist
  (the longest failure of each; never a successful trajectory), beside the category's
  3 shortest successes. Steps are
  {{a: executed action, o: observation, v: parsed-ok}}. When nothing failed, the entry
  says no_failures_to_show. Traces may be trimmed to fit; a trimmed packet says so —
  absence of traces never means the category had nothing to show.
- signals.failure_patterns[cat]: rule-computed counters over ALL failed rollouts
  (repeated_action: one command >=3 times and >=half the trajectory;
  looped_observation: the same observation >=3 times in a row; invalid_heavy: >=20%
  of actions unparseable). These cover every failure, not just the traces shown.
- valid_seen: the held-out standalone eval — the number this run exists to move.
  Each draw plays a DIFFERENT sample of held-out games; the spread across draws is
  the sampling noise floor of one reading.
- decision_history: your own past proposals with their exact wording, the A/B verdict
  each received, and the held-out number before/after. Text that already lost an A/B
  is recorded here; re-proposing it unchanged wastes a cycle.

HOW YOUR EDITS ARE JUDGED. Text changes are A/B tested before they apply: the frozen
current policy plays HELD-OUT games three ways (no text / current scaffold / your
candidate), over the UNION of the categories your proposal touches and the categories
the current scaffold reaches — and EACH CATEGORY IS JUDGED AND ACTED ON BY ITS OWN
NUMBERS (about 30 episodes per condition at full breadth; small-n verdicts can flip
on a few games — that resolution is a deliberate choice). Per category: your edits
there apply iff the candidate scores strictly above the current scaffold and no lower
than the no-text condition; that category's existing items are CLEARED iff no text
strictly beats both; otherwise it keeps what it has. General-scoped text reaches
every category, so general edits and general items follow the same rule computed on
the aggregate over the union. A p change for a category is discarded when that
category's text lost or was cleared; p for untouched categories applies as a plain
adjustment. The verdict in decision_history itemizes every category's three numbers
and its verdict. Rejected or cleared text is recorded; you see all numbers next
cycle. A p
change submitted together with text that then fails its A/B is discarded with the
text, so a p change you want judged on its own should be submitted alone. p-only
changes skip the A/B.

LIMITS, enforced by clamping or rejection: at most {S.BUDGET_CHANGES} adds+updates per
cycle; at most {S.MAX_ITEMS_PER_SCOPE} items per scope (delete to make room — deletes
are free); items at most {S.MAX_ITEM_CHARS} characters; p_task capped at {S.P_MAX}
(at least half of every category's groups always see the bare prompt) and moves at
most {S.P_MAX_DELTA} per cycle in either direction.

FORMAT RULES for item text: the policy answers with <think>...</think> then
<action>...</action>; never instruct any other output format. Keep items concrete:
name objects, receptacles, and action verbs the environment accepts (go to X, open X,
take A from B, put A in/on B, use X, heat/cool/clean A with B).

Reply with ONE JSON object:
{{"diagnosis": "<what the signals show, briefly>",
  "item_ops": [{{"op": "add", "scope": "<scope>", "kind": "skill|example", "text": "..."}},
               {{"op": "update", "id": "<id>", "text": "..."}},
               {{"op": "delete", "id": "<id>"}}],
  "p_ops": [{{"task": "<category>", "p": <0..{S.P_MAX}>}}]}}
Empty item_ops and empty p_ops means you choose not to intervene this cycle."""


def assemble_observation(scaffold, sig, decision_history, step, cycle, valid_seen=None):
    return {
        "step": step,
        "cycle": cycle,
        "scaffold": {"items": scaffold.get("items"), "p_task": scaffold.get("p_task"),
                     "version": scaffold.get("version")},
        "valid_seen": valid_seen or {},
        "signals": sig,
        "decision_history": _compact_history(decision_history),
    }


def _compact_history(history):
    return list(history or [])[-HISTORY_KEPT:]


def _count_traces(traces):
    return {c: {"failures": len(v.get("zero_gradient_failures") or []),
                "successes": len(v.get("successes_same_category") or [])}
            for c, v in (traces or {}).items()}


def _trim_traces(traces, fits):
    """Drop one trace at a time from whichever category currently holds the most, so
    every category stays represented as long as the budget allows. Successes go before
    failures: the zero-gradient failures are the thing the run is about."""
    cur = {c: dict(v) for c, v in (traces or {}).items()}
    budget = sum(len(v.get("zero_gradient_failures") or [])
                 + len(v.get("successes_same_category") or []) for v in cur.values()) + 1
    while not fits(cur):
        budget -= 1
        if budget < 0:
            return {}          # a picker bug must degrade to empty, never hang the cycle

        sizes = {c: len(v.get("zero_gradient_failures") or [])
                    + len(v.get("successes_same_category") or [])
                 for c, v in cur.items()}
        if not sizes or max(sizes.values()) == 0:
            return {}
        worst = max(sizes, key=lambda c: (sizes[c], c))
        v = dict(cur[worst])
        if v.get("successes_same_category"):
            v["successes_same_category"] = v["successes_same_category"][:-1]
        elif v.get("zero_gradient_failures"):
            v["zero_gradient_failures"] = v["zero_gradient_failures"][:-1]
        cur[worst] = v
    return cur


def render_user_prompt(obs):
    """The packet as JSON, trimmed to PROMPT_BUDGET. Only contrastive_traces shrink;
    failure_patterns are a few hundred characters and cover every failure, so they
    survive any trim."""
    body = json.dumps(obs, ensure_ascii=False, default=str)
    if len(body) <= PROMPT_BUDGET:
        return body
    sig = dict(obs.get("signals") or {})
    traces = sig.get("contrastive_traces") or {}
    before = _count_traces(traces)
    trimmed_obs = dict(obs)

    def fits(cand):
        sig2 = dict(sig, contrastive_traces=cand)
        return len(json.dumps(dict(trimmed_obs, signals=sig2),
                              ensure_ascii=False, default=str)) <= PROMPT_BUDGET

    kept = _trim_traces(traces, fits)
    sig["contrastive_traces"] = kept
    after = _count_traces(kept)
    if after != before:
        sig["contrastive_traces_dropped"] = {
            "note": ("trimmed to a character budget, taken from the largest category "
                     "each time; absence here does NOT mean the category had nothing "
                     "to show"),
            "kept": after, "before": before}
    trimmed_obs["signals"] = sig
    return json.dumps(trimmed_obs, ensure_ascii=False, default=str)
