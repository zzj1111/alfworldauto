"""The Teacher: propose scaffold edits from the observation packet.

The GPT call is injected (call_fn) so the loop and tests never touch the network.
Malformed output degrades to a no-op — the run continues on the current scaffold.
Content quality is never judged here; that is the A/B gate's job.

An unreachable Teacher (dead key, exhausted quota, no network) is marked with
UNREACHABLE_NOTE so the loop can count consecutive outage cycles. Without the mark,
"it declined" and "it was never asked" leave the same trace: an empty scaffold and a
run that finishes at full length — a plain-RL control that reads as a result.
"""
from __future__ import annotations

import json
import os

from . import scaffold as S

MODEL = os.environ.get("AUTOSCAFFOLD_TEACHER_MODEL", "gpt-5.5")
NOOP = {"diagnosis": "", "item_ops": [], "p_ops": []}
UNREACHABLE_NOTE = "teacher unreachable"

# Errors meaning no call can succeed this cycle, as opposed to a bad answer. Matched on
# the message: the openai SDK raises the same class for "retry later" and "no credits".
_UNREACHABLE = ("insufficient_quota", "credit_balance_exhausted", "no credits",
                "invalid_api_key", "incorrect api key", "authentication", "error code: 401",
                "connection error", "apiconnectionerror")


def teacher_unreachable(err):
    return any(s in str(err).lower() for s in _UNREACHABLE)


def _read_key():
    key = os.environ.get("OPENAI_API_KEY")
    if key:
        return key
    path = os.environ.get("AUTOSCAFFOLD_OPENAI_KEY_FILE")
    if not path:
        raise RuntimeError("no_key: neither OPENAI_API_KEY nor AUTOSCAFFOLD_OPENAI_KEY_FILE set")
    txt = open(path).read()
    for line in txt.splitlines():
        if line.strip().startswith("OPENAI_API_KEY="):
            return line.strip().split("=", 1)[1].strip()
    return txt.strip()


def openai_call(system, user, model=None, max_tokens=6000):
    """Real GPT call; imported lazily so tests never need the openai package."""
    from openai import OpenAI
    cli = OpenAI(api_key=_read_key(), timeout=300, max_retries=2)
    r = cli.chat.completions.create(
        model=model or MODEL,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}],
        response_format={"type": "json_object"}, max_completion_tokens=max_tokens)
    return json.loads(r.choices[0].message.content)


def normalize(raw, scaffold):
    """Coerce a raw Teacher dict into a validated action, or a no-op on any problem.
    Physical validation only — item ops are checked against the live scaffold because
    an update/delete names an id, and whether that id exists is not a property of the
    op alone."""
    if not isinstance(raw, dict):
        return dict(NOOP), "non-dict output -> no-op"
    action = {
        "diagnosis": str(raw.get("diagnosis", "") or "")[:2000],
        "item_ops": list(raw.get("item_ops") or []),
        "p_ops": list(raw.get("p_ops") or []),
    }
    ok, reason = S.validate_item_ops(action["item_ops"], scaffold)
    if not ok:
        return dict(NOOP), f"invalid item_ops ({reason}) -> no-op"
    for op in action["p_ops"]:
        if not isinstance(op, dict) or op.get("task") not in S.CATEGORIES:
            return dict(NOOP), f"invalid p op {op!r} -> no-op"
        try:
            float(op.get("p"))
        except (TypeError, ValueError):
            return dict(NOOP), f"non-numeric p in {op!r} -> no-op"
    return action, "ok"


def propose(system, user, scaffold, call_fn=openai_call):
    """(action, note). Any exception in the call becomes a no-op, never a crash;
    unreachable failures carry UNREACHABLE_NOTE for the loop to count."""
    try:
        raw = call_fn(system, user)
    except Exception as e:
        if teacher_unreachable(e):
            return dict(NOOP), f"{UNREACHABLE_NOTE} ({str(e)[:150]}) -> no-op"
        return dict(NOOP), f"teacher call failed ({str(e)[:150]}) -> no-op"
    return normalize(raw, scaffold)
