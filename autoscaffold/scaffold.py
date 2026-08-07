"""Scaffold state: addressable text items plus per-category injection probabilities.

A plain dict, persisted as JSON, hot-reloaded by training each cycle. The Teacher
edits it only through the validated operations here. Every function returns new
objects; nothing mutates its argument.

Locked constants (DESIGN.md): P_MAX 0.5, per-cycle p change +-0.2 (clamped, not
rejected), 3 add/update edits per cycle, 8 items per scope, dedup on normalized text.
"""
from __future__ import annotations

import copy
import json
import os
import re
import tempfile

CATEGORIES = (
    "pick_and_place",
    "pick_two_obj_and_place",
    "look_at_obj_in_light",
    "pick_heat_then_place_in_recep",
    "pick_cool_then_place_in_recep",
    "pick_clean_then_place_in_recep",
)
SCOPES = ("general",) + CATEGORIES
KINDS = ("skill", "example")

P_MAX = float(os.environ.get("AUTOSCAFFOLD_P_MAX", "0.5"))
P_MAX_DELTA = float(os.environ.get("AUTOSCAFFOLD_MAX_DP", "0.2"))
BUDGET_CHANGES = int(os.environ.get("AUTOSCAFFOLD_BUDGET_CHANGES", "3"))
MAX_ITEMS_PER_SCOPE = int(os.environ.get("AUTOSCAFFOLD_MAX_ITEMS", "8"))
MAX_ITEM_CHARS = int(os.environ.get("AUTOSCAFFOLD_MAX_ITEM_CHARS", "500"))


def empty_scaffold():
    return {
        "version": 0,
        "items": {s: [] for s in SCOPES},
        "p_task": {c: 0.0 for c in CATEGORIES},
        "next_item_n": 1,
    }


def category_of_gamefile(gamefile):
    """Task category from a gamefile path, or None. `pick_and_place_simple` must be
    tested after `pick_two_obj_and_place` never — the two prefixes do not collide, but
    plain `pick_and_place` is a substring of nothing else, so order by specificity."""
    name = os.path.basename(os.path.dirname(os.path.dirname(str(gamefile)))) \
        if "/" in str(gamefile) else str(gamefile)
    s = str(gamefile)
    for cat in ("pick_two_obj_and_place", "look_at_obj_in_light",
                "pick_heat_then_place_in_recep", "pick_cool_then_place_in_recep",
                "pick_clean_then_place_in_recep"):
        if cat in s:
            return cat
    if "pick_and_place" in s:
        return "pick_and_place"
    return None


def _norm(text):
    return re.sub(r"\s+", " ", str(text)).strip().lower()


def items_of(scaffold, scope):
    return list((scaffold.get("items") or {}).get(scope) or [])


def _find(scaffold, item_id):
    for scope in SCOPES:
        for it in items_of(scaffold, scope):
            if it.get("id") == item_id:
                return scope, it
    return None, None


def validate_item_ops(ops, scaffold):
    """(ok, reason). Physical validity only; content quality is the A/B's job."""
    if not isinstance(ops, list):
        return False, "item_ops is not a list"
    n_changes = 0
    pending_adds = {}
    pending_deletes = {s: set() for s in SCOPES}
    for op in ops:
        if not isinstance(op, dict):
            return False, "op is not an object"
        kind_of_op = op.get("op")
        if kind_of_op == "add":
            scope, kind, text = op.get("scope"), op.get("kind"), op.get("text")
            if scope not in SCOPES:
                return False, f"unknown scope {scope!r}"
            if kind not in KINDS:
                return False, f"unknown kind {kind!r} (allowed: {', '.join(KINDS)})"
            if not isinstance(text, str) or not text.strip():
                return False, "add without text"
            if len(text) > MAX_ITEM_CHARS:
                return False, f"item text over {MAX_ITEM_CHARS} chars"
            existing = {_norm(it["text"]) for it in items_of(scaffold, scope)}
            if _norm(text) in existing or _norm(text) in pending_adds.get(scope, set()):
                return False, f"duplicate text in scope {scope!r}"
            pending_adds.setdefault(scope, set()).add(_norm(text))
            n_changes += 1
        elif kind_of_op == "update":
            scope, it = _find(scaffold, op.get("id"))
            if it is None:
                return False, f"update names unknown id {op.get('id')!r}"
            text = op.get("text")
            if not isinstance(text, str) or not text.strip():
                return False, "update without text"
            if len(text) > MAX_ITEM_CHARS:
                return False, f"item text over {MAX_ITEM_CHARS} chars"
            others = {_norm(x["text"]) for x in items_of(scaffold, scope)
                      if x["id"] != op.get("id")}
            if _norm(text) in others:
                return False, f"update duplicates another item in scope {scope!r}"
            n_changes += 1
        elif kind_of_op == "delete":
            scope, it = _find(scaffold, op.get("id"))
            if it is None:
                return False, f"delete names unknown id {op.get('id')!r}"
            if op.get("id") in pending_deletes[scope]:
                return False, f"duplicate delete of {op.get('id')!r}"
            pending_deletes[scope].add(op.get("id"))
        else:
            return False, f"unknown op {kind_of_op!r}"
    if n_changes > BUDGET_CHANGES:
        return False, f"{n_changes} adds+updates exceed the budget of {BUDGET_CHANGES} per cycle"
    for scope, texts in pending_adds.items():
        room = MAX_ITEMS_PER_SCOPE - (len(items_of(scaffold, scope))
                                      - len(pending_deletes[scope]))
        if len(texts) > room:
            return False, (f"scope {scope!r} holds {len(items_of(scaffold, scope))} of "
                           f"{MAX_ITEMS_PER_SCOPE} items; delete before adding")
    return True, "ok"


def apply_item_ops(scaffold, ops):
    """(new_scaffold, notes). Deletes first, then updates, then adds, so delete-to-make-
    room works within one action regardless of the order the Teacher listed them."""
    ok, reason = validate_item_ops(ops, scaffold)
    if not ok:
        raise ValueError(reason)
    nxt = copy.deepcopy(scaffold)
    notes = []
    for op in ops:
        if op.get("op") == "delete":
            scope, it = _find(nxt, op["id"])
            if it is None:
                notes.append(f"delete {op['id']}: already gone; skipped")
                continue
            nxt["items"][scope] = [x for x in nxt["items"][scope] if x["id"] != op["id"]]
    for op in ops:
        if op.get("op") == "update":
            scope, it = _find(nxt, op["id"])
            if it is None:
                notes.append(f"update {op['id']}: deleted earlier in the same action; skipped")
                continue
            nxt["items"][scope] = [dict(x, text=op["text"].strip()) if x["id"] == op["id"]
                                   else x for x in nxt["items"][scope]]
    for op in ops:
        if op.get("op") == "add":
            item_id = f"i{nxt['next_item_n']}"
            nxt["next_item_n"] += 1
            nxt["items"][op["scope"]].append(
                {"id": item_id, "kind": op["kind"], "text": op["text"].strip()})
    nxt["version"] = int(nxt.get("version", 0)) + 1
    return nxt, notes


def apply_p_ops(scaffold, p_ops):
    """(new_scaffold, notes). Clamped to [0, P_MAX] and to +-P_MAX_DELTA per cycle."""
    nxt = copy.deepcopy(scaffold)
    notes = []
    for op in p_ops or []:
        cat = op.get("task")
        if cat not in CATEGORIES:
            notes.append(f"p op names unknown category {cat!r}; skipped")
            continue
        try:
            target = float(op.get("p"))
        except (TypeError, ValueError):
            notes.append(f"p op for {cat} has non-numeric p; skipped")
            continue
        old = float(nxt["p_task"].get(cat, 0.0))
        clamped = max(0.0, min(P_MAX, target))
        step_limited = max(old - P_MAX_DELTA, min(old + P_MAX_DELTA, clamped))
        if abs(step_limited - target) > 1e-9:
            notes.append(f"p[{cat}] {target} clamped to {round(step_limited, 4)}")
        nxt["p_task"][cat] = round(step_limited, 4)
    return nxt, notes


def render(scaffold, category):
    """The text a training prompt of `category` receives: general items first, then the
    category's own, as bullet lines. Empty string when nothing applies."""
    parts = [it["text"] for it in items_of(scaffold, "general")]
    if category in CATEGORIES:
        parts += [it["text"] for it in items_of(scaffold, category)]
    return "\n".join(f"- {t}" for t in parts)


def injects_nothing(scaffold):
    """True when no rollout can receive text: no items at all, or p zero everywhere."""
    has_text = any(items_of(scaffold, s) for s in SCOPES)
    has_p = any(float(v or 0) > 0 for v in (scaffold.get("p_task") or {}).values())
    return not (has_text and has_p)


def reached_categories(scaffold):
    """Categories the CURRENT scaffold's text can reach: all of them when general
    items exist, plus每 category with items of its own. The A/B measures over the
    union of this and the proposal's touched set, so a revert-to-bare only ever
    clears text the measurement actually covered."""
    reached = set()
    if items_of(scaffold, "general"):
        return list(CATEGORIES)
    for cat in CATEGORIES:
        if items_of(scaffold, cat):
            reached.add(cat)
    return sorted(reached)


def clear_items(scaffold):
    """(new_scaffold, cleared): every item removed, p_task kept (inert without text;
    any future item must pass the A/B before that p applies to anything)."""
    import copy as _copy
    nxt = _copy.deepcopy(scaffold)
    cleared = [dict(it, scope=scope) for scope in SCOPES for it in items_of(nxt, scope)]
    nxt["items"] = {s: [] for s in SCOPES}
    nxt["version"] = int(nxt.get("version", 0)) + 1
    return nxt, cleared


def touched_categories(item_ops, scaffold):
    """Categories whose prompts an action can change; 'general' touches all of them."""
    touched = set()
    for op in item_ops or []:
        # For update/delete the item's REAL scope decides what the edit can change;
        # a spurious 'scope' key on the op must not steer the A/B at the wrong games.
        if op.get("id"):
            scope, _ = _find(scaffold, op["id"])
        else:
            scope = op.get("scope")
        if scope == "general":
            return list(CATEGORIES)
        if scope in CATEGORIES:
            touched.add(scope)
    return sorted(touched)


def is_noop(action):
    return not (action.get("item_ops") or action.get("p_ops"))


def persist(scaffold, path):
    """Atomic write: training hot-reloads this file and must never see a torn one."""
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=d, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(scaffold, f, ensure_ascii=False, indent=1)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def load(path):
    try:
        with open(path) as f:
            d = json.load(f)
    except (OSError, ValueError):
        return None
    return d if isinstance(d, dict) and "items" in d else None
