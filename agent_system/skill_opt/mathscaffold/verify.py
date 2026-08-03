"""Answer extraction + verification for MATH-style problems.

The reward is binary and comes from the FINAL ANSWER only: extract what the model put in
`\\boxed{}` and check symbolic equivalence against the ground truth. This mirrors the
protocol in OC-GRPO (arXiv 2607.19313) so our numbers stay comparable to theirs.

Two layers, in order:
  1. `math_verify` (HF) — the standard LaTeX-aware parser/comparator.
  2. a sympy + normalized-string fallback, used when math_verify is unavailable or throws.

Everything here is pure and unit-tested; nothing loads a model or touches the network.
"""
from __future__ import annotations

import re

_BOXED = re.compile(r"\\boxed\s*{")


def extract_boxed(text):
    """Return the content of the LAST `\\boxed{...}` in `text`, or None.

    Brace-matched rather than regex-greedy: answers like `\\boxed{\\frac{1}{2}}` contain
    nested braces, and a naive `\\boxed{(.*?)}` truncates them to `\\frac{1`.
    """
    if not text:
        return None
    starts = [m.end() for m in _BOXED.finditer(text)]
    if not starts:
        return None
    i = starts[-1]                      # last boxed = the final answer
    depth, out = 1, []
    while i < len(text):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return "".join(out).strip()
        out.append(ch)
        i += 1
    return None                          # unbalanced braces -> treat as no answer


_STRIP = [
    (re.compile(r"\\left|\\right"), ""),
    (re.compile(r"\\!|\\,|\\;|\\:|\\ "), ""),
    (re.compile(r"\\text\s*{([^}]*)}"), r"\1"),
    (re.compile(r"\\mbox\s*{([^}]*)}"), r"\1"),
    (re.compile(r"\$"), ""),
    (re.compile(r"\s+"), ""),
    (re.compile(r"^\\\$|\\%$|^%|\\%"), ""),
    (re.compile(r"\.$"), ""),
]


def normalize(ans):
    """Cheap syntactic normalization so trivially-equal answers compare equal."""
    if ans is None:
        return None
    s = str(ans).strip()
    for pat, rep in _STRIP:
        s = pat.sub(rep, s)
    s = s.replace("dfrac", "frac").replace("tfrac", "frac")
    if s.endswith("\\"):
        s = s[:-1]
    # 0.50 -> 0.5, 3.0 -> 3
    if re.fullmatch(r"-?\d+\.\d+", s):
        s = s.rstrip("0").rstrip(".")
    return s


def _sympy_equal(a, b):
    try:
        from sympy import simplify
        from sympy.parsing.latex import parse_latex
        d = simplify(parse_latex(a) - parse_latex(b))
        return bool(d == 0)
    except Exception:
        return False


def answers_equal(pred, gold):
    """True iff `pred` and `gold` denote the same answer. Never raises."""
    if pred is None or gold is None:
        return False
    if normalize(pred) == normalize(gold):
        return True
    try:                                   # preferred path: LaTeX-aware comparator
        from math_verify import parse, verify
        p, g = parse(f"${pred}$"), parse(f"${gold}$")
        if p and g and verify(g, p):
            return True
    except Exception:
        pass
    return _sympy_equal(pred, gold)


def score(completion, gold):
    """Binary verifier reward for one completion. 1.0 iff the boxed answer matches."""
    return 1.0 if answers_equal(extract_boxed(completion), gold) else 0.0
