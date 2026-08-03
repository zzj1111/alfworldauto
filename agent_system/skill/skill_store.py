"""
SkillStore: the adaptive-scaffold skill container for the GiGPO + Helper project.

This object lives in the DRIVER process (the EnvironmentManager builds prompts
driver-side), so the Helper (P3) can update it in place between training steps
with NO Ray broadcast required.

Responsibilities:
  - hold per-task-type skill text + a general skill + per-task injection prob p_task
  - expose a `mode` (full | general_only | none) used by the 3-way eval (P2)
  - render the skill block that gets spliced into a prompt (text only; never the
    GiGPO `anchor` observation)
  - make the group-level inject decision reproducible given a seed

Skills are deliberately plain text + JSON so the Helper can read/rewrite them.
"""
import hashlib
import json
import os
import threading
from typing import Dict, List, Optional


# Scaffold information-disclosure levels (see the T0..T3 hierarchy).
#   T0 none          -> inject nothing
#   T1 distribution  -> per-bucket / task-family skill text  (keyed by data_source->bucket)
#   T2 instance-diag -> per-instance failure diagnosis, NO solution  (keyed by question)
#   T3 instance-soln -> per-instance solution structure (QuestA-style)(keyed by question)
# T2 and T3 share the SAME injection mechanism (instance-keyed hint); they differ only in
# what the teacher writes into the hint.
SCAFFOLD_LEVELS = ("none", "T0", "T1", "T2", "T3")


def question_key(q) -> str:
    """Stable per-instance key: sha1 of the normalized question. The env (training) and the
    teacher (capture) both compute this from the SAME raw question string, so instance hints
    line up. Normalization = lowercase + whitespace-collapsed."""
    s = " ".join(str(q or "").lower().split())
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:16]


# ALFWorld task types, matched as substrings of the gamefile path (same set the
# env_manager already uses for per-task success_rate logging).
ALFWORLD_TASK_TYPES: List[str] = [
    "pick_and_place",
    "pick_two_obj_and_place",
    "look_at_obj_in_light",
    "pick_heat_then_place_in_recep",
    "pick_cool_then_place_in_recep",
    "pick_clean_then_place_in_recep",
]

# Marker present in both ALFWORLD_TEMPLATE and ALFWORLD_TEMPLATE_NO_HIS; we splice
# the skill block immediately before it so the hint sits right above the action ask.
_ACTION_MARKER = "Now it's your turn to take an action."

_SKILL_BLOCK_HEADER = (
    "Here are skill hints for this task. Treat them as guidance, not gospel — "
    "you must still reason about the current observation yourself:"
)


def detect_task_type(gamefile: Optional[str]) -> str:
    """Map an ALFWorld gamefile path to its task type, or 'unknown'."""
    if not gamefile or not isinstance(gamefile, str):
        return "unknown"
    for t in ALFWORLD_TASK_TYPES:
        if t in gamefile:
            return t
    return "unknown"

SEARCH_TASK_TYPES: List[str] = ["single_hop", "multi_hop"]
_SEARCH_HOP = {
    "nq": "single_hop", "triviaqa": "single_hop", "popqa": "single_hop",
    "hotpotqa": "multi_hop", "2wikimultihopqa": "multi_hop",
    "musique": "multi_hop", "bamboogle": "multi_hop",
}


def detect_search_task_type(data_source) -> str:
    """Map a search-R1 data_source to single_hop / multi_hop (default multi_hop)."""
    if not data_source:
        return "multi_hop"
    return _SEARCH_HOP.get(str(data_source).strip().lower(), "multi_hop")


def build_buckets(d):
    """From a scaffold dict with a 'buckets' block, derive the injection structures.

    A bucket is {members:[data_source...], skill:str, p:float|None}. The auto-scaffold
    controller owns this partition, so the data_source->key mapping is DATA-DRIVEN
    (not the hard-coded single/multi split). Returns (buckets, skills, p_task,
    member2bucket); all empty when the dict has no 'buckets' block (legacy path).
    """
    raw = d.get("buckets")
    if not isinstance(raw, dict) or not raw:
        return {}, {}, {}, {}
    buckets, skills, p_task, m2b = {}, {}, {}, {}
    for name, b in raw.items():
        if not isinstance(b, dict):
            continue
        members = [str(x).strip().lower() for x in (b.get("members") or [])]
        skill = b.get("skill", "") or ""
        p = b.get("p")
        buckets[name] = {"members": members, "skill": skill, "p": p}
        skills[name] = skill                       # render() keys on the bucket name
        if p is not None:
            p_task[name] = float(p)                # get_p() keys on the bucket name
        for m in members:
            m2b[m] = name
    return buckets, skills, p_task, m2b


def build_instances(d):
    """From a scaffold dict with an 'instances' block, derive instance-keyed injection.

    instances: {question_key: {hint:str, p:float|None}}. Used by T2/T3. render()/get_p()
    then key on the question_key. Returns (instances, skills, p_task); empty when absent.
    """
    raw = d.get("instances")
    if not isinstance(raw, dict) or not raw:
        return {}, {}, {}
    instances, skills, p_task = {}, {}, {}
    for qid, e in raw.items():
        if not isinstance(e, dict):
            continue
        hint = e.get("hint", "") or ""
        p = e.get("p")
        instances[qid] = {"hint": hint, "p": p}
        skills[qid] = hint
        if p is not None:
            p_task[qid] = float(p)
    return instances, skills, p_task


def splice_skill(prompt: str, skill_block: str) -> str:
    """Insert `skill_block` into `prompt` just before the action marker.

    Pure function; never touches the raw observation. Returns `prompt` unchanged
    if `skill_block` is empty.
    """
    if not skill_block:
        return prompt
    block = "\n" + skill_block.strip() + "\n"
    if _ACTION_MARKER in prompt:
        return prompt.replace(_ACTION_MARKER, block + "\n" + _ACTION_MARKER, 1)
    return prompt.rstrip() + "\n" + block


class SkillStore:
    """Container for skills + injection probabilities. Thread-safe updates."""

    def __init__(
        self,
        skills: Optional[Dict[str, str]] = None,
        general_skill: str = "",
        p_task: Optional[Dict[str, float]] = None,
        mode: str = "full",
        default_p: float = 1.0,
    ):
        if mode not in ("full", "general_only", "none"):
            raise ValueError(f"invalid skill mode: {mode}")
        self._lock = threading.Lock()
        self.skills: Dict[str, str] = dict(skills or {})
        self.general_skill: str = general_skill or ""
        self.p_task: Dict[str, float] = dict(p_task or {})
        self.mode: str = mode
        self.default_p: float = float(default_p)
        self.version: int = 0
        self._src_path = None
        self._src_mtime = 0.0
        # Controller-defined buckets (auto-scaffold). Empty -> legacy single/multi path.
        self.buckets: Dict[str, dict] = {}
        self._member2bucket: Dict[str, str] = {}
        # Scaffold level + instance-keyed hints (T2/T3). level defaults are derived in _ingest.
        self.level: str = "T1"
        self.instances: Dict[str, dict] = {}

    # ----- construction -----
    @classmethod
    def from_json(cls, path: str, mode: Optional[str] = None) -> "SkillStore":
        with open(path) as f:
            d = json.load(f)
        store = cls(
            skills=d.get("skills", {}),
            general_skill=d.get("general_skill", ""),
            p_task=d.get("p_task", {}),
            mode=mode or d.get("mode", "full"),
            default_p=float(d.get("default_p", 1.0)),
        )
        store._src_path = path
        try:
            store._src_mtime = os.path.getmtime(path)
        except Exception:
            store._src_mtime = 0.0
        store._ingest(d)
        return store

    def _ingest(self, d):
        """Apply the scaffold LEVEL + its keyed content: buckets (T1, keyed by data_source)
        or instances (T2/T3, keyed by question). Explicit `level` wins; else derived from
        content. Legacy single/multi scaffolds stay T1 with skills untouched."""
        buckets, b_skills, b_ptask, m2b = build_buckets(d)
        instances, i_skills, i_ptask = build_instances(d)
        lvl = d.get("level")
        if lvl not in SCAFFOLD_LEVELS:
            lvl = "T2" if instances else ("T1" if (buckets or d.get("skills")) else "none")
        self.level = lvl
        if instances:
            self.instances, self.skills, self.p_task = instances, i_skills, i_ptask
            self.buckets, self._member2bucket = {}, {}
        elif buckets:
            self.buckets, self.skills, self.p_task, self._member2bucket = buckets, b_skills, b_ptask, m2b
            self.instances = {}

    @classmethod
    def from_env(cls) -> "SkillStore":
        """Build from env vars. No GIGPO_SKILL_PATH -> disabled (mode 'none')."""
        path = os.environ.get("GIGPO_SKILL_PATH")
        if not path:
            return cls(mode="none")
        mode = os.environ.get("GIGPO_SKILL_MODE")  # optional override of file mode
        store = cls.from_json(path, mode=mode)
        p = os.environ.get("GIGPO_SKILL_P")  # override injection prob for ALL tasks (withdrawal)
        if p is not None:
            pv = float(p)
            store.default_p = pv
            store.p_task = {k: pv for k in ALFWORLD_TASK_TYPES}
        return store

    # ----- queries -----
    def _maybe_reload(self):
        """Hot-reload skills if the source JSON changed on disk (Teacher updates it live)."""
        if not self._src_path:
            return
        try:
            m = os.path.getmtime(self._src_path)
            if m > self._src_mtime:
                with open(self._src_path) as f:
                    d = json.load(f)
                self.skills = dict(d.get("skills", {}))
                self.general_skill = d.get("general_skill", "") or ""
                # withdrawal curriculum: the supervisor writes default_p per step
                if "default_p" in d:
                    self.default_p = float(d["default_p"])
                if "p_task" in d:
                    self.p_task = dict(d["p_task"])
                # auto-scaffold: level + buckets(T1) / instances(T2,T3) keyed content
                self._ingest(d)
                self._src_mtime = m
                self.version += 1
        except Exception:
            pass

    def get_p(self, task_type: str) -> float:
        with self._lock:
            self._maybe_reload()
            return float(self.p_task.get(task_type, self.default_p))

    def episode_key(self, data_source, question) -> str:
        """Resolve one episode's injection key by the current LEVEL. Returns '' for
        no-injection (T0, or a T2/T3 instance with no hint). T1 -> data_source bucket;
        T2/T3 -> question key (only if that instance was annotated)."""
        with self._lock:
            self._maybe_reload()
            lvl = self.level
            insts = self.instances
        if lvl in ("none", "T0"):
            return ""
        if lvl in ("T2", "T3"):
            qid = question_key(question)
            return qid if qid in insts else ""
        return self.search_key(data_source)   # T1 (and legacy)

    def search_key(self, data_source) -> str:
        """Resolve a search-R1 data_source to its scaffold key. Uses the controller's
        buckets when defined (member->bucket, data-driven), else the legacy single/
        multi-hop map. Unmapped sources fall back to a deterministic existing bucket."""
        with self._lock:
            self._maybe_reload()
            if self._member2bucket:
                b = self._member2bucket.get(str(data_source or "").strip().lower())
                if b:
                    return b
                if self.buckets:
                    return sorted(self.buckets)[0]
        return detect_search_task_type(data_source)

    def render(self, task_type: str) -> str:
        """Skill block text for this task type given the current mode (or '')."""
        with self._lock:
            self._maybe_reload()
            if self.mode == "none":
                return ""
            parts: List[str] = []
            if self.mode == "full":
                s = (self.skills.get(task_type, "") or "").strip()
                if s:
                    parts.append(s)
            g = self.general_skill.strip()
            if g:
                parts.append(g)
            if not parts:
                return ""
            body = "\n".join(f"- {p}" for p in parts)
            return f"{_SKILL_BLOCK_HEADER}\n{body}"

    # ----- mutation (used by the Helper in P3) -----
    def update(
        self,
        skills: Optional[Dict[str, str]] = None,
        general_skill: Optional[str] = None,
        p_task: Optional[Dict[str, float]] = None,
        mode: Optional[str] = None,
    ) -> int:
        with self._lock:
            if skills is not None:
                self.skills.update(skills)
            if general_skill is not None:
                self.general_skill = general_skill
            if p_task is not None:
                self.p_task.update({k: float(v) for k, v in p_task.items()})
            if mode is not None:
                if mode not in ("full", "general_only", "none"):
                    raise ValueError(f"invalid skill mode: {mode}")
                self.mode = mode
            self.version += 1
            return self.version

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "skills": dict(self.skills),
                "general_skill": self.general_skill,
                "p_task": dict(self.p_task),
                "mode": self.mode,
                "default_p": self.default_p,
                "version": self.version,
            }

    def save_json(self, path: str) -> None:
        snap = self.snapshot()
        snap.pop("version", None)
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(snap, f, indent=2)
        os.replace(tmp, path)
