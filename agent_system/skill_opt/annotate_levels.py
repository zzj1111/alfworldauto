"""Generate a level-{T1,T2,T3} scaffold JSON from a capture of base-model failures.

For the fixed-level dose-response experiment (T0/T1/T2/T3 arms). Given capture_search.py
output (failures with question + gold + trajectory), the GPT-5.5 teacher writes:
  T1  per-BUCKET distribution knowledge  -> {"buckets": {...}}   (keyed by data_source)
  T2  per-INSTANCE failure diagnosis     -> {"instances": {qid: {hint,p}}}  (NO solution)
  T3  per-INSTANCE solution structure    -> {"instances": {qid: {hint,p}}}  (QuestA-style)

T2 is the load-bearing scientific level: its hint must NOT encode the answer. T3 may reveal
the bridge entity / solution path. Instance hints are keyed by question_key so training-time
injection (env) and these annotations line up.
"""
import argparse, json, os, sys
from collections import defaultdict
sys.path.insert(0, "/mnt/data1/zha00175/verl-agent")
from agent_system.skill.skill_store import question_key

MODEL = "gpt-5.5"
DEFAULT_KEY_FILE = "/mnt/data1/zha00175/.openai_key"
HOP = {"nq": "single_hop", "triviaqa": "single_hop", "popqa": "single_hop",
       "hotpotqa": "multi_hop", "2wikimultihopqa": "multi_hop", "musique": "multi_hop", "bamboogle": "multi_hop"}
KNOWN_SOURCES = list(HOP)


def _batches(seq, size):
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


def _gpt_json(prompt, key_file, max_tokens=4000, dry_run=False):
    if dry_run:
        return {"__dry_run__": True}
    from openai import OpenAI
    cli = OpenAI(api_key=open(key_file).read().strip(), timeout=600, max_retries=2)
    r = cli.chat.completions.create(model=MODEL, messages=[{"role": "user", "content": prompt}],
                                    max_completion_tokens=max_tokens, response_format={"type": "json_object"})
    return json.loads(r.choices[0].message.content)


def _fmt_item(i, f, with_gold):
    traj = json.dumps(f.get("trajectory", []), ensure_ascii=False)[:2500]
    g = f" | gold(for YOUR reference only): {f.get('gold')}" if with_gold else ""
    return f"{i}. Q: {f.get('question','')}\n   failed_trajectory: {traj}\n   final_answer_given: {f.get('final_answer','')}{g}"


_T2_HEAD = """You write a short DIAGNOSTIC HINT for each failed attempt by a weak search agent on multi-hop QA (<=4 turns; each turn <search>query</search> or <answer>..</answer>). You see the question, the failed trajectory, and the GOLD answer FOR YOUR REFERENCE ONLY.

Write a hint that says WHY the approach failed and WHAT DIFFERENT DIRECTION to try. HARD RULE: do NOT state, spell, paraphrase, translate, or encode the gold answer or the final solution entity. Point ONLY at the process mistake (searched the whole question, wrong/parcial bridge entity, wrong entity type, didn't decompose, gave up early). ~1 sentence, imperative.

Items:
{items}

Return ONLY JSON mapping each item number to its hint: {{"1": "...", "2": "..."}}."""

_T3_HEAD = """You write a short SOLUTION-STRUCTURE HINT (a partial worked solution, in the spirit of question augmentation) for each failed attempt by a weak search agent on multi-hop QA. You see the question, the failed trajectory, and the GOLD answer.

Reveal the SOLUTION PATH so the problem becomes discoverable: name the key intermediate/bridge entity and outline the steps to reach the answer. You MAY include the bridge entity; reveal enough structure to guide, ~1-2 sentences. This is a training scaffold that will be withdrawn.

Items:
{items}

Return ONLY JSON mapping each item number to its hint: {{"1": "...", "2": "..."}}."""

_T1_HEAD = """You write a per-CATEGORY strategy hint (distribution-level knowledge: positive skills + negative lessons) for a weak search agent on multi-hop QA (<=4 turns). Below are FAILED trajectories from the '{bucket}' category (data_sources: {members}). Study the recurring failure patterns across the whole set and write concrete, imperative guidance for how to solve THIS CLASS of question. Do not reference any single question's specific answer. <=120 words, start with a one-line header.

Failed trajectories:
{items}

Return ONLY JSON: {{"skill": "..."}}."""


def annotate_instances(failures, level, p, key_file, batch, dry_run):
    """T2/T3: per-instance hints keyed by question_key."""
    head = _T2_HEAD if level == "T2" else _T3_HEAD
    with_gold = True  # both see gold; T2 is forbidden from encoding it, T3 may use it
    instances = {}
    for chunk in _batches(failures, batch):
        items = "\n".join(_fmt_item(i + 1, f, with_gold) for i, f in enumerate(chunk))
        out = _gpt_json(head.format(items=items), key_file, dry_run=dry_run)
        for i, f in enumerate(chunk):
            hint = f"[dry_run {level} hint]" if dry_run else str(out.get(str(i + 1), "")).strip()
            if hint:
                instances[question_key(f["question"])] = {"hint": hint, "p": p}
    return instances


def annotate_buckets(failures, p, key_file, dry_run):
    """T1: per-bucket distribution skill."""
    by_bucket = defaultdict(list)
    for f in failures:
        by_bucket[HOP.get(str(f.get("data_source", "")).lower(), "multi_hop")].append(f)
    members = {"single_hop": [s for s in KNOWN_SOURCES if HOP[s] == "single_hop"],
               "multi_hop": [s for s in KNOWN_SOURCES if HOP[s] == "multi_hop"]}
    buckets = {}
    for bucket, fs in members.items():
        items = "\n".join(_fmt_item(i + 1, f, with_gold=False) for i, f in enumerate(by_bucket.get(bucket, [])[:40]))
        if dry_run:
            skill = f"[dry_run T1 {bucket} skill]"
        else:
            out = _gpt_json(_T1_HEAD.format(bucket=bucket, members=fs, items=items or "(none)"), key_file)
            skill = str(out.get("skill", "")).strip()
        buckets[bucket] = {"members": fs, "skill": skill, "p": p}
    return buckets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture", required=True)      # capture_search.py output
    ap.add_argument("--level", required=True, choices=["T1", "T2", "T3"])
    ap.add_argument("--out", required=True)          # scaffold JSON to write
    ap.add_argument("--p", type=float, default=1.0)  # initial injection prob (withdrawal reduces it later)
    ap.add_argument("--batch", type=int, default=12)
    ap.add_argument("--key_file", default=DEFAULT_KEY_FILE)
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    cap = json.load(open(args.capture))
    failures = cap.get("failures", [])
    if args.level == "T1":
        scaffold = {"level": "T1", "mode": "full", "default_p": args.p,
                    "buckets": annotate_buckets(failures, args.p, args.key_file, args.dry_run)}
        n = len(scaffold["buckets"])
    else:
        instances = annotate_instances(failures, args.level, args.p, args.key_file, args.batch, args.dry_run)
        scaffold = {"level": args.level, "mode": "full", "default_p": args.p, "instances": instances}
        n = len(instances)

    tmp = args.out + ".tmp"
    json.dump(scaffold, open(tmp, "w"), indent=2, ensure_ascii=False)
    os.replace(tmp, args.out)
    print(f"ANNOTATE_DONE level={args.level} from {len(failures)} failures -> {n} entries -> {args.out}")


if __name__ == "__main__":
    main()
