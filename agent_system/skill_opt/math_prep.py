"""Convert QuestA's filtered OpenR1 jsonl (problem + R1 solution + answer) into a verl
parquet for math GRPO/DAPO. The FULL post-think solution is carried in extra_info so our
DYNAMIC scaffold can truncate it to any hint fraction live during training (no fixed prefix
baked in — that is the difference from QuestA's static add_prefix.py)."""
import argparse, json, os
import pandas as pd

INSTR = "\nLet's think step by step and output the final answer within \\boxed{}."


def solution_of(generation):
    g = generation if isinstance(generation, str) else ""
    if g and g[0] == '"':
        g = g[1:-1]
    return g.split("</think>")[-1].strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", default="/mnt/data1/zha00175/data/QuestA_ds/OpenR1-50-0-4.jsonl")
    ap.add_argument("--out", default="/mnt/data1/zha00175/data/questa_math/train.parquet")
    ap.add_argument("--data_source", default="openr1_math")
    args = ap.parse_args()

    rows, dropped = [], 0
    for i, line in enumerate(open(args.jsonl)):
        try:
            d = json.loads(line)
        except Exception:
            continue
        problem, answer = d.get("problem", ""), str(d.get("answer", "")).strip()
        solution = solution_of(d.get("generation", ""))
        if not problem or not answer or len(solution) < 10 or answer not in solution:
            dropped += 1
            continue
        rows.append({
            "data_source": args.data_source,
            "prompt": [{"role": "user", "content": problem + INSTR}],
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": answer},
            "extra_info": {"query_id": str(i), "solution": solution, "problem": problem, "answer": answer},
        })
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    pd.DataFrame(rows).to_parquet(args.out)
    print(f"MATH_PREP_DONE kept={len(rows)} dropped={dropped} -> {args.out}")
    print(f"  sample query_id={rows[0]['extra_info']['query_id']} sol_len={len(rows[0]['extra_info']['solution'])} "
          f"answer={rows[0]['reward_model']['ground_truth']}")


if __name__ == "__main__":
    main()
