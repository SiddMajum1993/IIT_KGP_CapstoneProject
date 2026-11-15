"""Check the judge before trusting it: grade known-right and known-wrong answers on the fact rows.

  * positive control - the response is the gold answer itself; the judge should say "correct".
  * negative control - the response is the gold answer of a different fact row (real legal text, wrong
                       answer); the judge should say "incorrect".

The negative pairing is a fixed rotation (row i gets row i+7's answer), so it is reproducible and no row is paired
with itself. Both control runs are written as ordinary run files and graded with judge.py, so the judge sees
exactly the same prompt it uses for real runs.

Output: results/judge_calibration/{positive,negative}.jsonl, their _judged.jsonl files, and summary.md
"""
import argparse
import collections
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config  # noqa: E402

SHIFT = 7


def write_run(path: str, rows: list) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", default="data/processed/gold_qa.jsonl")
    ap.add_argument("--out-dir", default="results/judge_calibration")
    ap.add_argument("--judge-model", default=config.get("JUDGE_MODEL"))
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    facts = [g for g in map(json.loads, open(args.gold, encoding="utf-8")) if g["category"] == "fact"]
    base = lambda g: dict(id=g["id"], category=g["category"], subtype=g["subtype"], question=g["question"],
                          gold_answer=g["answer"])
    controls = {
        "positive": [dict(base(g), model_answer=g["answer"]) for g in facts],
        "negative": [dict(base(g), model_answer=facts[(i + SHIFT) % len(facts)]["answer"]) for i, g in enumerate(facts)],
    }
    expected = {"positive": "correct", "negative": "incorrect"}

    lines = [f"# Judge calibration: `{args.judge_model}` on the {len(facts)} fact rows", "",
             "| Control | Expected verdict | Correct | Partial | Incorrect | Judge accuracy |", "|---|---|---|---|---|---|"]
    misses = []
    for name, rows in controls.items():
        run = os.path.join(args.out_dir, f"{name}.jsonl")
        write_run(run, rows)
        subprocess.run([sys.executable, "scripts/eval/judge.py", run, "--gold", args.gold,
                        "--judge-model", args.judge_model], check=True)
        judged = [json.loads(line) for line in open(os.path.splitext(run)[0] + "_judged.jsonl", encoding="utf-8")]
        c = collections.Counter(j["verdict"] for j in judged)
        ok = c[expected[name]]
        lines.append(f"| {name} | {expected[name]} | {c['correct']} | {c['partial']} | {c['incorrect']} | "
                     f"{ok}/{len(judged)} ({100 * ok / len(judged):.0f}%) |")
        misses += [(name, j) for j in judged if j["verdict"] != expected[name]]

    if misses:
        lines += ["", "## Rows the judge got wrong", "", "| Control | Id | Verdict | Judge's rationale |", "|---|---|---|---|"]
        lines += [f"| {n} | {j['id']} | {j['verdict']} | {j.get('rationale', j.get('judge_error', '')).replace('|', '/')} |"
                  for n, j in misses]
    with open(os.path.join(args.out_dir, "summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
