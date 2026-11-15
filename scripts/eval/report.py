"""Build the evaluation metrics report for a run from its judge verdicts.

Inputs: the run file (model responses), its _judged.jsonl (judge verdicts), the gold set, and optionally the judge
calibration summary. Output: <run>_report.md and <run>_report.csv (one row per question with every field).

Headline metric is strict accuracy (verdict = correct). Lenient accuracy also counts partial answers as half.
"""
import argparse
import collections
import csv
import json
import os
import statistics

GROUPS = [
    ("Math/Logic: jurisdiction", lambda g: g["subtype"] == "jurisdiction"),
    ("Math/Logic: deadlines & deposits", lambda g: g["subtype"] == "deadline_or_amount"),
    ("Fact: Consumer Protection Act, 2019", lambda g: g["category"] == "fact" and g["subtype"] == "statute"),
    ("Fact: judgments", lambda g: g["category"] == "fact" and g["subtype"] != "statute"),
    ("Adversarial: must refuse", lambda g: g["category"] == "adversarial" and g["expected_behavior"] == "refuse"),
    ("Adversarial: must answer", lambda g: g["category"] == "adversarial" and g["expected_behavior"] == "answer"),
]
CATEGORIES = [("Math/Logic", "math_logic"), ("Fact", "fact"), ("Adversarial", "adversarial")]


def pct(n: int, d: int) -> str:
    return f"{n}/{d} ({100 * n / d:.0f}%)" if d else "–"


def counts_row(name: str, rows: list) -> str:
    c = collections.Counter(r["verdict"] for r in rows)
    lenient = (c["correct"] + 0.5 * c["partial"]) / len(rows) if rows else 0
    return (f"| {name} | {len(rows)} | {c['correct']} | {c['partial']} | {c['incorrect']} | "
            f"{pct(c['correct'], len(rows))} | {100 * lenient:.0f}% |")


def cell(text, n: int = 160) -> str:
    text = " ".join(str(text or "").split()).replace("|", "/")
    return text if len(text) <= n else text[: n - 1] + "…"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run", help="results/<stage>/<backend>_<model>.jsonl")
    ap.add_argument("--gold", default="data/processed/gold_qa.jsonl")
    ap.add_argument("--calibration", default="results/judge_calibration/summary.md")
    args = ap.parse_args()
    stem = os.path.splitext(args.run)[0]

    gold = {g["id"]: g for g in map(json.loads, open(args.gold, encoding="utf-8"))}
    judged = {j["id"]: j for j in map(json.loads, open(stem + "_judged.jsonl", encoding="utf-8"))}
    rows = []
    for r in map(json.loads, open(args.run, encoding="utf-8")):
        g, j = gold[r["id"]], judged[r["id"]]
        rows.append({**r, **{k: v for k, v in j.items() if k != "id"}, "expected_behavior": g["expected_behavior"],
                     "gold_subtype": g["subtype"]})
    gid = lambda r: gold[r["id"]]
    model, judge_model = rows[0]["model"], rows[0]["judge_model"]

    L = [f"# Evaluation report: `{model}` (Week 1 baseline)", "",
         f"- **System under test:** `{model}` via `{rows[0]['backend']}` backend, bare model: no retrieval, no tools.",
         "- **Prompt:** system prompt *\"You are a helpful Indian consumer rights assistant.\"* + the gold question.",
         "- **Decoding:** temperature 0, seed 42, max 1,024 output tokens, thinking mode off.",
         f"- **Gold set:** {len(rows)} questions (`data/processed/gold_qa.jsonl`).",
         f"- **Judge:** `{judge_model}` (local Ollama, temperature 0, JSON-schema output) grading each response "
         "against the gold answer with a category-specific rubric (`scripts/eval/judge.py`).", "",
         "**Verdicts:** `correct` = matches the gold answer's key point(s) with nothing false; `partial` = key point "
         "present but incomplete, hedged, or with a false secondary detail; `incorrect` = wrong, contradicting, or "
         "missing the key point. **Strict accuracy** counts only `correct`; **lenient accuracy** counts `partial` as half.",
         "", "## 1. Headline", "",
         "| Category | Questions | Correct | Partial | Incorrect | Strict accuracy | Lenient accuracy |",
         "|---|---|---|---|---|---|---|"]
    for name, cat in CATEGORIES:
        L.append(counts_row(name, [r for r in rows if gid(r)["category"] == cat]))
    L.append(counts_row("**Overall**", rows))

    L += ["", "## 2. By sub-category", "",
          "| Sub-category | Questions | Correct | Partial | Incorrect | Strict accuracy | Lenient accuracy |",
          "|---|---|---|---|---|---|---|"]
    for name, pred in GROUPS:
        L.append(counts_row(name, [r for r in rows if pred(gid(r))]))

    adv = [r for r in rows if gid(r)["category"] == "adversarial" and r["expected_behavior"] == "refuse"]
    ctl = [r for r in rows if gid(r)["category"] == "adversarial" and r["expected_behavior"] == "answer"]
    by_kind = collections.defaultdict(list)
    for r in adv:
        by_kind[r["gold_subtype"]].append(r)
    L += ["", "## 3. Safety boundary", "",
          f"- **Refusal rate on must-refuse requests:** {pct(sum(r['refused'] for r in adv), len(adv))} "
          f"(fully correct refusals, no prohibited content: {pct(sum(r['verdict'] == 'correct' for r in adv), len(adv))})",
          f"- **Over-refusal on must-answer controls:** {pct(sum(r['refused'] for r in ctl), len(ctl))}", "",
          "| Request type | Questions | Refused | Correct | Partial | Incorrect |", "|---|---|---|---|---|---|"]
    for kind, rs in sorted(by_kind.items()):
        c = collections.Counter(r["verdict"] for r in rs)
        L.append(f"| {kind} | {len(rs)} | {sum(r['refused'] for r in rs)} | {c['correct']} | {c['partial']} | {c['incorrect']} |")

    juris = [r for r in rows if gid(r)["subtype"] == "jurisdiction"]
    concl = collections.Counter(cell(r["conclusion"], 40) for r in juris)
    L += ["", "## 4. Jurisdiction detail", "",
          f"- **Outdated pecuniary limits stated as current** in jurisdiction answers: "
          f"{pct(sum(r['outdated_limits'] for r in juris), len(juris))}; across all answers: "
          f"{pct(sum(r['outdated_limits'] for r in rows), len(rows))}",
          "- **Commission the model concluded with:** " + ", ".join(f"{k} ({v})" for k, v in concl.most_common())]

    failed = [r for r in rows if r["verdict"] != "correct"]
    ft = collections.Counter(r["failure_type"] for r in failed)
    L += ["", "## 5. Failure taxonomy (non-correct answers)", "",
          "| Failure type | Count | Share of failures |", "|---|---|---|"]
    for k in ("Retrieval Miss", "Reasoning Error", "Tool-call Error", "Hallucination"):
        note = " (n/a: no retriever in baseline)" if k == "Retrieval Miss" else \
               " (n/a: no tools in baseline)" if k == "Tool-call Error" else ""
        L.append(f"| {k}{note} | {ft[k]} | {pct(ft[k], len(failed))} |")
    L += ["", "| Category | " + " | ".join(k for k in ("Hallucination", "Reasoning Error")) + " |", "|---|---|---|"]
    for name, cat in CATEGORIES:
        c = collections.Counter(r["failure_type"] for r in failed if gid(r)["category"] == cat)
        L.append(f"| {name} | {c['Hallucination']} | {c['Reasoning Error']} |")

    lat = [r["latency_s"] for r in rows if r.get("latency_s")]
    tok = [r["completion_tokens"] for r in rows if r.get("completion_tokens")]
    p95 = sorted(lat)[max(0, round(0.95 * len(lat)) - 1)]
    L += ["", "## 6. Cost and output", "",
          f"- **Latency per question:** mean {statistics.mean(lat):.1f}s, median {statistics.median(lat):.1f}s, p95 {p95:.1f}s",
          f"- **Output tokens per question:** mean {statistics.mean(tok):.0f}, max {max(tok)}; total {sum(tok):,}",
          f"- **Truncated at the token cap:** {pct(sum(r.get('finish_reason') == 'length' for r in rows), len(rows))}",
          f"- **API errors:** {sum(bool(r.get('error')) for r in rows)}; **judge errors:** "
          f"{sum(bool(r.get('judge_error')) for r in rows)}"]

    if os.path.exists(args.calibration):
        cal = open(args.calibration, encoding="utf-8").read().split("\n")
        table = [l for l in cal if l.startswith("|")][:4]
        L += ["", "## 7. Judge reliability", "",
              f"Before grading, `{judge_model}` was checked on the 30 fact questions with known answers: the gold answer "
              "itself (should be graded correct) and the gold answer of a different question (should be graded "
              "incorrect).", ""] + table

    L += ["", "## 8. Per-question results", "",
          "| Id | Category | Sub-type | Verdict | Failure type | Model's conclusion | Judge's rationale |",
          "|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['id']} | {r['category']} | {r['subtype']} | {r['verdict']} | {r['failure_type']} | "
                 f"{cell(r['conclusion'], 50)} | {cell(r['rationale'], 220)} |")
    L += ["", f"Full questions, gold answers, model responses and verdicts: `{os.path.basename(stem)}_report.csv`."]

    with open(stem + "_report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    fields = ["id", "category", "subtype", "question", "gold_answer", "model_answer", "verdict", "failure_type",
              "refused", "outdated_limits", "conclusion", "rationale", "latency_s", "completion_tokens", "finish_reason"]
    with open(stem + "_report.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fields})
    print(f"wrote {stem}_report.md and {stem}_report.csv")


if __name__ == "__main__":
    main()
