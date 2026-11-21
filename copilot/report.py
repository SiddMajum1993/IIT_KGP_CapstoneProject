"""The results sheet: every run (prompt style x RAG) against every metric, in one Excel file and one Markdown file.

Usage:  python -m copilot.report --mode api
        (reads results/week2/<mode>_*.jsonl that have a _judged.jsonl next to them, plus recall_<mode>.md)
Output: results/week2/results_<mode>.xlsx and results_<mode>.md

Sheets in the Excel file:
  Summary      - one row per run: ROUGE, exact match, judge score, strict / lenient accuracy, judge type accuracy,
                 jurisdiction and safety F1, failure counts, outdated limits, latency.
  By category  - strict accuracy per question group, one column per run.
  Per question - every answer with its scores, for reading individual cases.
  Definitions  - what each metric means.
"""
import argparse
import glob
import os

from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill

from copilot import config
from copilot.judge import FAILURE_TYPES
from copilot.metrics import GROUPS, run_metrics

PROMPT_NAMES = {"zero": "Zero-shot", "few": "Few-shot", "cot": "Chain-of-thought"}
FAILURES = ["Retrieval Miss"] + FAILURE_TYPES

DEFINITIONS = [
    ("ROUGE-1 / ROUGE-2 / ROUGE-L", "F1 overlap of words / word pairs / longest common word sequence between the "
     "answer and the gold answer. No LLM involved. Low for long answers even when they are right."),
    ("Exact match", "Math/Logic only: the gold answer's key value (the commission, the date or the amount) "
     "appears word for word in the answer."),
    ("Judge score", "1-5 from the LLM judge (gemma3:27b), which is not told the question type."),
    ("Strict / lenient accuracy", "Share of answers scored 5 / scored 4 or 5."),
    ("Judge type accuracy", "Share of questions where the judge's own guess of the question type matches the gold "
     "set. Shows whether the judge understood what it was grading."),
    ("Jurisdiction F1", "District / State / National classification of the final recommendation, macro-averaged."),
    ("Safety F1", "Refuse vs answer over the adversarial questions; positive class = refuse. Recall = share of "
     "must-refuse requests that were refused."),
    ("Failure types", "For answers scored below 5. Hallucination and Reasoning Error come from the judge; "
     "Retrieval Miss is set when a RAG answer failed and the gold passage was not retrieved."),
    ("Outdated limits", "Jurisdiction answers that quote the old Rs 20 lakh / Rs 1 crore limits as current."),
]


def run_label(path: str) -> str:
    """results/week2/api_cot_rag.jsonl -> "Chain-of-thought + RAG"."""
    _, prompt, rag = os.path.basename(path)[:-len(".jsonl")].split("_")
    return f"{PROMPT_NAMES[prompt]} + {'RAG' if rag == 'rag' else 'no RAG'}"


def find_runs(mode: str) -> list:
    runs = sorted(p for p in glob.glob(f"{config.RESULTS_DIR}/{mode}_*_*.jsonl")
                  if not p.endswith("_judged.jsonl") and os.path.exists(p[:-len(".jsonl")] + "_judged.jsonl"))
    order = {f"{p}_{r}": i for i, (p, r) in enumerate((p, r) for r in ("norag", "rag") for p in PROMPT_NAMES)}
    return sorted(runs, key=lambda p: order.get(os.path.basename(p)[len(mode) + 1:-len(".jsonl")], 99))


def summary_row(label: str, m: dict) -> dict:
    o = m["overall"]
    return {"Run": label, "Questions": o["questions"], "ROUGE-1": o["rouge1"], "ROUGE-2": o["rouge2"],
            "ROUGE-L": o["rougeL"], "Exact match": o["exact_match"], "Judge score (1-5)": o["mean_score"],
            "Strict accuracy": o["strict"], "Lenient accuracy": o["lenient"],
            "Judge type accuracy": o["type_accuracy"], "Jurisdiction F1": m["jurisdiction"]["f1"],
            "Safety recall": m["safety"]["recall"], "Safety F1": m["safety"]["f1"],
            **{f: m["failures"].get(f, 0) for f in FAILURES},
            "Outdated limits": m["outdated_limits"], "Truncated": m["truncated"],
            "Median latency (s)": m["median_latency_s"]}


def write_sheet(ws, rows: list) -> None:
    ws.append(list(rows[0]))
    for r in rows:
        ws.append([round(v, 3) if isinstance(v, float) else v for v in r.values()])
    for cell in ws[1]:
        cell.font = Font(bold=True, color="FFFFFF")
        cell.fill = PatternFill("solid", fgColor="305496")
        cell.alignment = Alignment(wrap_text=True, vertical="top")
    for col in ws.columns:
        ws.column_dimensions[col[0].column_letter].width = min(60, max(10, *(len(str(c.value or ""))
                                                                              for c in col[:30])) + 2)
    ws.freeze_panes = "B2"


def markdown_table(rows: list, pct: set) -> list:
    cell = lambda k, v: "-" if v is None else f"{v:.0%}" if k in pct else f"{v:.2f}" if isinstance(v, float) \
        else str(v)
    return ["| " + " | ".join(rows[0]) + " |", "|" + "---|" * len(rows[0])] + \
           ["| " + " | ".join(cell(k, v) for k, v in r.items()) + " |" for r in rows]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["local", "api"], default=config.get("COPILOT_MODE"))
    mode = ap.parse_args().mode
    runs = find_runs(mode)
    assert runs, f"no judged runs found in {config.RESULTS_DIR} for mode {mode}"
    metrics = {run_label(p): run_metrics(p) for p in runs}

    summary = [summary_row(label, m) for label, m in metrics.items()]
    by_cat = [{"Question group": name, "Questions": next(iter(metrics.values()))["groups"][name]["questions"],
               **{label: m["groups"][name]["strict"] for label, m in metrics.items()}} for name, _ in GROUPS]
    per_q = [{"Run": label, "ID": r["id"], "Category": r["category"], "Subtype": r["subtype"],
              "Question": r["gold"]["question"], "Gold answer": r["gold"]["answer"], "Answer": r["answer"],
              "ROUGE-L": r["rougeL"], "Exact match": r["exact_match"], "Score": r.get("score"),
              "Judge type": r.get("question_type"), "Gold type": r.get("gold_type"),
              "Failure type": r.get("failure_type"), "Judge rationale": r.get("answer_rationale")}
             for label, m in metrics.items() for r in m["rows"]]

    wb = Workbook()
    wb.remove(wb.active)
    for title, rows in [("Summary", summary), ("By category", by_cat), ("Per question", per_q),
                        ("Definitions", [{"Metric": k, "Meaning": v} for k, v in DEFINITIONS])]:
        write_sheet(wb.create_sheet(title), rows)
    out = f"{config.RESULTS_DIR}/results_{mode}"
    wb.save(out + ".xlsx")

    pct = {"Exact match", "Strict accuracy", "Lenient accuracy", "Judge type accuracy", "Safety recall"} | \
          set(metrics)
    lines = [f"# Week 2 results ({mode} mode)", "",
             f"Model: {next(iter(metrics.values()))['rows'][0].get('model', '')}. Judge: {config.get('JUDGE_MODEL')}, "
             "not told the question type.", "", "## Summary", ""] + markdown_table(summary, pct) + \
            ["", "## Strict accuracy by question group", ""] + markdown_table(by_cat, pct)
    recall = f"{config.RESULTS_DIR}/recall_{mode}.md"
    if os.path.exists(recall):
        lines += ["", "## Retrieval (no LLM)", ""] + open(recall, encoding="utf-8").read().splitlines()[2:]
    lines += ["", "## Definitions", ""] + [f"- **{k}**: {v}" for k, v in DEFINITIONS]
    open(out + ".md", "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print(f"wrote {out}.xlsx and {out}.md ({len(runs)} runs)")


if __name__ == "__main__":
    main()
