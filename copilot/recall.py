"""Retrieval evaluation: Recall@3 of the retriever alone, with no LLM involved.

For every gold question that names a source passage (gold_context_ids), a method scores a hit if any of its
top 3 chunks comes from that master-set row. Measured for each method (BM25, dense, hybrid, hybrid + rerank)
and each chunk size, so a retrieval miss can be told apart from a reasoning error later.

Usage:  python -m copilot.recall --mode local
Output: results/week2/recall_<mode>.md and .jsonl
"""
import argparse
import json
import os

from copilot import config, rag

CHUNK_SIZES = [256, 500]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["local", "api"], default=config.get("COPILOT_MODE"))
    args = ap.parse_args()

    gold = [g for g in map(json.loads, open(config.GOLD_PATH, encoding="utf-8")) if g["gold_context_ids"]]
    rows = []
    for size in CHUNK_SIZES:
        retriever = rag.load_retriever(args.mode, size)
        for g in gold:
            for method in rag.METHODS:
                found = [d.metadata["qa_id"] for d in retriever.search(g["question"], method)]
                rows.append(dict(id=g["id"], category=g["category"], chunk_size=size, method=method,
                                 retrieved=found, hit=any(i in g["gold_context_ids"] for i in found)))
        print(f"chunk size {size}: done")

    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    out = f"{config.RESULTS_DIR}/recall_{args.mode}"
    with open(out + ".jsonl", "w", encoding="utf-8") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)

    groups = [("Fact questions", "fact"), ("All questions with a gold passage", None)]
    lines = [f"# Recall@3 - retriever only ({args.mode} mode)", "",
             "| Chunk size (BERT tokens) | Method | " + " | ".join(name for name, _ in groups) + " |",
             "|---|---|" + "---|" * len(groups)]
    for size in CHUNK_SIZES:
        for method in rag.METHODS:
            cells = []
            for _, cat in groups:
                sel = [r for r in rows if r["chunk_size"] == size and r["method"] == method
                       and (cat is None or r["category"] == cat)]
                cells.append(f"{sum(r['hit'] for r in sel)}/{len(sel)} ({sum(r['hit'] for r in sel) / len(sel):.0%})")
            lines.append(f"| {size} | {method} | " + " | ".join(cells) + " |")
    open(out + ".md", "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
