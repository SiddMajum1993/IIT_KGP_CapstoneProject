"""Ask the model every gold question with one prompt style, with or without RAG, and save the answers.

Usage:  python -m copilot.run --mode local --prompt zero --rag off
        python -m copilot.run --mode api --prompt cot --rag on

Output: results/week2/<mode>_<prompt>_<rag|norag>.jsonl, one row per gold question.
The run is resumable: answered rows are kept, and rows that failed (no answer) are asked again.
"""
import argparse
import json
import os

from copilot import config, llm, prompts


def load_done(out: str) -> list:
    """Rows already answered. Failed rows are dropped from the file so they are asked again."""
    if not os.path.exists(out):
        return []
    rows = [json.loads(line) for line in open(out, encoding="utf-8")]
    good = [r for r in rows if r.get("answer") is not None]
    if len(good) < len(rows):
        with open(out, "w", encoding="utf-8") as f:
            f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in good)
    return good


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["local", "api"], default=config.get("COPILOT_MODE"))
    ap.add_argument("--prompt", choices=prompts.STYLES, default="zero")
    ap.add_argument("--rag", choices=["on", "off"], default="off")
    ap.add_argument("--limit", type=int, help="only the first N questions (smoke test)")
    args = ap.parse_args()

    out = f"{config.RESULTS_DIR}/{args.mode}_{args.prompt}_{'rag' if args.rag == 'on' else 'norag'}.jsonl"
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    gold = [json.loads(line) for line in open(config.GOLD_PATH, encoding="utf-8")][: args.limit]
    done = {r["id"] for r in load_done(out)}
    todo = [g for g in gold if g["id"] not in done]

    client, model, extra_body = llm.make_client(args.mode)
    retriever = None
    if args.rag == "on":
        from copilot import rag  # loaded only when needed: it pulls in the embedding and reranker models
        retriever = rag.load_retriever(args.mode)
    print(f"{model} | prompt={args.prompt} rag={args.rag}: {len(gold)} questions, {len(done)} done, "
          f"{len(todo)} to run -> {out}")

    with open(out, "a", encoding="utf-8") as f:
        for i, g in enumerate(todo, 1):
            docs = retriever.search(g["question"]) if retriever else []
            messages = prompts.build_messages(g["question"], args.prompt, [d.page_content for d in docs] or None)
            res = llm.chat(client, model, extra_body, messages)
            row = dict(id=g["id"], category=g["category"], subtype=g["subtype"], mode=args.mode, model=model,
                       prompt=args.prompt, rag=args.rag, question=g["question"], gold_answer=g["answer"],
                       retrieved_ids=[d.metadata["qa_id"] for d in docs], **res)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            status = res["error"] or f"{res['latency_s']}s, {res['completion_tokens']} tok, {res['finish_reason']}"
            print(f"[{i}/{len(todo)}] {g['id']} {g['category']}/{g['subtype']}: {status}", flush=True)


if __name__ == "__main__":
    main()
