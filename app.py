"""Prototype: ask one question and see the answer without RAG and with RAG, side by side.

Usage:
  python app.py --id G045                          a gold question (also shows the gold answer)
  python app.py "Can I appeal a District Commission order?"
  python app.py --mode api --prompt cot --id G021

The answer with RAG also lists the three passages the retriever passed to the model.
"""
import argparse
import json
import re
import textwrap

from copilot import config, llm, prompts, rag

DISCLAIMER = ("This is a procedural advisor only, not legal advice. A qualified lawyer must review any legal "
              "action before you take it.")


def scrub(text: str) -> str:
    """Remove phone numbers and e-mail addresses before the question reaches the model."""
    text = re.sub(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b", "[email]", text)
    return re.sub(r"(\+91[\s-]?)?\b[6-9]\d{4}[\s-]?\d{5}\b", "[phone]", text)


def show(title: str, text: str) -> None:
    print(f"\n=== {title} " + "=" * max(0, 70 - len(title)))
    for para in (text or "").splitlines():
        print(textwrap.fill(para, 100) if para.strip() else "")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("question", nargs="?", help="your own question")
    ap.add_argument("--id", help="a gold question id, e.g. G045")
    ap.add_argument("--mode", choices=["local", "api"], default=config.get("COPILOT_MODE"))
    ap.add_argument("--prompt", choices=prompts.STYLES, default="zero")
    args = ap.parse_args()
    assert args.question or args.id, "give a question or --id"

    gold = None
    if args.id:
        gold = next(g for g in map(json.loads, open(config.GOLD_PATH, encoding="utf-8")) if g["id"] == args.id)
    question = scrub(gold["question"] if gold else args.question)
    client, model, extra_body = llm.make_client(args.mode)
    retriever = rag.load_retriever(args.mode)

    show("Question", question)
    if gold:
        show("Gold answer", gold["answer"])

    plain = llm.chat(client, model, extra_body, prompts.build_messages(question, args.prompt))
    show(f"{model} without RAG", plain["answer"] or plain["error"])

    docs = retriever.search(question)
    grounded = llm.chat(client, model, extra_body,
                        prompts.build_messages(question, args.prompt, [d.page_content for d in docs]))
    show(f"{model} with RAG", grounded["answer"] or grounded["error"])
    show("Passages given to the model", "\n".join(
        f"[{i}] {d.metadata['qa_id']}: {d.page_content[:200]}..." for i, d in enumerate(docs, 1)))
    if gold and gold["gold_context_ids"]:
        hit = any(d.metadata["qa_id"] in gold["gold_context_ids"] for d in docs)
        print(f"\nGold passage {gold['gold_context_ids']} retrieved: {'yes' if hit else 'no'}")
    print(f"\n{DISCLAIMER}")


if __name__ == "__main__":
    main()
