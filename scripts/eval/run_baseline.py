"""Week 1 baseline: ask the bare model every gold question (no retrieval, no tools) and log the answers.

Backends (settings come from the environment or .env; see .env.example):
  * local - Ollama (LOCAL_LLM_BASE_URL, LOCAL_LLM_MODEL)
  * class - the course inference API (LLM_BASE_URL, LLM_MODEL, LLM_API_KEY or LLM_API_KEY_FILE, LLM_CA_CERT).
            Keys and certificates are read at runtime and never written into code or results.

The run is resumable: rows already in the output file are skipped.

Output: results/baseline/<backend>_<model>.jsonl, one row per gold question.
"""
import argparse
import json
import os
import re
import sys
import time

import httpx
from openai import APIConnectionError, APIStatusError, OpenAI

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config  # noqa: E402

SYSTEM_PROMPT = "You are a helpful Indian consumer rights assistant."  # PRD Week 1 Step 4
RETRY_DELAYS = [2, 4, 8, 16]  # seconds, for 429 / 503 / dropped connections

BACKENDS = {
    # Ollama's OpenAI endpoint turns Qwen3 thinking off with reasoning_effort="none".
    "local": dict(model_var="LOCAL_LLM_MODEL", extra_body={"reasoning_effort": "none"}),
    # vLLM-style servers take the chat-template switch instead.
    "class": dict(model_var="LLM_MODEL", extra_body={"chat_template_kwargs": {"enable_thinking": False}}),
}


def make_client(backend: str) -> OpenAI:
    if backend == "local":
        return OpenAI(base_url=config.get("LOCAL_LLM_BASE_URL"), api_key="ollama")  # Ollama ignores the key
    base_url, key, cert = config.get("LLM_BASE_URL"), config.secret("LLM_API_KEY"), config.get("LLM_CA_CERT")
    assert base_url and key, "set LLM_BASE_URL and LLM_API_KEY (or LLM_API_KEY_FILE) in .env (see .env.example)"
    http = httpx.Client(verify=os.path.expanduser(cert)) if cert else None
    return OpenAI(base_url=base_url, api_key=key, http_client=http)


def ask(client: OpenAI, model: str, question: str, extra_body: dict, max_tokens: int) -> dict:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": question}]
    for attempt, delay in enumerate([0] + RETRY_DELAYS):
        time.sleep(delay)
        start = time.perf_counter()
        try:
            r = client.chat.completions.create(model=model, messages=messages, temperature=0.0, seed=42,
                                               max_tokens=max_tokens, extra_body=extra_body)
        except APIStatusError as e:
            if e.status_code in (429, 503) and attempt < len(RETRY_DELAYS):
                continue
            return dict(model_answer=None, error=f"HTTP {e.status_code}", attempts=attempt + 1)
        except APIConnectionError as e:
            if attempt < len(RETRY_DELAYS):
                continue
            return dict(model_answer=None, error=f"connection: {e}", attempts=attempt + 1)
        content = r.choices[0].message.content or ""
        # Thinking should be off; strip any stray <think> block so it is not scored as the answer.
        content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
        return dict(
            model_answer=content, error=None, attempts=attempt + 1,
            latency_s=round(time.perf_counter() - start, 2),
            prompt_tokens=r.usage.prompt_tokens if r.usage else None,
            completion_tokens=r.usage.completion_tokens if r.usage else None,
            finish_reason=r.choices[0].finish_reason,
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=list(BACKENDS), default="local")
    ap.add_argument("--model", help="override the backend's default model")
    ap.add_argument("--gold", default="data/processed/gold_qa.jsonl")
    ap.add_argument("--out", help="default: results/baseline/<backend>_<model>.jsonl")
    ap.add_argument("--max-tokens", type=int, default=1024, help="the course API allows at most 1024")
    ap.add_argument("--limit", type=int, help="only the first N questions (smoke test)")
    args = ap.parse_args()

    cfg = BACKENDS[args.backend]
    model = args.model or config.get(cfg["model_var"])
    out = args.out or f"results/baseline/{args.backend}_{re.sub(r'[^A-Za-z0-9._-]+', '-', model)}.jsonl"
    os.makedirs(os.path.dirname(out), exist_ok=True)

    gold = [json.loads(line) for line in open(args.gold, encoding="utf-8")][: args.limit]
    done = set()
    if os.path.exists(out):
        done = {json.loads(line)["id"] for line in open(out, encoding="utf-8")}
    todo = [g for g in gold if g["id"] not in done]
    print(f"{args.backend}/{model}: {len(gold)} gold rows, {len(done)} already done, {len(todo)} to run -> {out}")

    client = make_client(args.backend)
    with open(out, "a", encoding="utf-8") as f:
        for i, g in enumerate(todo, 1):
            res = ask(client, model, g["question"], cfg["extra_body"], args.max_tokens)
            row = dict(id=g["id"], category=g["category"], subtype=g["subtype"], backend=args.backend, model=model,
                       question=g["question"], gold_answer=g["answer"], **res)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            status = res["error"] or f"{res['latency_s']}s, {res['completion_tokens']} tok, {res['finish_reason']}"
            print(f"[{i}/{len(todo)}] {g['id']} {g['category']}/{g['subtype']}: {status}", flush=True)


if __name__ == "__main__":
    main()
