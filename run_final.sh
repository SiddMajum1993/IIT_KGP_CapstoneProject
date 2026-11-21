#!/usr/bin/env bash
# The final API session: everything that needs the course APIs, in one go.
# Run it with the VPN / Zscaler OFF (the APIs are not reachable through it):   bash run_final.sh
#
#   1. check that both APIs answer
#   2. build the two Chroma indexes with API embeddings (256 and 500 BERT-token chunks)
#   3. Recall@3 of the retriever alone
#   4. the five remaining runs (zero-shot / no RAG already exists from Week 1)
#
# Every step is resumable: if the network drops, run the script again and it carries on where it stopped.
# Judging and the results sheet run locally afterwards and do not need the APIs (see the end of this file).
set -euo pipefail
cd "$(dirname "$0")"
PY=.venv/bin/python

echo "== 1. API check"
$PY - <<'EOF'
from copilot import llm, rag
client, model, extra = llm.make_client("api")
r = llm.chat(client, model, extra, [{"role": "user", "content": "Reply with the word OK."}])
assert r["answer"], f"chat API failed: {r['error']}"
print(f"chat API ok: {model} answered {r['answer']!r} in {r['latency_s']}s")
v = rag.APIEmbeddings().embed_query("consumer complaint")
print(f"embedding API ok: {len(v)} dimensions")
EOF

echo "== 2-3. Indexes and Recall@3 (builds chroma_db/api_256 and chroma_db/api_500 on first use)"
$PY -m copilot.recall --mode api

echo "== 4. Runs"
for rag in off on; do
  for prompt in zero few cot; do
    [ "$prompt" = zero ] && [ "$rag" = off ] && continue   # the Week 1 baseline
    $PY -m copilot.run --mode api --prompt "$prompt" --rag "$rag"
  done
done

echo "== API work done. Check that no row is missing an answer:"
$PY - <<'EOF'
import glob, json
for p in sorted(glob.glob("results/week2/api_*_*.jsonl")):
    if p.endswith("_judged.jsonl"):
        continue
    rows = [json.loads(l) for l in open(p)]
    print(f"{p}: {len(rows)} rows, {sum(r['answer'] is None for r in rows)} without an answer")
EOF

cat <<'EOF'

Next (VPN can be back on; judging uses local Ollama, about 20 s per answer, ~2 hours for five runs):
  for f in results/week2/api_*_{norag,rag}.jsonl; do .venv/bin/python -m copilot.judge "$f"; done
  .venv/bin/python -m copilot.report --mode api
EOF
