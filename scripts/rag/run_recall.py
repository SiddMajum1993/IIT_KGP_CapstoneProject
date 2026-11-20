"""Build/evaluate Week 2 hybrid RAG indexes on the 30 fact gold questions."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.rag.retrieval import (  # noqa: E402
    DEFAULT_CANDIDATE_K,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZES,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_RERANKER_MODEL,
    DEFAULT_TOP_K,
    chunk_corpus,
    hit_at_k,
    load_jsonl,
    make_embeddings,
    make_hybrid_retriever,
    make_reranker,
    make_vector_store,
    parse_context_ids,
    rerank_documents,
    validate_corpus,
)
from scripts.eval import config  # noqa: E402


def model_slug(model_name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", model_name).strip("-")


def fact_queries(gold_rows: List[Dict]) -> List[Dict]:
    facts = [row for row in gold_rows if row.get("category") == "fact"]
    if not facts:
        raise ValueError("The gold set contains no rows with category='fact'")
    for row in facts:
        if not parse_context_ids(row.get("gold_context_ids")):
            raise ValueError(f"{row.get('id', '<unknown>')} has no gold_context_ids")
    return facts


def evaluate_chunk_size(
    chunk_size: int,
    overlap: int,
    rows: List[Dict],
    queries: List[Dict],
    index_root: Path,
    embeddings,
    reranker,
    candidate_k: int,
    top_k: int,
    rebuild_index: bool,
) -> List[Dict]:
    documents = chunk_corpus(rows, chunk_size, overlap)
    print(f"chunk size {chunk_size}: {len(documents)} chunks", flush=True)

    vector_store = make_vector_store(
        documents=documents,
        embeddings=embeddings,
        persist_directory=index_root,
        collection_name=f"legal_chunks_{chunk_size}_{overlap}",
        rebuild=rebuild_index,
    )
    hybrid = make_hybrid_retriever(documents, vector_store, candidate_k)

    results = []
    for index, query_row in enumerate(queries, start=1):
        question = query_row["question"]
        gold_ids = parse_context_ids(query_row["gold_context_ids"])
        candidates = hybrid.invoke(question)
        ranked = rerank_documents(reranker, question, candidates, top_k)
        retrieved_ids = [
            document.metadata.get("qa_id", "")
            for document, _score in ranked
        ]
        hit = hit_at_k(ranked, gold_ids)
        results.append(
            {
                "id": query_row["id"],
                "chunk_size": chunk_size,
                "chunk_overlap": overlap,
                "question": question,
                "gold_context_ids": sorted(gold_ids),
                "retrieved_context_ids": retrieved_ids,
                "hit_at_3": hit,
            }
        )
        print(
            f"[{index}/{len(queries)}] {query_row['id']}: "
            f"{'hit' if hit else 'miss'} (top {top_k}: {', '.join(retrieved_ids)})",
            flush=True,
        )
    return results


def write_results(
    out_dir: Path,
    results: List[Dict],
    chunk_sizes: List[int],
    candidate_k: int,
    top_k: int,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "recall_at_3.jsonl"
    csv_path = out_dir / "recall_at_3.csv"
    report_path = out_dir / "recall_at_3.md"

    with jsonl_path.open("w", encoding="utf-8") as output:
        for result in results:
            output.write(json.dumps(result, ensure_ascii=False) + "\n")

    fieldnames = [
        "id",
        "chunk_size",
        "chunk_overlap",
        "hit_at_3",
        "gold_context_ids",
        "retrieved_context_ids",
        "question",
    ]
    with csv_path.open("w", encoding="utf-8-sig", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    **result,
                    "gold_context_ids": ";".join(result["gold_context_ids"]),
                    "retrieved_context_ids": ";".join(result["retrieved_context_ids"]),
                }
            )

    lines = [
        "# Week 2 Hybrid RAG: Recall@3",
        "",
        "Evaluation uses the fact questions only. A question is a hit when any of its "
        "gold master row IDs appears among the three cross-encoder-reranked hybrid results.",
        "",
        "| Chunk size (tokens) | Overlap (tokens) | Hits | Questions | Recall@3 |",
        "|---:|---:|---:|---:|---:|",
    ]
    for chunk_size in chunk_sizes:
        subset = [row for row in results if row["chunk_size"] == chunk_size]
        hits = sum(bool(row["hit_at_3"]) for row in subset)
        recall = hits / len(subset) if subset else 0.0
        lines.append(
            f"| {chunk_size} | {subset[0]['chunk_overlap'] if subset else '-'} "
            f"| {hits} | {len(subset)} | {recall:.1%} |"
        )
    lines.extend(
        [
            "",
            "The dense and BM25 retrievers contribute equal weight (0.5 each) to "
            "LangChain's ensemble rank fusion. The cross-encoder reranks the top "
            f"{candidate_k} hybrid candidates and returns the top {top_k}.",
            "",
        ]
    )
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {jsonl_path}, {csv_path}, and {report_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--master", default="data/processed/master_qa.jsonl")
    parser.add_argument("--gold", default="data/processed/gold_qa.jsonl")
    parser.add_argument("--index-dir", default="chroma_db")
    parser.add_argument("--out-dir", default="results/rag")
    parser.add_argument(
        "--chunk-sizes",
        nargs="+",
        type=int,
        default=list(DEFAULT_CHUNK_SIZES),
        help="Token chunk sizes to compare; provide at least two.",
    )
    parser.add_argument("--chunk-overlap", type=int, default=DEFAULT_CHUNK_OVERLAP)
    parser.add_argument("--candidate-k", type=int, default=DEFAULT_CANDIDATE_K)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument(
        "--embedding-backend",
        choices=("auto", "api", "huggingface"),
        default="auto",
        help="Use configured API credentials when present, otherwise Hugging Face.",
    )
    parser.add_argument(
        "--embedding-model",
        help="Override the model in .env (API) or the default Hugging Face model.",
    )
    parser.add_argument("--reranker-model", default=DEFAULT_RERANKER_MODEL)
    parser.add_argument("--device", help="Hugging Face device, e.g. cpu or cuda")
    parser.add_argument(
        "--rebuild-index",
        action="store_true",
        help="Replace the existing Chroma collection for each requested chunk size.",
    )
    args = parser.parse_args()

    if len(set(args.chunk_sizes)) < 2:
        parser.error("Provide at least two distinct --chunk-sizes for the required experiment")
    if args.chunk_overlap < 0 or any(args.chunk_overlap >= size for size in args.chunk_sizes):
        parser.error("--chunk-overlap must be smaller than every chunk size")
    if args.candidate_k < args.top_k:
        parser.error("--candidate-k must be at least --top-k")
    if args.top_k != DEFAULT_TOP_K:
        parser.error("This evaluation reports Recall@3; --top-k must be 3")

    master_path = Path(args.master)
    gold_path = Path(args.gold)
    rows = load_jsonl(master_path)
    validate_corpus(rows)
    queries = fact_queries(load_jsonl(gold_path))
    print(f"Loaded {len(rows)} master rows and {len(queries)} fact queries")

    embedding_model = args.embedding_model
    if not embedding_model and args.embedding_backend in ("auto", "api"):
        embedding_model = config.get("EMBEDDING_MODEL")
    embedding_model = embedding_model or DEFAULT_EMBEDDING_MODEL
    embedding_backend = args.embedding_backend
    endpoint = ""
    if embedding_backend == "auto":
        endpoint = config.get("EMBEDDING_BASE_URL") or ""
        api_configured = any(
            (
                endpoint,
                config.get("EMBEDDING_MODEL"),
                config.secret("EMBEDDING_API_KEY"),
                config.get("EMBEDDING_CA_CERT"),
            )
        )
        embedding_backend = "api" if api_configured else "huggingface"
    if embedding_backend == "api":
        endpoint = config.get("EMBEDDING_BASE_URL") or ""
    namespace = hashlib.sha256(
        f"{embedding_backend}:{endpoint}:{embedding_model}".encode("utf-8")
    ).hexdigest()[:12]
    index_root = Path(args.index_dir) / f"{model_slug(embedding_model)}-{namespace}"
    embeddings = make_embeddings(
        embedding_model,
        args.device,
        backend=args.embedding_backend,
    )
    reranker = make_reranker(args.reranker_model, args.device)
    results = []
    for chunk_size in args.chunk_sizes:
        results.extend(
            evaluate_chunk_size(
                chunk_size=chunk_size,
                overlap=args.chunk_overlap,
                rows=rows,
                queries=queries,
                index_root=index_root,
                embeddings=embeddings,
                reranker=reranker,
                candidate_k=args.candidate_k,
                top_k=args.top_k,
                rebuild_index=args.rebuild_index,
            )
        )
    write_results(
        Path(args.out_dir), results, args.chunk_sizes, args.candidate_k, args.top_k
    )


if __name__ == "__main__":
    main()
