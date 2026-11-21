"""The retriever: chunk the corpus, embed it into Chroma, search with BM25 + dense, rerank to the top 3.

  1. Chunk   - only the "context" field of each master-set row is indexed (the passage from the judgment or the
               Act). Questions and answers are not indexed: the gold questions come from the same rows, so
               indexing them would let the retriever find a gold row by matching its question word for word.
               Chunk length is counted in BERT tokens, the tokenizer bge-small uses.
  2. Embed   - local mode: all-mpnet-base-v2 on this machine; api mode: bge-small-en-v1.5 on the course API.
               The vectors are stored in Chroma on disk (chroma_db/), so they are computed only once.
  3. Search  - BM25 (exact words, e.g. "Section 35") and dense (meaning) each return 10 chunks; LangChain's
               EnsembleRetriever merges the two lists with equal weight.
  4. Rerank  - a cross-encoder (ms-marco-MiniLM-L-6-v2) reads each (question, chunk) pair and keeps the best 3.

Usage: python -m copilot.rag --mode local    builds the index (if needed) and answers one test query.
"""
import argparse
import json
import os
import re
import time

import httpx
from langchain_chroma import Chroma
from langchain_classic.retrievers import ContextualCompressionRetriever, EnsembleRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI
from transformers import AutoTokenizer

from copilot import config

CHUNK_SIZE = 500     # BERT tokens; the retrieval evaluation compares 256 and 500
CHUNK_OVERLAP = 100
CANDIDATES = 10      # chunks each of BM25 and dense return before reranking
TOP_K = 3            # chunks passed to the model
RERANKER = "cross-encoder/ms-marco-MiniLM-L-6-v2"
METHODS = ["bm25", "dense", "hybrid", "rerank"]


def bm25_words(text: str) -> list:
    """BM25 tokens: lower-case letters and digits, so "Section 35." and "section 35" match."""
    return re.findall(r"[a-z0-9]+", text.lower())


class APIEmbeddings(Embeddings):
    """bge-small-en-v1.5 on the course embedding API (OpenAI format, at most 16 texts per request)."""

    def __init__(self):
        base_url, key = config.get("EMBEDDING_BASE_URL"), config.secret("EMBEDDING_API_KEY")
        assert base_url and key, "set EMBEDDING_BASE_URL and EMBEDDING_API_KEY (or _FILE) in .env"
        cert = config.cert("EMBEDDING_CA_CERT")
        self.client = OpenAI(base_url=base_url, api_key=key,
                             http_client=httpx.Client(verify=cert, timeout=120) if cert else None)
        self.model = config.get("EMBEDDING_MODEL")

    def _embed(self, batch: list) -> list:
        for delay in [0, 2, 4, 8, 16, 32]:  # retry on rate limits and dropped connections
            time.sleep(delay)
            try:
                r = self.client.embeddings.create(model=self.model, input=batch)
                return [d.embedding for d in sorted(r.data, key=lambda d: d.index)]
            except Exception as e:  # noqa: BLE001 - any API failure is retried, the last one is raised
                error = e
        raise error

    def embed_documents(self, texts: list) -> list:
        vectors = []
        for start in range(0, len(texts), 16):
            vectors += self._embed(texts[start:start + 16])
            print(f"  embedded {min(start + 16, len(texts))}/{len(texts)}", end="\r", flush=True)
        return vectors

    def embed_query(self, text: str) -> list:
        return self._embed([text])[0]


def make_embeddings(mode: str) -> Embeddings:
    if mode == "api":
        return APIEmbeddings()
    return HuggingFaceEmbeddings(model_name=config.get("LOCAL_EMBEDDING_MODEL"),
                                 encode_kwargs={"normalize_embeddings": True})


def chunk_corpus(chunk_size: int) -> list:
    """Split every row's context into chunks. Each chunk remembers which master-set row it came from (qa_id)."""
    rows = [json.loads(line) for line in open(config.CORPUS_PATH, encoding="utf-8")]
    splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        AutoTokenizer.from_pretrained("bert-base-uncased"), chunk_size=chunk_size, chunk_overlap=CHUNK_OVERLAP)
    chunks = []
    for row in rows:
        for i, text in enumerate(splitter.split_text(row["context"])):
            chunks.append(Document(page_content=text, id=f"{row['id']}::{i}",
                                   metadata=dict(qa_id=row["id"], source=row["source"], topic=row["topic"])))
    return chunks


def build_vector_store(mode: str, chunks: list, chunk_size: int) -> Chroma:
    """Open the Chroma collection on disk and embed any chunks it does not have yet (so a broken build resumes)."""
    folder = os.path.join(config.CHROMA_DIR, f"{mode}_{chunk_size}")
    store = Chroma(collection_name="consumer_law", embedding_function=make_embeddings(mode), persist_directory=folder,
                   collection_metadata={"hnsw:space": "cosine"})
    have = set(store.get(include=[])["ids"])
    missing = [c for c in chunks if c.id not in have]
    if missing:
        print(f"embedding {len(missing)} of {len(chunks)} chunks into {folder}")
        for start in range(0, len(missing), 256):  # save in batches so progress survives an interruption
            batch = missing[start:start + 256]
            store.add_documents(batch, ids=[c.id for c in batch])
        print()
    return store


class Retriever:
    """Holds the four retrieval methods compared in the evaluation. search() uses the final one (rerank)."""

    def __init__(self, mode: str, chunk_size: int = CHUNK_SIZE):
        chunks = chunk_corpus(chunk_size)
        store = build_vector_store(mode, chunks, chunk_size)
        bm25 = BM25Retriever.from_documents(chunks, k=CANDIDATES, preprocess_func=bm25_words)
        dense = store.as_retriever(search_kwargs={"k": CANDIDATES})
        hybrid = EnsembleRetriever(retrievers=[bm25, dense], weights=[0.5, 0.5])
        reranker = CrossEncoderReranker(model=HuggingFaceCrossEncoder(model_name=RERANKER), top_n=TOP_K)
        self.methods = dict(bm25=bm25, dense=dense, hybrid=hybrid,
                            rerank=ContextualCompressionRetriever(base_compressor=reranker, base_retriever=hybrid))

    def search(self, question: str, method: str = "rerank") -> list:
        return self.methods[method].invoke(question)[:TOP_K]


def load_retriever(mode: str, chunk_size: int = CHUNK_SIZE) -> Retriever:
    return Retriever(mode, chunk_size)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["local", "api"], default=config.get("COPILOT_MODE"))
    ap.add_argument("--chunk-size", type=int, default=CHUNK_SIZE)
    ap.add_argument("--query", default="Can a complaint be filed electronically under Section 35?")
    args = ap.parse_args()
    retriever = load_retriever(args.mode, args.chunk_size)
    for d in retriever.search(args.query):
        print(d.metadata["qa_id"], "|", d.page_content[:150].replace("\n", " "))


if __name__ == "__main__":
    main()
