"""Chunking, indexing, and hybrid retrieval for the legal QA corpus."""
from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set

from langchain_core.embeddings import Embeddings


DEFAULT_EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
DEFAULT_RERANKER_MODEL = "BAAI/bge-reranker-base"
DEFAULT_CHUNK_SIZES = (500, 1000)
DEFAULT_CHUNK_OVERLAP = 100
DEFAULT_CANDIDATE_K = 10
DEFAULT_TOP_K = 3


class OpenAICompatibleEmbeddings(Embeddings):
    """LangChain embeddings adapter for an OpenAI-compatible embedding endpoint."""

    def __init__(
        self,
        base_url: str,
        model_name: str,
        api_key: str,
        ca_cert: Optional[str] = None,
    ) -> None:
        import httpx
        from openai import OpenAI

        if not base_url or not model_name or not api_key:
            raise ValueError(
                "The embedding API requires EMBEDDING_BASE_URL, EMBEDDING_MODEL, "
                "and EMBEDDING_API_KEY (or EMBEDDING_API_KEY_FILE)."
            )
        verify: Any = ca_cert if ca_cert else True
        if ca_cert and not Path(ca_cert).expanduser().is_file():
            raise FileNotFoundError(f"Embedding CA certificate not found: {ca_cert}")
        self.model_name = model_name
        self.max_batch_size = 16
        self.max_input_tokens = 350
        import tiktoken

        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key,
            http_client=httpx.Client(verify=verify, timeout=120.0),
        )

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        chunks: List[str] = []
        chunk_owners: List[int] = []
        chunk_weights: List[int] = []
        for owner, text in enumerate(texts):
            tokens = self.tokenizer.encode(text)
            for start in range(0, len(tokens), self.max_input_tokens):
                token_chunk = tokens[start : start + self.max_input_tokens]
                chunks.append(self.tokenizer.decode(token_chunk))
                chunk_owners.append(owner)
                chunk_weights.append(len(token_chunk))

        vectors: List[List[float]] = []
        for start in range(0, len(chunks), self.max_batch_size):
            batch = chunks[start : start + self.max_batch_size]
            response = self.client.embeddings.create(
                model=self.model_name,
                input=batch,
            )
            data = sorted(response.data, key=lambda item: item.index)
            if len(data) != len(batch):
                raise RuntimeError(
                    f"Embedding endpoint returned {len(data)} vectors for "
                    f"{len(batch)} input texts"
                )
            vectors.extend(item.embedding for item in data)

        pooled: List[List[float]] = [[] for _ in texts]
        total_weights = [0 for _ in texts]
        for owner, weight, vector in zip(chunk_owners, chunk_weights, vectors):
            if not pooled[owner]:
                pooled[owner] = [0.0] * len(vector)
            if len(pooled[owner]) != len(vector):
                raise RuntimeError("Embedding endpoint returned inconsistent vector dimensions")
            for index, value in enumerate(vector):
                pooled[owner][index] += float(value) * weight
            total_weights[owner] += weight

        embeddings: List[List[float]] = []
        for vector, weight in zip(pooled, total_weights):
            if not vector or not weight:
                raise RuntimeError("Embedding endpoint returned no vector for an input text")
            averaged = [value / weight for value in vector]
            norm = math.sqrt(sum(value * value for value in averaged))
            if norm:
                averaged = [value / norm for value in averaged]
            embeddings.append(averaged)
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        response = self.client.embeddings.create(
            model=self.model_name,
            input=[text],
        )
        if len(response.data) != 1:
            raise RuntimeError(
                f"Embedding endpoint returned {len(response.data)} vectors for one query"
            )
        return response.data[0].embedding


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Load JSONL records with line-numbered errors instead of skipping bad rows."""
    records: List[Dict[str, Any]] = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
    return records


def parse_context_ids(value: Any) -> Set[str]:
    """Normalize gold_context_ids across list, scalar, and serialized-list formats."""
    if value is None:
        return set()
    if isinstance(value, dict):
        values: Iterable[Any] = value.keys()
    elif isinstance(value, (list, tuple, set)):
        values = value
    elif isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return set()
        try:
            decoded = json.loads(stripped)
        except json.JSONDecodeError:
            decoded = None
        if isinstance(decoded, (list, tuple, set)):
            values = decoded
        elif isinstance(decoded, str):
            values = [decoded]
        else:
            values = re.split(r"[\s,;|]+", stripped.strip("{}[] "))
    else:
        values = [value]
    return {str(item).strip().strip("'\"") for item in values if str(item).strip().strip("'\"")}


def document_text(row: Dict[str, Any]) -> str:
    """Create an indexed view of a QA row without changing its source fields."""
    parts = []
    for label, field in (("Question", "question"), ("Context", "context"), ("Answer", "answer")):
        value = row.get(field)
        if value is not None and str(value).strip():
            parts.append(f"{label}: {str(value).strip()}")
    text = "\n\n".join(parts)
    if not text:
        raise ValueError(f"Corpus row {row.get('id', '<unknown>')} has no indexable text")
    return text


def bm25_tokens(text: str) -> List[str]:
    """Keep legal section/case numbers while normalizing punctuation and case."""
    return re.findall(r"[a-z0-9]+", text.casefold())


def validate_corpus(rows: Sequence[Dict[str, Any]]) -> None:
    """Fail early if corpus rows cannot be reliably cited or evaluated."""
    if not rows:
        raise ValueError("The master corpus is empty")
    seen: Set[str] = set()
    for position, row in enumerate(rows, start=1):
        row_id = row.get("id")
        if not isinstance(row_id, str) or not row_id.strip():
            raise ValueError(f"Corpus row {position} has no non-empty string id")
        if row_id in seen:
            raise ValueError(f"Duplicate corpus id: {row_id}")
        seen.add(row_id)
        for field in ("question", "context", "answer"):
            if not isinstance(row.get(field), str) or not row[field].strip():
                raise ValueError(f"Corpus row {row_id} has an empty or invalid {field}")


def chunk_corpus(
    rows: Sequence[Dict[str, Any]],
    chunk_size: int,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
):
    """Split QA rows into token-sized LangChain Documents with source IDs attached."""
    if chunk_size <= 0 or chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_size must be positive and chunk_overlap in [0, chunk_size)")

    from langchain_core.documents import Document
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    documents = []
    for row in rows:
        row_id = row["id"]
        chunks = splitter.split_text(document_text(row))
        for chunk_index, chunk in enumerate(chunks):
            metadata = {
                "qa_id": row_id,
                "chunk_index": chunk_index,
                "source": str(row.get("source") or ""),
                "topic": str(row.get("topic") or ""),
                "act_era": str(row.get("act_era") or ""),
                "section": str(row.get("section") or ""),
            }
            documents.append(
                Document(
                    page_content=chunk,
                    metadata=metadata,
                    id=f"{row_id}::chunk:{chunk_index}",
                )
            )
    return documents


def make_embeddings(
    model_name: Optional[str],
    device: Optional[str] = None,
    backend: str = "auto",
):
    """Create API or normalized Hugging Face sentence embeddings for Chroma."""
    from scripts.eval import config

    base_url = config.get("EMBEDDING_BASE_URL")
    api_model = model_name or config.get("EMBEDDING_MODEL")
    api_key = config.secret("EMBEDDING_API_KEY")
    cert = config.get("EMBEDDING_CA_CERT")
    api_configured = any((base_url, api_model, api_key, cert))

    if backend not in {"auto", "api", "huggingface"}:
        raise ValueError("embedding backend must be 'auto', 'api', or 'huggingface'")
    selected_backend = "api" if backend == "auto" and api_configured else backend
    if selected_backend == "auto":
        selected_backend = "huggingface"
    if selected_backend == "api":
        if not base_url or not api_model or not api_key:
            raise ValueError(
                "Configure EMBEDDING_BASE_URL, EMBEDDING_MODEL, and "
                "EMBEDDING_API_KEY (or EMBEDDING_API_KEY_FILE) in the project-root .env."
            )
        return OpenAICompatibleEmbeddings(
            base_url=base_url,
            model_name=api_model,
            api_key=api_key,
            ca_cert=cert,
        )

    from langchain_huggingface import HuggingFaceEmbeddings

    model_name = model_name or DEFAULT_EMBEDDING_MODEL
    model_kwargs = {"device": device} if device else {}
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs={
            "normalize_embeddings": True,
            "batch_size": 64,
            "show_progress_bar": True,
        },
    )


def make_vector_store(
    documents,
    embeddings,
    persist_directory: Path,
    collection_name: str,
    rebuild: bool = False,
):
    """Create or reuse a persistent Chroma index for one chunking experiment."""
    from langchain_chroma import Chroma

    persist_directory.mkdir(parents=True, exist_ok=True)
    store = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=str(persist_directory),
    )
    existing = store.get()
    existing_ids = existing["ids"]
    expected_ids = [document.id for document in documents]
    expected_text = {document.id: document.page_content for document in documents}
    existing_text = dict(zip(existing_ids, existing.get("documents") or []))
    unexpected_ids = set(existing_ids) - set(expected_ids)
    conflicting_ids = {
        row_id
        for row_id in set(existing_ids) & set(expected_ids)
        if existing_text.get(row_id) != expected_text[row_id]
    }
    if (unexpected_ids or conflicting_ids) and not rebuild:
        raise RuntimeError(
            f"Chroma collection {collection_name!r} contains a different corpus. "
            "Re-run with --rebuild-index to replace only this collection."
        )
    if unexpected_ids or conflicting_ids:
        store.delete(ids=existing_ids)
        existing_ids = []
    if set(existing_ids) == set(expected_ids):
        return store
    present_ids = set(existing_ids)
    missing_documents = [
        document for document in documents if document.id not in present_ids
    ]
    batch_size = 500
    for start in range(0, len(missing_documents), batch_size):
        batch = missing_documents[start : start + batch_size]
        store.add_documents(
            documents=batch,
            ids=[document.id for document in batch],
        )
    return store


def make_hybrid_retriever(documents, vector_store, candidate_k: int = DEFAULT_CANDIDATE_K):
    """Combine Chroma dense search and BM25 with equal-weight rank fusion."""
    from langchain_classic.retrievers import EnsembleRetriever
    from langchain_community.retrievers import BM25Retriever

    if candidate_k <= 0:
        raise ValueError("candidate_k must be positive")
    dense = vector_store.as_retriever(search_kwargs={"k": candidate_k})
    sparse = BM25Retriever.from_texts(
        texts=[document.page_content for document in documents],
        metadatas=[document.metadata for document in documents],
        k=candidate_k,
        preprocess_func=bm25_tokens,
    )
    return EnsembleRetriever(retrievers=[dense, sparse], weights=[0.5, 0.5])


def make_reranker(model_name: str, device: Optional[str] = None):
    """Load a Hugging Face cross-encoder for query/document relevance scoring."""
    from sentence_transformers import CrossEncoder

    return CrossEncoder(model_name, device=device)


def rerank_documents(reranker, query: str, documents, top_k: int = DEFAULT_TOP_K):
    """Score the hybrid candidates with a cross-encoder and return the top results."""
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if not documents:
        return []
    pairs = [(query, document.page_content) for document in documents]
    scores = reranker.predict(pairs)
    if len(scores) != len(documents):
        raise RuntimeError(
            f"Reranker returned {len(scores)} scores for {len(documents)} documents"
        )
    ranked = sorted(
        zip(documents, scores),
        key=lambda item: float(item[1]),
        reverse=True,
    )
    return [(document, float(score)) for document, score in ranked[:top_k]]


def hit_at_k(ranked_documents, gold_ids: Set[str]) -> bool:
    """A hit occurs when any returned chunk belongs to a gold master row."""
    return any(document.metadata.get("qa_id") in gold_ids for document, _ in ranked_documents)
