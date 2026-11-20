import tempfile
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import List
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
RAG_DIR = ROOT / "scripts" / "rag"
if str(RAG_DIR) not in sys.path:
    sys.path.insert(0, str(RAG_DIR))

from langchain_core.documents import Document  # noqa: E402
from langchain_core.retrievers import BaseRetriever  # noqa: E402

from retrieval import (  # noqa: E402
    bm25_tokens,
    chunk_corpus,
    document_text,
    hit_at_k,
    make_hybrid_retriever,
    OpenAICompatibleEmbeddings,
    make_vector_store,
    parse_context_ids,
    rerank_documents,
    validate_corpus,
)


class StaticRetriever(BaseRetriever):
    documents: List[Document]

    def _get_relevant_documents(self, query, *, run_manager):
        return self.documents


class StaticVectorStore:
    def __init__(self, documents):
        self.documents = documents

    def as_retriever(self, search_kwargs):
        return StaticRetriever(documents=self.documents[: search_kwargs["k"]])


class SectionReranker:
    def predict(self, pairs):
        return [float("section 35" in text.lower()) for _query, text in pairs]


class RetrievalHelpersTests(unittest.TestCase):
    def test_parse_context_ids_accepts_gold_set_formats(self):
        self.assertEqual(parse_context_ids(["CPA-s2(7)", "PR-42"]), {"CPA-s2(7)", "PR-42"})
        self.assertEqual(parse_context_ids('["CPA-s2(7)", "PR-42"]'), {"CPA-s2(7)", "PR-42"})
        self.assertEqual(parse_context_ids("{CPA-s2(7)}"), {"CPA-s2(7)"})
        self.assertEqual(parse_context_ids(None), set())

    def test_document_text_includes_qa_fields_without_mutating_row(self):
        row = {
            "id": "CPA-s1",
            "question": "What does Section 1 provide?",
            "context": "Consumer Protection Act, 2019.",
            "answer": "It is called the Consumer Protection Act, 2019.",
        }
        original = row.copy()

        text = document_text(row)

        self.assertIn("Question: What does Section 1 provide?", text)
        self.assertIn("Context: Consumer Protection Act, 2019.", text)
        self.assertIn("Answer: It is called the Consumer Protection Act, 2019.", text)
        self.assertEqual(row, original)

    def test_bm25_tokenizer_preserves_section_and_case_numbers(self):
        self.assertEqual(
            bm25_tokens("Section 35, Case No. 402!"),
            ["section", "35", "case", "no", "402"],
        )

    def test_corpus_validation_rejects_duplicate_ids_and_empty_fields(self):
        good = {"id": "PR-1", "question": "Q?", "context": "C", "answer": "A"}
        validate_corpus([good])
        with self.assertRaisesRegex(ValueError, "Duplicate corpus id"):
            validate_corpus([good, good.copy()])
        with self.assertRaisesRegex(ValueError, "empty or invalid context"):
            validate_corpus([{**good, "context": ""}])

    def test_hit_at_k_matches_parent_row_not_chunk_id(self):
        returned = [
            (SimpleNamespace(metadata={"qa_id": "PR-99"}), 0.9),
            (SimpleNamespace(metadata={"qa_id": "CPA-s2(7)"}), 0.8),
        ]
        self.assertTrue(hit_at_k(returned, {"CPA-s2(7)"}))
        self.assertFalse(hit_at_k(returned[:1], {"CPA-s2(7)"}))

    def test_chunker_splits_rows_and_retains_parent_ids(self):
        row = {
            "id": "CPA-s35",
            "source": "cpa2019",
            "topic": "statute",
            "act_era": "2019",
            "section": "35",
            "question": "What does Section 35 say?",
            "context": "Section 35 sets the procedure for filing a complaint. " * 120,
            "answer": "A complaint may be filed in the prescribed manner.",
        }

        documents = chunk_corpus([row], chunk_size=100, chunk_overlap=20)

        self.assertGreater(len(documents), 1)
        self.assertTrue(all(document.metadata["qa_id"] == "CPA-s35" for document in documents))
        self.assertEqual(len({document.id for document in documents}), len(documents))

    def test_hybrid_retriever_and_cross_encoder_rerank(self):
        documents = [
            Document(
                id="CPA-s35::chunk:0",
                page_content="Section 35 provides the complaint filing procedure.",
                metadata={"qa_id": "CPA-s35", "section": "35"},
            ),
            Document(
                id="PR-2::chunk:0",
                page_content="Insurance claim and policy dispute.",
                metadata={"qa_id": "PR-2", "topic": "insurance"},
            ),
        ]
        retriever = make_hybrid_retriever(
            documents,
            StaticVectorStore(documents),
            candidate_k=2,
        )

        results = retriever.invoke("Section 35 complaint filing")
        ranked = rerank_documents(
            SectionReranker(),
            "Section 35 complaint filing",
            results,
            top_k=1,
        )

        self.assertEqual(ranked[0][0].metadata["qa_id"], "CPA-s35")

    def test_vector_index_resumes_only_missing_chunks(self):
        documents = [
            Document(id="CPA-s1::chunk:0", page_content="existing text", metadata={"qa_id": "CPA-s1"}),
            Document(id="CPA-s2::chunk:0", page_content="new text", metadata={"qa_id": "CPA-s2"}),
        ]

        class PartialStore:
            def __init__(self):
                self.added_ids = []

            def get(self):
                return {"ids": [documents[0].id], "documents": [documents[0].page_content]}

            def add_documents(self, documents, ids):
                self.added_ids.extend(ids)

        store = PartialStore()
        with tempfile.TemporaryDirectory() as directory:
            with patch("langchain_chroma.Chroma", return_value=store):
                result = make_vector_store(
                    documents,
                    embeddings=object(),
                    persist_directory=Path(directory),
                    collection_name="test",
                )

        self.assertIs(result, store)
        self.assertEqual(store.added_ids, [documents[1].id])

    def test_api_embeddings_batch_and_sort_by_response_index(self):
        fake_client = SimpleNamespace(
            embeddings=SimpleNamespace(
                create=unittest.mock.Mock(
                    side_effect=[
                        SimpleNamespace(
                            data=[
                                SimpleNamespace(index=1, embedding=[0.0, 1.0]),
                                SimpleNamespace(index=0, embedding=[1.0, 0.0]),
                            ]
                        ),
                        SimpleNamespace(data=[SimpleNamespace(index=0, embedding=[3.0, 0.0])]),
                    ]
                )
            )
        )
        with patch("openai.OpenAI", return_value=fake_client):
            embeddings = OpenAICompatibleEmbeddings(
                "https://embedding.example/v1",
                "test-model",
                "test-key",
            )

        self.assertEqual(embeddings.embed_documents(["first", "second"]), [[1.0, 0.0], [0.0, 1.0]])
        self.assertEqual(embeddings.embed_query("query"), [3.0, 0.0])
        self.assertEqual(fake_client.embeddings.create.call_count, 2)

    def test_api_embeddings_splits_long_inputs_under_endpoint_token_limit(self):
        create = unittest.mock.Mock(
            return_value=SimpleNamespace(
                data=[
                    SimpleNamespace(index=0, embedding=[1.0, 0.0]),
                    SimpleNamespace(index=1, embedding=[0.0, 1.0]),
                ]
            )
        )
        fake_client = SimpleNamespace(embeddings=SimpleNamespace(create=create))
        with patch("openai.OpenAI", return_value=fake_client):
            embeddings = OpenAICompatibleEmbeddings(
                "https://embedding.example/v1",
                "test-model",
                "test-key",
            )

        vector = embeddings.embed_documents(["section " * 451])[0]

        self.assertEqual(create.call_count, 1)
        input_batch = create.call_args.kwargs["input"]
        self.assertEqual(len(input_batch), 2)
        input_weights = [len(embeddings.tokenizer.encode(text)) for text in input_batch]
        self.assertTrue(all(weight <= 350 for weight in input_weights))
        norm = sum(weight**2 for weight in input_weights) ** 0.5
        self.assertAlmostEqual(vector[0], input_weights[0] / norm, places=5)
        self.assertAlmostEqual(vector[1], input_weights[1] / norm, places=5)

    def test_api_embeddings_respects_maximum_batch_size(self):
        def response_for_batch(*, model, input):
            self.assertLessEqual(len(input), 16)
            return SimpleNamespace(
                data=[
                    SimpleNamespace(index=index, embedding=[float(index + 1), 1.0])
                    for index in range(len(input))
                ]
            )

        fake_client = SimpleNamespace(
            embeddings=SimpleNamespace(create=unittest.mock.Mock(side_effect=response_for_batch))
        )
        with patch("openai.OpenAI", return_value=fake_client):
            embeddings = OpenAICompatibleEmbeddings(
                "https://embedding.example/v1",
                "test-model",
                "test-key",
            )

        result = embeddings.embed_documents(["short input"] * 17)

        self.assertEqual(len(result), 17)
        self.assertEqual(fake_client.embeddings.create.call_count, 2)


if __name__ == "__main__":
    unittest.main()
