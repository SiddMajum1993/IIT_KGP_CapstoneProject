# Recall@3 - retriever only (api mode)

| Chunk size (BERT tokens) | Method | Fact questions | All questions with a gold passage |
|---|---|---|---|
| 256 | bm25 | 23/30 (77%) | 26/42 (62%) |
| 256 | dense | 22/30 (73%) | 25/42 (60%) |
| 256 | hybrid | 25/30 (83%) | 29/42 (69%) |
| 256 | rerank | 28/30 (93%) | 34/42 (81%) |
| 500 | bm25 | 24/30 (80%) | 27/42 (64%) |
| 500 | dense | 20/30 (67%) | 25/42 (60%) |
| 500 | hybrid | 26/30 (87%) | 30/42 (71%) |
| 500 | rerank | 27/30 (90%) | 33/42 (79%) |
