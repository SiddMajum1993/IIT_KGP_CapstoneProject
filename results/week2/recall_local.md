# Recall@3 - retriever only (local mode)

| Chunk size (BERT tokens) | Method | Fact questions | All questions with a gold passage |
|---|---|---|---|
| 256 | bm25 | 23/30 (77%) | 26/42 (62%) |
| 256 | dense | 16/30 (53%) | 20/42 (48%) |
| 256 | hybrid | 20/30 (67%) | 24/42 (57%) |
| 256 | rerank | 28/30 (93%) | 35/42 (83%) |
| 500 | bm25 | 24/30 (80%) | 27/42 (64%) |
| 500 | dense | 19/30 (63%) | 26/42 (62%) |
| 500 | hybrid | 24/30 (80%) | 29/42 (69%) |
| 500 | rerank | 27/30 (90%) | 34/42 (81%) |
