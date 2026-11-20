# Week 2 Hybrid RAG: Recall@3

Evaluation uses the fact questions only. A question is a hit when any of its gold master row IDs appears among the three cross-encoder-reranked hybrid results.

| Chunk size (tokens) | Overlap (tokens) | Hits | Questions | Recall@3 |
|---:|---:|---:|---:|---:|
| 500 | 100 | 30 | 30 | 100.0% |
| 1000 | 100 | 30 | 30 | 100.0% |

The dense and BM25 retrievers contribute equal weight (0.5 each) to LangChain's ensemble rank fusion. The cross-encoder reranks the top 10 hybrid candidates and returns the top 3.
