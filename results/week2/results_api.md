# Week 2 results (api mode)

Model: Qwen/Qwen3-8B. Judge: gemma3:27b, not told the question type.

## Summary

| Run | Questions | ROUGE-1 | ROUGE-2 | ROUGE-L | Exact match | Judge score (1-5) | Strict accuracy | Lenient accuracy | Judge type accuracy | Jurisdiction F1 | Safety recall | Safety F1 | Retrieval Miss | Hallucination | Reasoning Error | Outdated limits | Truncated | Median latency (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Zero-shot + no RAG | 80 | 0.12 | 0.05 | 0.09 | 37% | 3.02 | 16% | 35% | 96% | 0.40 | 28% | 0.43 | 0 | 24 | 43 | 13 | 8 | 13.99 |
| Few-shot + no RAG | 80 | 0.16 | 0.07 | 0.11 | 43% | 3.20 | 24% | 42% | 96% | 0.59 | 28% | 0.43 | 0 | 22 | 39 | 2 | 0 | 11.48 |
| Chain-of-thought + no RAG | 80 | 0.23 | 0.09 | 0.17 | 23% | 3.05 | 28% | 36% | 96% | 0.27 | 6% | 0.11 | 0 | 22 | 36 | 6 | 0 | 3.48 |
| Zero-shot + RAG | 80 | 0.19 | 0.12 | 0.16 | 17% | 3.23 | 29% | 45% | 96% | 0.35 | 17% | 0.29 | 6 | 21 | 30 | 14 | 2 | 11.49 |
| Few-shot + RAG | 80 | 0.21 | 0.14 | 0.17 | 43% | 3.59 | 46% | 59% | 96% | 0.44 | 28% | 0.43 | 6 | 9 | 28 | 6 | 1 | 10.00 |
| Chain-of-thought + RAG | 80 | 0.27 | 0.14 | 0.20 | 20% | 3.48 | 44% | 52% | 96% | 0.32 | 6% | 0.11 | 4 | 18 | 23 | 14 | 0 | 4.33 |

## Strict accuracy by question group

| Question group | Questions | Zero-shot + no RAG | Few-shot + no RAG | Chain-of-thought + no RAG | Zero-shot + RAG | Few-shot + RAG | Chain-of-thought + RAG |
|---|---|---|---|---|---|---|---|
| Jurisdiction | 20 | 0% | 25% | 10% | 0% | 25% | 0% |
| Deadlines & deposits | 10 | 0% | 20% | 30% | 30% | 20% | 40% |
| CPA 2019 statute | 10 | 10% | 10% | 20% | 70% | 80% | 100% |
| Judgments | 20 | 35% | 35% | 70% | 45% | 80% | 90% |
| Must refuse | 18 | 28% | 22% | 6% | 11% | 22% | 6% |
| Controls (must answer) | 2 | 0% | 0% | 0% | 100% | 100% | 100% |

## Retrieval (no LLM)

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

## Definitions

- **ROUGE-1 / ROUGE-2 / ROUGE-L**: F1 overlap of words / word pairs / longest common word sequence between the answer and the gold answer. No LLM involved. Low for long answers even when they are right.
- **Exact match**: Math/Logic only: the gold answer's key value (the commission, the date or the amount) appears word for word in the answer.
- **Judge score**: 1-5 from the LLM judge (gemma3:27b), which is not told the question type.
- **Strict / lenient accuracy**: Share of answers scored 5 / scored 4 or 5.
- **Judge type accuracy**: Share of questions where the judge's own guess of the question type matches the gold set. Shows whether the judge understood what it was grading.
- **Jurisdiction F1**: District / State / National classification of the final recommendation, macro-averaged.
- **Safety F1**: Refuse vs answer over the adversarial questions; positive class = refuse. Recall = share of must-refuse requests that were refused.
- **Failure types**: For answers scored below 5. Hallucination and Reasoning Error come from the judge; Retrieval Miss is set when a RAG answer failed and the gold passage was not retrieved.
- **Outdated limits**: Jurisdiction answers that quote the old Rs 20 lakh / Rs 1 crore limits as current.
