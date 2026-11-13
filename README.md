# Indian Consumer Rights Copilot

IIT KGP Executive PG Programme (GenAI & Agentic AI) capstone, Track B.

The Copilot is an agentic RAG assistant that answers procedural questions about Indian consumer law under the Consumer Protection Act, 2019. It runs on a small LLM (≤8B parameters, e.g. `qwen2.5:7b` or `llama3.1:8b`). Deterministic work such as pecuniary jurisdiction and deadline arithmetic is done in Python tools, and facts are grounded through retrieval. The full design is in [`Product Requirements Document (PRD)_ Indian Consumer Rights Copilot.md`](<Product Requirements Document (PRD)_ Indian Consumer Rights Copilot.md>).

> **Disclaimer:** this is a procedural advisor only. A qualified lawyer must review any legal action.

---

## Repository layout

```
.
├── Product Requirements Document (PRD)_ Indian Consumer Rights Copilot.md
├── scripts/
│   └── data_prep/
│       ├── 01_extract_candidates.py   # scan raw Prarabdha shards -> consumer-law candidates
│       ├── 02_build_master.py         # candidates + CPA 2019 -> 2,000-row master QA set
│       └── 03_build_gold.py           # master -> 60-100 row gold evaluation set
└── data/                              # NOT in git (see .gitignore); rebuild with the steps below
    ├── raw/
    │   ├── prarabdha_indian_legal_sft/   # 33 parquet shards, ~8.5 GB
    │   └── cpa_2019/                     # Consumer Protection Act, 2019 (JSON + PDF)
    ├── interim/
    │   └── consumer_candidates.parquet
    └── processed/
        ├── master_qa.jsonl
        ├── gold_qa.jsonl
        └── gold_qa.csv
```

`data/processed/` (the 2,000-row master set and the 80-row gold set) is committed, so you can start on retrieval and evaluation straight after cloning. `data/raw/` and `data/interim/` are gitignored because they are large and can be regenerated. Follow the steps below to rebuild everything from the sources, for example to change a filter or check that a row is verbatim.

---

## 1. Prerequisites

| Requirement | Notes |
|---|---|
| Python 3.9+ | Developed on 3.9.6 |
| `pandas`, `pyarrow` | Tested with pandas 2.3.3 and pyarrow 21.0.0 |
| `curl` | Used for the downloads |
| ~10 GB free disk | 8.5 GB for raw shards, plus a few hundred MB for derived files |
| Network access to `huggingface.co` | **No account or token is needed**, since both datasets are public |

Set up a virtual environment:

```bash
git clone <repo-url> IIT_KGP_CapstoneProject
cd IIT_KGP_CapstoneProject
python3 -m venv .venv
source .venv/bin/activate
pip install "pandas>=2.0" "pyarrow>=14"
```

> All commands below are run from the **repo root**, because the scripts' default paths are relative to it.

---

## 2. Source datasets

We treat both sources as ground truth. The pipeline **selects and reformats** rows. It does not rewrite their content (see [§5](#5-what-is-source-text-vs-what-is-authored)).

| Local dir | Hugging Face dataset | Pinned revision | License | Size |
|---|---|---|---|---|
| `data/raw/prarabdha_indian_legal_sft/` | [`Prarabdha/indian-legal-supervised-fine-tuning-data`](https://huggingface.co/datasets/Prarabdha/indian-legal-supervised-fine-tuning-data) | `05e30cb9536ae6f0d7570c490eb95d73a5117552` | Apache-2.0 | 33 shards, 6,055,371 rows, ~8.5 GB |
| `data/raw/cpa_2019/` | [`GSMS-B/indian-law-the-consumer-protection-act-2019`](https://huggingface.co/datasets/GSMS-B/indian-law-the-consumer-protection-act-2019) | `b671f64b9cd12c94cf497b97b78c5f2045e171b4` | CC-BY-4.0 | 107 sections, ~780 KB |

- **Prarabdha** is a multilingual Indian legal SFT corpus with `context`, `question` and `response` columns. Most of it is not consumer law, so stage 1 filters it.
- **CPA 2019** is the text of the Consumer Protection Act, 2019 as structured JSON (`sections[]` with `section_number`, `section_title`, `chapter_title` and `text`) plus the Gazette PDF.

Pin both downloads to these revisions. Otherwise the row indices (`source_row`, and therefore the `PR-<n>` ids) may not match ours.

---

## 3. Download the raw data

### 3a. Consumer Protection Act, 2019 (small)

```bash
mkdir -p data/raw/cpa_2019
SHA=b671f64b9cd12c94cf497b97b78c5f2045e171b4
BASE=https://huggingface.co/datasets/GSMS-B/indian-law-the-consumer-protection-act-2019/resolve/$SHA
for f in README.md the_consumer_protection_act_2019.json the_consumer_protection_act_2019.pdf; do
  curl -sSfL --retry 5 -o "data/raw/cpa_2019/$f" "$BASE/$f"
done
```

Expected checksums (`shasum -a 256 data/raw/cpa_2019/*`):

```
b72c932f797b1274414fdcf6fdfc5c115a2f5b8702841d05895947d6b85507cf  the_consumer_protection_act_2019.json
4bbed366bc2206bb1c3b1817c5ccdd652d1e66d9476c6779cf5239e71e3ba2a7  the_consumer_protection_act_2019.pdf
```

### 3b. Prarabdha legal SFT corpus (~8.5 GB)

The Hub serves large files through a CDN that sometimes drops connections. The loop below resumes partial files (`-C -`) and retries, two shards at a time. It is safe to re-run until every shard prints `ok`.

```bash
mkdir -p data/raw/prarabdha_indian_legal_sft
cd data/raw/prarabdha_indian_legal_sft
SHA=05e30cb9536ae6f0d7570c490eb95d73a5117552
BASE=https://huggingface.co/datasets/Prarabdha/indian-legal-supervised-fine-tuning-data/resolve/$SHA
curl -sSfL -o README.md "$BASE/README.md"
export BASE
seq -f "%05g" 0 32 | xargs -n 1 -P 2 sh -c '
  f=train-$1-of-00033.parquet
  curl -sSfL --retry 10 --retry-all-errors --retry-delay 5 -C - -o "$f" "$BASE/data/$f" && echo "ok $1" || echo "FAIL $1"
' _
cd -
```

> If `curl -C -` reports `HTTP 416` for a shard, that shard is already complete.

**Verify the download.** This checks that every shard's size matches the Hub and that every file is readable parquet:

```bash
python3 - data/raw/prarabdha_indian_legal_sft <<'EOF'
import sys, json, os, urllib.request
import pyarrow.parquet as pq
d = sys.argv[1]
url = ("https://huggingface.co/api/datasets/Prarabdha/indian-legal-supervised-fine-tuning-data/tree/"
       "05e30cb9536ae6f0d7570c490eb95d73a5117552/data")
bad, rows = [], 0
for f in json.load(urllib.request.urlopen(url)):
    p = os.path.join(d, os.path.basename(f["path"]))
    if not os.path.exists(p) or os.path.getsize(p) != f["size"]:
        bad.append(p); continue
    try:
        rows += pq.ParquetFile(p).metadata.num_rows
    except Exception as e:
        bad.append(f"{p}: {e}")
print("bad:", bad or "none", "| total rows:", rows)
EOF
```

Expected output: `bad: none | total rows: 6055371`. Re-run the download loop for any file listed as bad.

---

## 4. Build the datasets

Run the three stages in order. Each one is deterministic, so the same inputs always produce the same outputs.

```bash
python3 scripts/data_prep/01_extract_candidates.py   # ~3 min
python3 scripts/data_prep/02_build_master.py         # ~1 min
python3 scripts/data_prep/03_build_gold.py           # seconds
```

Every script takes `--help` and accepts path overrides.

### Stage 1: extract consumer-law candidates

`01_extract_candidates.py` streams all 33 shards in 50k-row batches, so it never loads the full 16 GB into memory.

1. **Arrow prefilter.** A fast, vectorised substring match on lowercased `context`/`question`/`response` for any of: `consumer`, `ncdrc`, `deficiency in service`, `unfair trade practice`. This step brings the run down from hours to about 5 s per shard.
2. **Regex classifier.** A row is kept if it matches a *strong* signal: Consumer Protection Act, NCDRC, consumer disputes redressal, consumer forum/commission, deficiency in service, unfair or restrictive trade practice, product liability, or Central Consumer Protection Authority. A row is also kept if it matches a *weak* signal (district forum, state/national commission, opposite party, complainant) and also contains the word "consumer".
3. Each kept row records its **global row index** (`source_row`, counted across shards in sorted order) and its `shard`, so it can be traced back to the source.

**Output:** `data/interim/consumer_candidates.parquet` (27,399 rows. Columns: `context, question, response, source_row, shard`).

### Stage 2: build the master QA set (2,000 rows)

`02_build_master.py` combines the two sources into one `question / context / answer` table. This table is the RAG corpus.

**a) CPA 2019 rows (169):**
- One row per section. Section 2 (Definitions) is split into one row per definition, e.g. `CPA-s2(7)` for "consumer".
- Sections longer than 2,500 characters (about the size of a Prarabdha context) are split at clause markers (`(1)`, `(2)`, …) into chunks with ids such as `CPA-s38-2`.
- `context` is the section text, prefixed with `Consumer Protection Act, 2019 - Section N (Title):`.
- `answer` is the opening sentences of that same text, taken verbatim.
- `question` is a fixed template, e.g. *"What does Section 35 of the Consumer Protection Act, 2019 (Manner in which complaint shall be made) provide?"* or *"How does the Consumer Protection Act, 2019 define "unfair contract"?"*.
- Sections 34, 47 and 58 print the pecuniary limits from the Act. These were revised by the 2021 Jurisdiction Rules, so those rows carry a `note` saying so (see [Pecuniary jurisdiction](#pecuniary-jurisdiction-2021-rules)).

**b) Prarabdha rows (1,831):** chosen from the stage-1 candidates in three steps.

1. **Hard filters:**
   - the question is 30–300 characters and the answer is 40–1,200 characters;
   - the context is at least 97% ASCII, which keeps English only;
   - the question is not identical to the answer;
   - the answer is not a placeholder ("not specified", "[...]");
   - the question is not an instruction ("draft…", "could you find…");
   - the **context itself** must match a strong consumer-law signal;
   - rows whose Q/A states a pecuniary limit in lakh or crore are dropped, because almost all of them are 1986-era and now wrong.
2. **Grounding check:** at least 60% of the answer's content words must appear in the context. This is a cheap guard against hallucinated answers in the source.
3. **Ranking and diversity:**
   - Rows are ranked by a quality score. The score is built from grounding, consumer-law relevance, whether the digits are intact (the source sometimes scrubbed numbers), whether the question ends in "?", and whether the answer length is reasonable.
   - Duplicates are removed by context, by the first 300 characters of the context (same judgment page) and by question.
   - No single topic may exceed **25%** of the set. Topics include insurance, banking, housing, medical, telecom and others.

**Tags added for the RAG and evaluation layers:**
- `act_era`: `1986` / `2019` / `both` / `unknown`. Most judgments predate the 2019 Act.
- `needs_review = true` (44 rows): 1986-era rows that quote a time limit, which may have changed under the 2019 Act.
- `section`: section numbers referenced in the context.
- `topic`, `quality_score`, `grounding`, `digits_intact`.

**Output:** `data/processed/master_qa.jsonl`. Each line has the fields `id, source, source_row, section, chapter, question, context, answer, act_era, topic, quality_score, grounding, digits_intact, needs_review, note`. Ids are `CPA-…` for statute rows and `PR-<source_row>` for Prarabdha rows.

### Stage 3: build the gold evaluation set (60–100 rows)

`03_build_gold.py` produces the test set used for the bare-model baseline (Week 1) and for the RAG and agent runs that follow. It has three categories:

| Category | Rows | What it tests | Where the answer comes from |
|---|---|---|---|
| `math_logic` / `jurisdiction` | 20 | Pecuniary jurisdiction, incl. boundary values (₹50 lakh, ₹2 crore) and traps (1986 limits, the Act's printed limits, market value vs consideration paid) | Computed in Python by `commission_for()` with the 2021 limits. No context; `expected_behavior` is `tool:Calculate_Jurisdiction` |
| `math_logic` / `deadline_or_amount` | 10 | Appeal deadlines, limitation period, pre-deposit amounts | Date and percentage arithmetic on the cited CPA section, whose text is the context |
| `fact` | 30 | Questions about the Act (10) and judgments (20) across topics | **Copied verbatim from master rows** (question, context and answer of the same row). `gold_context_ids` names that row, so Recall@3 can be scored on the retriever alone without calling the LLM |
| `adversarial` | 20 | Drafting notices, predicting outcomes, guaranteed advice, prompt injection, illegal requests (18 refusals) plus 2 control questions that must be answered | Expected behaviour (`refuse` / `answer`) |

**Output:** `data/processed/gold_qa.jsonl` and `gold_qa.csv` (for the shared sheet). Each row has the fields `id (G###), category, subtype, origin, question, context, answer, gold_context_ids, expected_behavior, note`. `origin` is `source` (verbatim from the datasets, all 30 `fact` rows) or `authored` (the other 50).

### Sanity checks

```bash
wc -l data/processed/master_qa.jsonl     # 2000
wc -l data/processed/gold_qa.jsonl       # 80 (30 math_logic, 30 fact, 20 adversarial)
python3 -c "
import json, collections
m = [json.loads(l) for l in open('data/processed/master_qa.jsonl')]
print(collections.Counter(r['source'] for r in m))        # cpa2019: 169, prarabdha: 1831
print(sum(r['needs_review'] for r in m), 'flagged')       # 44
"
```

---

## 5. What is source text vs. what is authored

The two Hugging Face datasets are treated as the source of truth.

- **Master set:**
  - Every `context` and `answer` is source text. The only changes are whitespace and punctuation normalisation: collapsed spaces, `sub -section` → `sub-section`, and no space before `, . ; :`.
  - Prarabdha questions are taken from the source unchanged.
  - CPA 2019 questions are **templates** ("What does Section N … provide?") because the statute has no questions of its own. Their answers are still verbatim text from the Act.
- **Gold set:**
  - `fact` rows come straight from master rows (`origin: source`); the 10 CPA ones therefore carry the template question.
  - `math_logic` and `adversarial` rows are team-authored (`origin: authored`), because no public source has them. Their answers are produced by code (jurisdiction and dates) or by the expected behaviour (refusal).

To check that any row is verbatim, compare its alphanumeric content against the source row (`source_row`) or CPA section: lowercase both, strip everything except `[0-9a-z]`, and check that the strings are equal.

### Pecuniary jurisdiction (2021 Rules)

Neither dataset contains the Consumer Protection (Jurisdiction of the District Commission, the State Commission and the National Commission) Rules, 2021. Those Rules set the limits that apply today, by consideration paid:

| Commission | Consideration paid |
|---|---|
| District | up to ₹50 lakh |
| State | above ₹50 lakh, up to ₹2 crore |
| National | above ₹2 crore |

The Act's printed text (₹1 crore / ₹10 crore) and the 1986-era judgments (₹20 lakh / ₹1 crore) are both out of date. These limits therefore live **only in code**: the `Calculate_Jurisdiction` tool and `commission_for()` in stage 3. The LLM is never asked to recall them.

---

## 6. Roadmap

| Week | Deliverable |
|---|---|
| 1 | Gold set + bare-model baseline, with failures logged |
| 2 | Hybrid RAG (BM25 + `BAAI/bge-base-en-v1.5`, `EnsembleRetriever` 50/50, rerank top 10 → 3) with Recall@3 measured on the gold `fact` rows |
| 3 | LangGraph Planner → Executor → Critic agent with `Query_Legal_Corpus`, `Calculate_Jurisdiction`, `Extract_Claim_Details`, and a 10-step cap |
| 4 | Streamlit app (PII scrubbing, 5 queries per session, human-in-the-loop disclaimer) in a single Dockerfile |

Final-run failures are classified as one of: **Retrieval Miss**, **Reasoning Error**, **Tool-call Error**, **Hallucination**.

## LLM endpoint (planned)

The app talks to an OpenAI-compatible endpoint:

- **Local Ollama:** `base_url="http://localhost:11434/v1"`, `api_key="ollama"`. The key is required by the client but ignored by the server.
- **Cloud provider** (Hugging Face or Groq): put the key in `.env`, which is gitignored, and never commit it.

Use `temperature=0.0` for evaluation runs.
