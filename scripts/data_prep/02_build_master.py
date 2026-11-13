"""Stage 2: build the master QA dataset (question, context, answer) for the RAG corpus.

Two sources:
  * cpa2019   - sections of the Consumer Protection Act, 2019, one row per section chunk
                (Section 2 is split per definition). Questions/answers are templated from the text.
  * prarabdha - the best consumer-law rows from stage 1, chosen by a quality score and capped
                per topic so no single domain (e.g. insurance) dominates.

Output: data/processed/master_qa.jsonl
"""
import argparse
import json
import os
import re

import pandas as pd

MAX_CTX_CHARS = 2500  # matches the Prarabdha context size, ~600 tokens

STOPWORDS = set("""a an the of to in on for by with and or not is are was were be been being as at from that this
these those it its he she they them his her their which who whom whose what when where why how shall may can
could would should will has have had do does did any such said under than then there so if into also no yes""".split())

TOPICS = {
    "insurance": r"insur|policy holder|claim was repudiated|repudiat",
    "banking_finance": r"\bbank|loan|credit card|cheque|\bemi\b|interest rate|non-banking",
    "medical": r"medical negligence|hospital|doctor|surgery|patient",
    "housing_real_estate": r"builder|flat|apartment|allot|possession of the (flat|plot)|housing board|development authority",
    "ecommerce_online": r"e-commerce|online|website|internet",
    "telecom": r"telecom|mobile|telephone",
    "electricity_utilities": r"electricity|power supply|water supply|gas connection",
    "transport_travel": r"airline|railway|flight|courier|bus|travel",
    "education": r"universit|college|coaching|institute|admission|education",
    "goods_defect": r"manufactur|defect in (the )?goods|defective|vehicle|car |seeds|warranty",
}
TOPIC_RE = {k: re.compile(v, re.IGNORECASE) for k, v in TOPICS.items()}

STRONG = re.compile(
    r"consumer protection act|\bncdrc\b|national consumer disputes|consumer disputes redressal|consumer forum"
    r"|district forum|district commission|state commission|national commission|deficiency in service"
    r"|unfair trade practice|consumer complaint",
    re.IGNORECASE,
)
PLACEHOLDER = re.compile(r"not specified|not mentioned|not provided|not available in the|\[[^\]]*\]", re.IGNORECASE)
# Requests to act (draft, instruct, find for me) rather than questions about the law.
INSTRUCTION_Q = re.compile(
    r"^(?:could|can|would|will) you (?:please )?(?:instruct|direct|draft|write|prepare|help|find|search|provide a reference|"
    r"list|suggest|recommend)|^(?:please )?(?:draft|write|prepare|instruct|direct|find)\b",
    re.IGNORECASE,
)
ERA_1986 = re.compile(r"\b1986\b|district forum|\bforum\b", re.IGNORECASE)
ERA_2019 = re.compile(r"\b2019\b|district commission|central consumer protection authority", re.IGNORECASE)
# The source scrubbed many numbers, leaving "Act,  " or "Section  of" behind.
STRIPPED_NUM = re.compile(r"(?:Act|[Ss]ection|[Ss]ub-section|Rs\.?|dated)[,.]?\s{2,}")
SECTION_REF = re.compile(r"[Ss]ection[s]?\s+(\d+[A-Z]?)")
# Q/A that state a pecuniary limit; those from Prarabdha are almost all pre-2021 figures.
AMOUNT = r"(?:\d+(?:\.\d+)?|one|two|five|ten|twenty|fifty)\s*(?:lakhs?|crores?)"
JURIS = r"pecuniary|jurisdiction|entertain|district forum|district commission|state commission|national commission"
PECUNIARY = re.compile(rf"(?:{AMOUNT}).{{0,150}}(?:{JURIS})|(?:{JURIS}).{{0,150}}(?:{AMOUNT})", re.IGNORECASE | re.DOTALL)
# Time limits and appeal periods changed between the 1986 and 2019 Acts.
TIMELINE = re.compile(r"\b(?:\d+|thirty|forty-five|ninety|two years?)\s*(?:days|months|years)\b", re.IGNORECASE)

SUPERSEDED_LIMITS = {"34", "47", "58"}


def clean(text: str) -> str:
    text = re.sub(r"(\w) -(\w)", r"\1-\2", text)  # "sub -section" -> "sub-section"
    text = re.sub(r"\s+([.,;:])", r"\1", text)
    return re.sub(r"\s+", " ", text).strip()


def content_words(text: str) -> set:
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOPWORDS and len(w) > 2}


def grounding(answer: str, context: str) -> float:
    """Share of the answer's content words that appear in the context (cheap hallucination check)."""
    a = content_words(answer)
    return len(a & content_words(context)) / len(a) if a else 0.0


def topic_of(text: str) -> str:
    hits = {k: len(r.findall(text)) for k, r in TOPIC_RE.items()}
    best = max(hits, key=hits.get)
    return best if hits[best] else "general"


def era_of(text: str) -> str:
    old, new = bool(ERA_1986.search(text)), bool(ERA_2019.search(text))
    return "both" if old and new else "1986" if old else "2019" if new else "unknown"


def split_long(text: str, limit: int = MAX_CTX_CHARS) -> list:
    """Split at clause markers like ' (3) ' so each chunk stays under the limit."""
    if len(text) <= limit:
        return [text]
    parts = re.split(r"(?= \(\d+[A-Z]?\) )", text)
    chunks, cur = [], ""
    for p in parts:
        if cur and len(cur) + len(p) > limit:
            chunks.append(cur.strip())
            cur = ""
        cur += p
        while len(cur) > limit:  # a single clause longer than the limit
            cut = cur.rfind("; ", 0, limit)
            cut = cut + 1 if cut > limit // 2 else limit
            chunks.append(cur[:cut].strip())
            cur = cur[cut:]
    if cur.strip():
        chunks.append(cur.strip())
    return chunks


def first_sentences(text: str, max_chars: int = 600) -> str:
    out = ""
    for s in re.split(r"(?<=[.:;])\s+", text):
        if out and len(out) + len(s) > max_chars:
            break
        out += (" " if out else "") + s
    return out[:max_chars].strip()


def cpa_rows(path: str) -> list:
    sections = json.load(open(path, encoding="utf-8"))["sections"]
    rows = []
    for s in sections:
        num, title, text = s["section_number"], s["section_title"].strip(), clean(s["text"])
        if num == "2":
            # One row per definition: (n) "term" means ...
            for m in re.finditer(r"\((\d+)\) “([^”]+)”(.*?)(?= \(\d+\) “|$)", text):
                term, body = m.group(2).strip(), clean(m.group(0))
                for k, chunk in enumerate(split_long(body)):
                    rows.append(dict(
                        id=f"CPA-s2({m.group(1)})" + (f"-{k}" if k else ""), source="cpa2019", section=f"2({m.group(1)})", chapter=s["chapter_title"],
                        question=f"How does the Consumer Protection Act, 2019 define “{term}”?" if k == 0 else
                        f"What else does the definition of “{term}” in Section 2({m.group(1)}) of the Consumer Protection Act, 2019 include?",
                        context=f"Consumer Protection Act, 2019 - Section 2 (Definitions): {chunk}",
                        answer=first_sentences(chunk),
                    ))
            continue
        for k, chunk in enumerate(split_long(text)):
            q = (f"What does Section {num} of the Consumer Protection Act, 2019 ({title}) provide?" if k == 0 else
                 f"What further provisions does Section {num} of the Consumer Protection Act, 2019 ({title}) contain?")
            rows.append(dict(
                id=f"CPA-s{num}" + (f"-{k}" if k else ""), source="cpa2019", section=num, chapter=s["chapter_title"], question=q,
                context=f"Consumer Protection Act, 2019 - Section {num} ({title}): {chunk}",
                answer=first_sentences(re.sub(rf"^{num}\.\s*{re.escape(title)}\s*\.?\s*[—-]*", "", chunk)),
            ))
    for r in rows:
        r.update(act_era="2019", topic="statute", quality_score=None, grounding=1.0,
                 needs_review=False, source_row=None, digits_intact=True,
                 note=("Pecuniary limit printed here was revised by the 2021 Jurisdiction Rules; "
                       "the jurisdiction tool applies the current limits.") if r["section"] in SUPERSEDED_LIMITS else None)
    return rows


def prarabdha_rows(path: str, n: int, topic_cap: float) -> list:
    df = pd.read_parquet(path)
    df = df.rename(columns={"response": "answer"})
    for c in ("context", "question", "answer"):
        df[c] = df[c].fillna("").map(clean)

    ascii_ratio = df.context.map(lambda s: sum(ch.isascii() for ch in s) / max(1, len(s)))
    keep = (
        (df.question.str.len().between(30, 300))
        & (df.answer.str.len().between(40, 1200))
        & (ascii_ratio >= 0.97)
        & (df.question.str.lower() != df.answer.str.lower())
        & ~df.answer.str.contains(PLACEHOLDER)
        & ~df.question.str.contains(INSTRUCTION_Q)
        & df.context.str.contains(STRONG)  # the context itself must be consumer law, not just the question
        & ~(df.question + " " + df.answer).str.contains(PECUNIARY)
    )
    df = df[keep].copy()
    print(f"prarabdha: {keep.size} candidates -> {len(df)} after hard filters")

    df["grounding"] = [grounding(a, c) for a, c in zip(df.answer, df.context)]
    df = df[df.grounding >= 0.6].copy()
    print(f"prarabdha: {len(df)} after grounding >= 0.6")

    df["digits_intact"] = ~df.context.str.contains(STRIPPED_NUM)
    df["relevance"] = df.context.str.count(STRONG).clip(upper=5) / 5
    df["quality_score"] = (
        2.0 * df.grounding + df.relevance + 0.5 * df.digits_intact
        + 0.2 * df.question.str.strip().str.endswith("?")
        + 0.3 * df.answer.str.len().between(60, 600)
    ).round(3)
    df["topic"] = (df.context + " " + df.question).map(topic_of)
    df["act_era"] = df.context.map(era_of)

    # Diversity: one QA per context, and drop contexts that share an opening (same judgment page).
    df = df.sort_values(["quality_score", "source_row"], ascending=[False, True], kind="mergesort")
    df["ctx_head"] = df.context.str[:300]
    df = df.drop_duplicates("context").drop_duplicates("ctx_head")
    df = df[~df.question.str.lower().duplicated()]
    print(f"prarabdha: {len(df)} after dedup")

    cap = int(n * topic_cap)
    picked = df.groupby("topic", group_keys=False).head(cap)
    picked = picked.sort_values(["quality_score", "source_row"], ascending=[False, True], kind="mergesort").head(n)
    print("prarabdha topic mix:", picked.topic.value_counts().to_dict())
    print("prarabdha act era:", picked.act_era.value_counts().to_dict())
    # 1986-era answers that quote a time limit may be stale under the 2019 Act.
    stale = (picked.act_era == "1986") & (picked.question + " " + picked.answer).str.contains(TIMELINE)
    picked = picked.assign(source="prarabdha", chapter=None, needs_review=stale,
                           note=stale.map({True: "1986-era time limit; verify against the 2019 Act.", False: None}),
                           id=picked.source_row.map(lambda r: f"PR-{int(r)}"))
    picked["section"] = picked.context.map(lambda c: ",".join(sorted(set(SECTION_REF.findall(c)))[:5]) or None)
    return picked[["id", "source", "source_row", "section", "chapter", "question", "context", "answer", "act_era",
                   "topic", "quality_score", "grounding", "digits_intact", "needs_review", "note"]].to_dict("records")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", default="data/interim/consumer_candidates.parquet")
    ap.add_argument("--cpa-json", default="data/raw/cpa_2019/the_consumer_protection_act_2019.json")
    ap.add_argument("--out", default="data/processed/master_qa.jsonl")
    ap.add_argument("--total", type=int, default=2000)
    ap.add_argument("--topic-cap", type=float, default=0.25)
    args = ap.parse_args()

    statute = cpa_rows(args.cpa_json)
    print(f"cpa2019: {len(statute)} rows from 107 sections")
    pra = prarabdha_rows(args.candidates, args.total - len(statute), args.topic_cap)
    rows = statute + pra
    assert len({r["id"] for r in rows}) == len(rows), "duplicate ids"

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in rows:
            r = {k: None if not isinstance(v, (list, dict)) and pd.isna(v) else v for k, v in r.items()}
            if r["source_row"] is not None:
                r["source_row"] = int(r["source_row"])
            r["needs_review"] = bool(r["needs_review"])
            r["digits_intact"] = bool(r["digits_intact"])
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} rows ({sum(r['needs_review'] for r in rows)} flagged for review) -> {args.out}")


if __name__ == "__main__":
    main()
