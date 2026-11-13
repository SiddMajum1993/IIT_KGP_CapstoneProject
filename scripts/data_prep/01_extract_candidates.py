"""Stage 1: scan every Prarabdha shard and keep rows that look like Indian consumer law.

A cheap Arrow substring prefilter runs first; the Python regex only sees rows that pass it.

Output: data/interim/consumer_candidates.parquet (all matches, with a global source row id).
"""
import argparse
import glob
import os
import re

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

PREFILTER = ["consumer", "ncdrc", "deficiency in service", "unfair trade practice"]

# Strong signals: any one of these marks a row as consumer-law.
STRONG = re.compile(
    r"consumer protection act|\bncdrc\b|national consumer disputes|consumer disputes redressal"
    r"|state consumer|district consumer|consumer forum|consumer commission|consumer complaint"
    r"|deficiency in service|unfair trade practice|restrictive trade practice|product liability"
    r"|central consumer protection authority",
    re.IGNORECASE,
)
# Weak signals: only count when the word "consumer" also appears.
WEAK = re.compile(r"district forum|state commission|national commission|opposite part(y|ies)|complainant",
                  re.IGNORECASE)
CONSUMER = re.compile(r"\bconsumers?\b", re.IGNORECASE)


def is_candidate(text: str) -> bool:
    return bool(STRONG.search(text)) or (bool(WEAK.search(text)) and bool(CONSUMER.search(text)))


def prefilter_mask(batch: pa.RecordBatch) -> pa.Array:
    mask = None
    for col in ("context", "question", "response"):
        lower = pc.utf8_lower(pc.fill_null(batch.column(col), ""))
        for term in PREFILTER:
            m = pc.match_substring(lower, term)
            mask = m if mask is None else pc.or_(mask, m)
    return mask


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default="data/raw/prarabdha_indian_legal_sft")
    ap.add_argument("--out", default="data/interim/consumer_candidates.parquet")
    args = ap.parse_args()

    shards = sorted(glob.glob(os.path.join(args.raw_dir, "train-*.parquet")))
    assert len(shards) == 33, f"expected 33 shards, found {len(shards)}"
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    kept, offset = [], 0
    for i, path in enumerate(shards):
        n_rows = n_hits = 0
        for batch in pq.ParquetFile(path).iter_batches(batch_size=50_000):
            ids = pa.array(range(offset, offset + batch.num_rows), pa.int64())
            batch = batch.append_column("source_row", ids)
            offset += batch.num_rows
            n_rows += batch.num_rows
            df = batch.filter(prefilter_mask(batch)).to_pandas()
            if df.empty:
                continue
            text = df["context"].fillna("") + " " + df["question"].fillna("") + " " + df["response"].fillna("")
            hit = df[text.map(is_candidate)].copy()
            hit["shard"] = i
            n_hits += len(hit)
            kept.append(hit)
        print(f"shard {i:02d}: {n_rows:>7} rows, {n_hits:>5} candidates", flush=True)

    out = pd.concat(kept, ignore_index=True)
    out.to_parquet(args.out, index=False)
    print(f"total scanned {offset}, candidates {len(out)} -> {args.out}")


if __name__ == "__main__":
    main()
