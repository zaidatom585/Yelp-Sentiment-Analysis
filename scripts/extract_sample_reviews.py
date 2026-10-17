"""Pull the full review text for the businesses in the 10,000-business sample.

Reads the newest output/yelp_10k_sample_*.csv (made by Importer_10k_reviews.py), streams the Yelp review
file once, and writes every review for those businesses to data/sample_reviews.parquet. The parquet file
stays in data/, which is git-ignored: Yelp's licence does not allow redistributing review text.

    python scripts/extract_sample_reviews.py
"""
import argparse
import glob
import json
import os
import sys

import pyarrow as pa
import pyarrow.parquet as pq

REVIEW_FILE = os.path.join("data", "yelp_academic_dataset_review.json")
OUT_FILE = os.path.join("data", "sample_reviews.parquet")
CHUNK = 100_000
SCHEMA = pa.schema([("review_id", pa.string()), ("business_id", pa.string()), ("stars", pa.int8()),
                    ("useful", pa.int32()), ("date", pa.string()), ("text", pa.string())])


def sample_business_ids():
    files = sorted(glob.glob(os.path.join("output", "yelp_10k_sample_*.csv")), key=os.path.getmtime)
    if not files:
        sys.exit("No output/yelp_10k_sample_*.csv found. Run scripts/Importer_10k_reviews.py first.")
    import pandas as pd
    ids = set(pd.read_csv(files[-1], usecols=["business_id"])["business_id"].dropna())
    print(f"Sample file: {files[-1]} ({len(ids):,} businesses)")
    return ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--review-file", default=REVIEW_FILE, help="Yelp review JSON (e.g. a Kaggle input path)")
    ap.add_argument("--out", default=OUT_FILE)
    args = ap.parse_args()
    if not os.path.exists(args.review_file):
        sys.exit(f"Missing {args.review_file}")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    ids = sample_business_ids()
    writer = pq.ParquetWriter(args.out, SCHEMA, compression="zstd")
    buf, kept, read, bad = [], 0, 0, 0
    with open(args.review_file, encoding="utf-8") as f:
        for line in f:
            read += 1
            if read % 1_000_000 == 0:
                print(f"  read {read:,} reviews, kept {kept + len(buf):,}")
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                bad += 1
                continue
            if r.get("business_id") not in ids or not r.get("text"):
                continue
            buf.append({"review_id": r.get("review_id"), "business_id": r["business_id"],
                        "stars": int(r.get("stars") or 0), "useful": int(r.get("useful") or 0),
                        "date": r.get("date"), "text": r["text"]})
            if len(buf) >= CHUNK:
                writer.write_table(pa.Table.from_pylist(buf, schema=SCHEMA))
                kept += len(buf)
                buf = []
    if buf:
        writer.write_table(pa.Table.from_pylist(buf, schema=SCHEMA))
        kept += len(buf)
    writer.close()
    print(f"Done: read {read:,} reviews ({bad:,} malformed), kept {kept:,} for the sample -> {args.out}")


if __name__ == "__main__":
    main()
