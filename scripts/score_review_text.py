"""Score sentiment from review text and compare it with star ratings.

1. VADER (a rule-based lexicon model) scores every review in data/sample_reviews.parquet.
2. A pretrained transformer (RoBERTa fine-tuned for 3-class sentiment) scores a stratified sample of
   reviews, the same number per star rating, so the two models can be compared fairly.
3. Aspect scores: sentences mentioning food, service, price, wait time or cleanliness are scored
   separately, so a review can be positive about food and negative about service.
4. Both models are evaluated against a label taken from the review's own stars (4-5 positive,
   3 neutral, 1-2 negative). Stars are a proxy, not ground truth, so disagreements are kept for review.

Outputs (output/ is git-ignored except the evaluation summary, which holds no review text):
    output/business_text_sentiment.csv   one row per business, for Power BI
    output/review_scores.parquet         one row per review (local only)
    output/disagreements.csv             e.g. 5-star reviews whose text reads negative (local only)
    results/sentiment_evaluation.md      accuracy, macro F1 and confusion matrices

    python scripts/score_review_text.py                     # VADER on all + transformer on 5,000 per star
    python scripts/score_review_text.py --per-star 1000     # quicker transformer pass
    python scripts/score_review_text.py --no-transformer    # VADER and aspects only
"""
import argparse
import os
import re
import time

import numpy as np
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

IN_FILE = os.path.join("data", "sample_reviews.parquet")
MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
LABELS = ["negative", "neutral", "positive"]
ASPECTS = {
    "food": r"\b(food|dish|dishes|meal|taste|tasty|flavou?r|delicious|menu|pizza|burger|portion)s?\b",
    "service": r"\b(service|staff|server|waiter|waitress|manager|rude|friendly|employee|customer service)s?\b",
    "price": r"\b(price|prices|priced|expensive|cheap|overpriced|value|cost|worth)\b",
    "wait": r"\b(wait|waited|waiting|slow|line|queue|minutes|hour|took forever)\b",
    "cleanliness": r"\b(clean|dirty|filthy|smell|smelled|bathroom|restroom|hygiene|sticky)\b",
}
SENT_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")


def star_label(stars):
    return np.select([stars >= 4, stars == 3], ["positive", "neutral"], "negative")


def vader_label(compound):
    return np.select([compound >= 0.05, compound <= -0.05], ["positive", "negative"], "neutral")


def score_vader(df):
    sia = SentimentIntensityAnalyzer()
    patterns = {a: re.compile(p, re.I) for a, p in ASPECTS.items()}
    compound = np.empty(len(df))
    aspect = {a: np.full(len(df), np.nan) for a in ASPECTS}
    t0 = time.time()
    for i, text in enumerate(df["text"].values):
        compound[i] = sia.polarity_scores(text)["compound"]
        sentences = [s for s in SENT_SPLIT.split(text) if s.strip()]
        hits = {a: [] for a in ASPECTS}
        for s in sentences:
            matched = [a for a, p in patterns.items() if p.search(s)]
            if matched:
                c = sia.polarity_scores(s)["compound"]
                for a in matched:
                    hits[a].append(c)
        for a, vals in hits.items():
            if vals:
                aspect[a][i] = float(np.mean(vals))
        if (i + 1) % 50_000 == 0:
            print(f"  VADER: {i + 1:,} / {len(df):,} reviews ({time.time() - t0:.0f}s)")
    df["vader_compound"] = compound
    df["vader_label"] = vader_label(compound)
    for a in ASPECTS:
        df[f"aspect_{a}"] = aspect[a]
    return df


def score_transformer(df, per_star, batch_size):
    from transformers import pipeline
    sample = (df.groupby("stars", group_keys=False)
                .apply(lambda g: g.sample(n=min(per_star, len(g)), random_state=42)))
    print(f"Transformer: scoring {len(sample):,} reviews ({per_star:,} per star) with {MODEL}")
    clf = pipeline("sentiment-analysis", model=MODEL, tokenizer=MODEL, truncation=True, max_length=512)
    t0, preds = time.time(), []
    texts = sample["text"].tolist()
    for start in range(0, len(texts), batch_size):
        preds += clf(texts[start:start + batch_size], batch_size=batch_size)
        if (start // batch_size) % 50 == 0:
            print(f"  {start + batch_size:,} / {len(texts):,} ({time.time() - t0:.0f}s)")
    sample = sample.copy()
    sample["transformer_label"] = [p["label"].lower() for p in preds]
    sample["transformer_score"] = [p["score"] for p in preds]
    return df.merge(sample[["review_id", "transformer_label", "transformer_score"]], on="review_id", how="left")


def evaluate(df, col):
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
    d = df.dropna(subset=[col])
    y, p = d["star_label"], d[col]
    cm = pd.DataFrame(confusion_matrix(y, p, labels=LABELS),
                      index=[f"stars: {l}" for l in LABELS], columns=[f"text: {l}" for l in LABELS])
    return len(d), accuracy_score(y, p), f1_score(y, p, labels=LABELS, average="macro"), cm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-star", type=int, default=5000)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--no-transformer", action="store_true")
    ap.add_argument("--reviews", default=IN_FILE, help="parquet made by extract_sample_reviews.py")
    args = ap.parse_args()

    df = pd.read_parquet(args.reviews)
    print(f"Loaded {len(df):,} reviews for {df['business_id'].nunique():,} businesses")
    df["star_label"] = star_label(df["stars"].values)
    df = score_vader(df)
    if not args.no_transformer:
        df = score_transformer(df, args.per_star, args.batch_size)

    os.makedirs("output", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    df.drop(columns=["text"]).to_parquet(os.path.join("output", "review_scores.parquet"), index=False)

    # Business-level table for Power BI
    g = df.groupby("business_id")
    biz = pd.DataFrame({
        "reviews_scored": g.size(),
        "avg_review_stars": g["stars"].mean().round(2),
        "pct_star_positive": (g["star_label"].apply(lambda s: (s == "positive").mean()) * 100).round(1),
        "pct_text_positive": (g["vader_label"].apply(lambda s: (s == "positive").mean()) * 100).round(1),
        "pct_text_negative": (g["vader_label"].apply(lambda s: (s == "negative").mean()) * 100).round(1),
        "avg_text_compound": g["vader_compound"].mean().round(3),
    })
    biz["star_text_gap_pts"] = (biz["pct_star_positive"] - biz["pct_text_positive"]).round(1)
    for a in ASPECTS:
        biz[f"{a}_score"] = g[f"aspect_{a}"].mean().round(3)
        biz[f"{a}_mentions"] = g[f"aspect_{a}"].count()
    biz.reset_index().to_csv(os.path.join("output", "business_text_sentiment.csv"), index=False)

    # Clear disagreements, for reading by hand (contains review text: local only)
    dis = df[((df["stars"] == 5) & (df["vader_compound"] <= -0.5)) | ((df["stars"] == 1) & (df["vader_compound"] >= 0.5))]
    dis[["review_id", "business_id", "stars", "vader_compound", "text"]].head(500).to_csv(
        os.path.join("output", "disagreements.csv"), index=False)

    # Evaluation summary (no review text, safe to commit)
    lines = ["# Text sentiment vs star rating", "",
             f"Reviews scored with VADER: {len(df):,} across {df['business_id'].nunique():,} businesses.",
             "Star label: 4-5 stars positive, 3 neutral, 1-2 negative. Stars are a proxy, not ground truth.", ""]
    models = [("VADER", "vader_label")] + ([] if args.no_transformer else [("RoBERTa transformer", "transformer_label")])
    for name, col in models:
        if col == "vader_label" and not args.no_transformer:
            n, acc, f1, cm = evaluate(df[df["transformer_label"].notna()], col)
            name += " (same stratified sample)"
        else:
            n, acc, f1, cm = evaluate(df, col)
        lines += [f"## {name}", "", f"Reviews: {n:,} · accuracy {acc:.3f} · macro F1 {f1:.3f}", "",
                  cm.to_markdown(), ""]
    lines += ["## Aspect coverage", "", "| Aspect | Reviews mentioning it | Mean sentence score |", "| --- | --- | --- |"]
    for a in ASPECTS:
        col = df[f"aspect_{a}"]
        lines.append(f"| {a} | {col.notna().sum():,} ({100 * col.notna().mean():.1f}%) | {col.mean():.3f} |")
    lines += ["", f"Clear disagreements (5 stars with text <= -0.5, or 1 star with text >= 0.5): {len(dis):,}"]
    with open(os.path.join("results", "sentiment_evaluation.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
