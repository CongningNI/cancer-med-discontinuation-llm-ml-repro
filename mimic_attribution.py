#!/usr/bin/env python3
"""
mimic_attribution.py

Implements the paper's "mimic-SHAP" attribution aggregation for fine-tuned GPT-4o outputs.

Inputs (no patient data embedded):
- citations.csv : one row per (instance, cited_feature) with at least:
    instance_id, feature, predicted_label, probability
  where probability is the model's P(discontinuation) in [0, 1].

Outputs:
- llm_attribution_scores.csv : feature-level scores:
    freq, weighted_freq, dir_score, n_citations
- top_features.txt : top-K features by weighted_freq

Notes:
- weighted_freq uses confidence weight w_i = |p_i - 0.5| (as in the manuscript)
- dir_score summarizes directionality from predicted_label (simple, optional)
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd


def compute_scores(
    citations: pd.DataFrame,
    prob_col: str = "probability",
    feature_col: str = "feature",
    instance_col: str = "instance_id",
    label_col: str = "predicted_label",
) -> pd.DataFrame:
    required = {prob_col, feature_col, instance_col}
    missing = required - set(citations.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    c = citations.copy()

    # Basic hygiene
    c[prob_col] = pd.to_numeric(c[prob_col], errors="coerce")
    c = c.dropna(subset=[prob_col, feature_col, instance_col])
    c[feature_col] = c[feature_col].astype(str)

    # confidence weight
    c["confidence"] = (c[prob_col] - 0.5).abs()

    n_instances = c[instance_col].nunique()
    total_conf = float(c["confidence"].sum())
    if n_instances == 0:
        raise ValueError("No instances found after cleaning.")

    rows = []
    for feat, g in c.groupby(feature_col):
        n_cit = len(g)
        freq = n_cit / n_instances
        wfreq = float(g["confidence"].sum()) / total_conf if total_conf > 0 else 0.0

        # Optional simple directionality summary:
        # proportion of citations in instances predicted as class 1 minus class 0.
        dir_score = np.nan
        if label_col in c.columns:
            # Cast to int if possible
            gg = g.dropna(subset=[label_col]).copy()
            if len(gg) > 0:
                gg[label_col] = pd.to_numeric(gg[label_col], errors="coerce")
                pos = float((gg[label_col] == 1).mean())
                neg = float((gg[label_col] == 0).mean())
                dir_score = pos - neg

        rows.append({
            "feature": feat,
            "n_citations": int(n_cit),
            "freq": float(freq),
            "weighted_freq": float(wfreq),
            "dir_score": float(dir_score) if dir_score == dir_score else np.nan,  # keep NaN
        })

    out = pd.DataFrame(rows).sort_values("weighted_freq", ascending=False).reset_index(drop=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--citations", type=Path, required=True, help="CSV with (instance_id, feature, probability, ...).")
    ap.add_argument("--outdir", type=Path, default=Path("outputs_attr"))
    ap.add_argument("--topk", type=int, default=10)
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.citations)
    scores = compute_scores(df)

    out_csv = args.outdir / "llm_attribution_scores.csv"
    scores.to_csv(out_csv, index=False)

    top = scores.head(args.topk)
    out_txt = args.outdir / "top_features.txt"
    with out_txt.open("w", encoding="utf-8") as f:
        for i, r in top.iterrows():
            f.write(f"{i+1}. {r['feature']}\tweighted_freq={r['weighted_freq']:.4f}\n")

    print(f"Saved: {out_csv.resolve()}")
    print(f"Saved: {out_txt.resolve()}")


if __name__ == "__main__":
    main()
