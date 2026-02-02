#!/usr/bin/env python3
"""
gpt_prompts_and_finetune_jsonl.py

Utilities to (1) render a structured "Patient Clinical Profile" prompt and
(2) generate JSONL files for fine-tuning a GPT-4o-style model.

This file intentionally avoids:
- any API keys, endpoints, or vendor-specific configuration
- any patient-level example rows

It focuses on the prompt TEMPLATE and fine-tuning DATA FORMAT, which reviewers requested.

Expected label convention:
- "0" = completed treatment
- "1" = early discontinuation
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


SYSTEM_INSTRUCTION_CLASSIFICATION = (
    "You are a clinical AI assistant specializing in predicting cancer medication discontinuation patterns. "
    "Respond with only the predicted class: Output '0' if you predict the patient will COMPLETE treatment. "
    "Output '1' if you predict the patient will DISCONTINUE EARLY."
)

SYSTEM_INSTRUCTION_ATTRIBUTION = (
    "You are a clinical AI assistant specializing in predicting cancer medication discontinuation patterns. "
    "Return a JSON object with keys: "
    "{'prediction': {'label': 0 or 1, 'probability': float}, "
    "'reasoning': {'features': [list of feature names cited], 'explanation': string}}. "
    "Use only input information. Keep 'features' concise (<=10)."
)


def render_patient_profile(record: Dict) -> str:
    """
    Render a human-readable profile from a dict.

    Required keys (recommended):
      - med_name, med_ingredient
      - age, gender, race, ethnicity, insurance, bmi
      - famhist (string or list)
      - meds, dx, procedures, labs (each list of strings)
    """
    def _list_block(title: str, items: Optional[List[str]]) -> str:
        items = items or []
        if not items:
            return f"{title} (0):\n  • None\n"
        lines = "\n".join([f"  • {x}" for x in items])
        return f"{title} ({len(items)}):\n{lines}\n"

    famhist = record.get("famhist", "")
    if isinstance(famhist, list):
        famhist = ", ".join(famhist)

    txt = []
    txt.append("Patient Clinical Profile:\n")
    txt.append("Cancer Medication Under Evaluation:\n")
    txt.append(f"- Medication: {record.get('med_name', 'UNKNOWN')}\n")
    if record.get("med_ingredient"):
        txt.append(f"- Active Ingredient: {record.get('med_ingredient')}\n")
    txt.append("\nDemographics:\n")
    txt.append(f"- Age: {record.get('age', 'NA')} years\n")
    txt.append(f"- Gender: {record.get('gender', 'NA')}\n")
    txt.append(f"- Race: {record.get('race', 'NA')}\n")
    txt.append(f"- Ethnicity: {record.get('ethnicity', 'NA')}\n")
    txt.append(f"- Insurance: {record.get('insurance', 'NA')}\n")
    txt.append(f"- BMI: {record.get('bmi', 'NA')}\n")
    if famhist:
        txt.append(f"- Family History: {famhist}\n")

    txt.append("\nBaseline Clinical Features (pre-treatment):\n\n")
    txt.append(_list_block("Medications", record.get("meds")))
    txt.append(_list_block("Diagnoses", record.get("dx")))
    txt.append(_list_block("Procedures", record.get("procedures")))
    txt.append(_list_block("Laboratory Results", record.get("labs")))

    txt.append(
        "\nTask: Based on this patient's baseline characteristics, predict whether they will discontinue "
        "the prescribed cancer medication early (before treatment completion) or complete the full treatment course."
    )
    return "".join(txt)


def build_messages(record: Dict, mode: str) -> List[Dict]:
    """
    mode:
      - 'classify'    -> system instruction expects only "0" or "1"
      - 'attribute'   -> system instruction expects JSON with probability + cited features
    """
    if mode not in {"classify", "attribute"}:
        raise ValueError("mode must be 'classify' or 'attribute'")

    system = SYSTEM_INSTRUCTION_CLASSIFICATION if mode == "classify" else SYSTEM_INSTRUCTION_ATTRIBUTION
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": render_patient_profile(record)},
    ]


def make_finetune_jsonl(
    df: pd.DataFrame,
    out_path: Path,
    label_col: str = "label",
    mode: str = "classify",
) -> None:
    """
    Create JSONL suitable for fine-tuning.
    Each line: {"messages": [system, user, assistant]}

    The assistant response is the label ("0"/"1") for classification mode.
    """
    if label_col not in df.columns:
        raise ValueError(f"Missing label column: {label_col}")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            record = row.to_dict()
            label = str(int(record.pop(label_col)))
            messages = build_messages(record, mode=mode)
            messages.append({"role": "assistant", "content": label})
            f.write(json.dumps({"messages": messages}, ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True, help="CSV/Parquet with required columns + label.")
    ap.add_argument("--out", type=Path, required=True, help="Output JSONL path.")
    ap.add_argument("--label-col", type=str, default="label")
    ap.add_argument("--mode", type=str, default="classify", choices=["classify", "attribute"])
    args = ap.parse_args()

    if args.input.suffix.lower() == ".csv":
        df = pd.read_csv(args.input)
    elif args.input.suffix.lower() in {".parquet", ".pq"}:
        df = pd.read_parquet(args.input)
    else:
        raise ValueError("Input must be CSV or Parquet")

    make_finetune_jsonl(df, args.out, label_col=args.label_col, mode=args.mode)
    print(f"Saved JSONL: {args.out.resolve()}")


if __name__ == "__main__":
    main()
