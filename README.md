# MIE 2026 Reproducibility Package (Code-Only)

This repository contains a minimal, **data-free** code package to support reproducibility of:
1) traditional ML baselines (LR/DT/RF/XGBoost),
2) GPT-4o prompt templates + fine-tuning JSONL generation, and
3) the SHAP-like **mimic attribution** aggregation used for Figure 1.

**No patient-level EHR data are included.** Users must supply their own de-identified tables.

## Files

- `train_ml_models.py`
  - Trains/evaluates LR, DT, RF, and XGBoost with the fixed hyperparameters used in the paper.
  - Input: pre-split `train` and `test` tables (CSV/Parquet) with a binary label column (0/1).
  - Output: `outputs_ml/metrics.csv`

- `gpt_prompts_and_finetune_jsonl.py`
  - Contains the structured "Patient Clinical Profile" template and system instructions.
  - Generates JSONL suitable for fine-tuning from a tabular input file (CSV/Parquet).
  - Output schema matches the paper’s label convention: `"0"`=complete, `"1"`=early discontinuation.

- `mimic_attribution.py`
  - Aggregates per-instance LLM feature citations into global importance using confidence-weighted frequency:
    `w_i = |p_i - 0.5|`.
  - Input: `citations.csv` with columns `(instance_id, feature, probability)` and optionally `predicted_label`.
  - Output: `outputs_attr/llm_attribution_scores.csv` and `top_features.txt`.

## Inputs (expected columns)

### Traditional ML (`train_ml_models.py`)
Your `train` and `test` files must contain:
- a binary label column (default: `label`)
- numeric feature columns (e.g., one-hot encoded variables)

If present, the script scales only the continuous columns:
- `age_at_cutoff`, `bmi_at_baseline`
(you may rename and modify in the script if needed)

### Fine-tune JSONL (`gpt_prompts_and_finetune_jsonl.py`)
Your input file should contain the following recommended columns:
- `med_name`, `med_ingredient`
- `age`, `gender`, `race`, `ethnicity`, `insurance`, `bmi`, `famhist`
- `meds`, `dx`, `procedures`, `labs`
- `label` (0/1)

For list-like fields, store them as JSON strings or pre-expanded strings; adapt parsing as needed.

### LLM citations (`mimic_attribution.py`)
CSV with at least:
- `instance_id`: unique id per test instance
- `feature`: cited feature name (already mapped to your final feature space)
- `probability`: model probability of discontinuation (0..1)
Optional:
- `predicted_label`: 0/1

## Example commands

### 1) Train and evaluate ML baselines
```bash
python train_ml_models.py --train train.parquet --test test.parquet --label-col label --outdir outputs_ml
```

### 2) Create fine-tuning JSONL
```bash
python gpt_prompts_and_finetune_jsonl.py --input finetune_input.csv --out train_gpt_finetune.jsonl --label-col label --mode classify
```

### 3) Compute mimic attribution scores
```bash
python mimic_attribution.py --citations citations.csv --outdir outputs_attr --topk 10
```

## Hyperparameters (as used in the paper)

- Logistic Regression (LR): penalty=l2, C=1.0, class_weight=balanced, solver=lbfgs, max_iter=1000
- Decision Tree (DT): max_depth=8, min_samples_split=20, min_samples_leaf=10, class_weight=balanced
- Random Forest (RF): n_estimators=200, max_depth=10, min_samples_split=10, min_samples_leaf=5, class_weight=balanced
- XGBoost (GB): n_estimators=200, max_depth=6, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8,
  scale_pos_weight = (#neg/#pos), eval_metric=logloss

## Data sharing

Underlying EHR data are not shareable due to patient privacy constraints. This package provides code and prompt templates only.
