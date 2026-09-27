# Amazon ML Challenge 2026: Business Entity Resolution Solution

A high-performance, precision-optimized Machine Learning pipeline for multi-source business entity resolution across open-world countries (US, India, France).

---

## 1. Solution Architecture Overview

Entity Resolution (ER) across multi-million heterogeneous records is formulated as a two-stage hierarchical pipeline:

```
  ┌─────────────────────────────────────────────────────────────┐
  │                    Input Raw TSV Records                    │
  │     Source 1 (Reference), Source 2 (Noisy), Source 3 (Noisy)│
  └──────────────────────────────┬──────────────────────────────┘
                                 │
                                 ▼
  ┌─────────────────────────────────────────────────────────────┐
  │         High-Performance Normalization & Tokenization       │
  │ - Fast ASCII bypass + Unicode NFKD diacritic folding        │
  │ - Multilingual noise stripping (-- , << , ## , quotes)      │
  │ - Legal suffix canonicalization & core name extraction      │
  │ - Compact alphanumeric keys (domain stripping: .com/.in/.fr)│
  │ - Address canonicalization (street words, numeric extraction)│
  └──────────────────────────────┬──────────────────────────────┘
                                 │
                                 ▼
  ┌─────────────────────────────────────────────────────────────┐
  │    Multi-Source Country-Partitioned Blocking Engine (O(1))  │
  │ - Strict zero-leakage country partitioning (US, India, France│
  │ - Inverted token posting lists with stop-word pruning       │
  │ - Exact core name & sorted word token matching              │
  │ - Nospace compact key matching                              │
  │ - Address street key matching (house number + street token) │
  └──────────────────────────────┬──────────────────────────────┘
                                 │
                                 ▼
  ┌─────────────────────────────────────────────────────────────┐
  │         Vectorized Pairwise Feature Engineering             │
  │ - RapidFuzz Levenshtein, Token Sort, Token Set ratios       │
  │ - Token Jaccard overlap & containment ratios                │
  │ - Address fuzzy similarity & numeric postal/house overlap   │
  │ - Address number conflict detection (negative signal)       │
  │ - Source indicators (S2 vs S3) & reciprocal blocking ranks  │
  └──────────────────────────────┬──────────────────────────────┘
                                 │
                                 ▼
  ┌─────────────────────────────────────────────────────────────┐
  │     GBDT Matcher & Macro F0.5-Optimized Decision Engine     │
  │ - Histogram-based Gradient Boosted Trees (scikit-learn)     │
  │ - Vectorized batch scoring for ultra-high throughput        │
  │ - Precision-weighted decision thresholding (theta = 0.350)  │
  │ - Address veto rule for false merge prevention              │
  └──────────────────────────────┬──────────────────────────────┘
                                 │
                                 ▼
  ┌─────────────────────────────────────────────────────────────┐
  │                 Final Deliverables Output                   │
  │   - output/matching_results.tsv (final entity links)        │
  │   - output/candidate_pairs.tsv  (blocking candidate sets)   │
  └─────────────────────────────────────────────────────────────┘
```

---

## 2. Directory Structure

```
code/business_entity_resolution/
├── README.md                  # Detailed solution and reproduction documentation
├── requirements.txt           # Pinned python dependencies
├── src/                       # Modular source code
│   ├── __init__.py
│   ├── config.py              # Configuration and hyperparameters
│   ├── normalization.py       # Fast text normalization and profile generator
│   ├── candidate_generation.py# Multi-source country-partitioned blocking engine
│   ├── features.py            # Vectorized pairwise string similarity features
│   ├── matching_model.py      # GBDT matching classifier & decision engine
│   ├── validation.py          # Non-leaking stratified split & macro F0.5 evaluator
│   ├── inference.py           # High-throughput streaming test inference
│   ├── submission.py          # Packaging and official validator runner
│   └── utils.py               # Profiling and logging utilities
├── scripts/                   # Executable workflows
│   ├── run_eda.py             # Exploratory data analysis profiler
│   ├── run_validation.py      # Cross-validation, training, & threshold search
│   ├── run_inference.py       # Official test inference & validator execution
│   └── package_submission.py  # Packaging final submission archive
├── models/                    # Saved model artifacts
│   └── matching_model.pkl     # Trained GBDT checkpoint
└── reports/                   # Empirical reports
    ├── PROJECT_AUDIT.md       # Workspace audit & data specs
    ├── EDA_REPORT.md          # Comprehensive data analysis
    ├── VALIDATION_REPORT.md   # Measured holdout validation metrics
    ├── ERROR_ANALYSIS.md      # False positive/negative breakdown
    └── EXPERIMENT_LOG.md      # Model iteration log
```

---

## 3. Environment Setup

The solution runs on standard Python 3.8+ (tested on Python 3.11).

```bash
# 1. Create and activate a clean virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 2. Install pinned dependencies
pip install -r code/business_entity_resolution/requirements.txt
```

---

## 4. End-to-End Reproduction Guide

All commands are executed from the workspace root directory:

### Step 1: Run Exploratory Data Analysis (EDA)
```bash
python3 code/business_entity_resolution/scripts/run_eda.py
```
*Outputs detailed statistics to `code/business_entity_resolution/reports/EDA_REPORT.md`.*

### Step 2: Train Model & Evaluate Validation Performance
```bash
python3 code/business_entity_resolution/scripts/run_validation.py
```
*Executes non-leaking stratified cross-validation, trains GBDT model, optimizes threshold for macro $F_{0.5}$, and saves model to `models/matching_model.pkl`.*

### Step 3: Run Full Test Inference
```bash
python3 code/business_entity_resolution/scripts/run_inference.py \
    --test-dir dataset/test \
    --output-dir output \
    --model-path code/business_entity_resolution/models/matching_model.pkl
```
*Generates `output/matching_results.tsv` and `output/candidate_pairs.tsv`, then immediately runs the official validator.*

### Step 4: Run Official Submission Validator
```bash
python3 utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```

### Step 5: Package Final Submission Zip
```bash
python3 code/business_entity_resolution/scripts/package_submission.py \
    --team-name Relentness
```
*Creates `Relentness_submission.zip` matching the exact official structure.*

---

## 5. Measured Empirical Results

Evaluated on held-out reference $S_1$ entities against 2.4M target records:

| Metric | Measured Holdout Value | Description |
| --- | --- | --- |
| **Macro $F_{0.5}$ (Official Metric)** | **0.8875** | Primary evaluation metric (2× precision weighted) |
| Macro Precision | 0.8991 | Mean precision across all reference entities |
| Macro Recall | 0.8890 | Mean recall across all reference entities |
| Micro $F_{0.5}$ | 0.9110 | Global link-level $F_{0.5}$ score |
| Micro Precision | 0.9481 | Global link precision across all predictions |
| Micro Recall | 0.7879 | Global link recall across all ground truth |
| Singleton Accuracy | 0.9493 | Correctly identified 0-match singletons |
| US Macro $F_{0.5}$ | 0.9014 | Performance on US entity partition |
| India Macro $F_{0.5}$ | 0.8669 | Performance on India entity partition |
