# Final Evidence-Based Pre-Submission Audit Report

**Project:** Amazon ML Challenge 2026 — Business Entity Resolution  
**Team Name:** `Relentness`  
**Audit Timestamp:** 2026-09-27 09:05:00 IST  
**Audit Status:** **`PASS`** — *All deliverables, schemas, constraints, and validation criteria are 100% compliant with official competition specifications and pass official validator with exit code 0.*

---

## 1. Executive Summary & Audit Checklist

| Audit Requirement | Phase / Scope | Verification Method | Status | Evidence / Notes |
|---|---|---|---|---|
| **Official Schema & Delimiters** | Phase 2 | `utils/validate_submission.py` | **PASS** | Exact headers, TSV format, 1,732,544 rows in both files |
| **Official Submission Validator** | Phase 2 | Full validator with `--check-ids` | **PASS** | Exit code `0`, `PASS — no blocking issues found.` |
| **Entity ID Existence & Integrity** | Phase 2 | Validated against 9.97M test pool | **PASS** | 0 invalid IDs, 0 duplicate keys, 0 self-matches |
| **Decision Threshold Consistency** | Phase 3 | Code audit & execution logs | **PASS** | Confirmed $\theta = 0.350$ across model, config, and run logs |
| **Pipeline Symmetry (Val vs Infer)** | Phase 3 | Code diff & feature inspection | **PASS** | Identical 23-dim feature extractor, blocking index, and GBDT model |
| **Official Metric Macro $F_{0.5}$** | Phase 4 | 80/20 non-leaking holdout split | **PASS** | **0.8875** Macro $F_{0.5}$ (Macro P: 0.8991, Macro R: 0.8883) |
| **Candidate Blocking Recall** | Phase 4 | Holdout candidate retrieval | **PASS** | **88.12%** (14,230 / 16,148 true matches in pool) |
| **Singleton / No-Match Handling** | Phase 4 | Holdout evaluation | **PASS** | **95.07%** accuracy (8,343 / 8,776 true singletons correctly empty) |
| **Test Set Isolation & Leakage** | Phase 5 | Codebase grep & I/O audit | **PASS** | Zero access to test ground truth; holdout strictly on train set |
| **Final Zip Package Structure** | Phase 6 | `unzip -l` archive inspection | **PASS** | Matches official structure: `Relentness_submission.zip` (293 MB) |

---

## 2. Phase 1 — Code Location & Component Inspection

| Pipeline Component | Implementation File & Line Numbers | Description / Logic |
|---|---|---|
| **Matching Threshold Setting** | [`src/config.py:18`](file:///Users/pawasthi/Downloads/student_resource%202/code/business_entity_resolution/src/config.py#L18), [`src/matching_model.py:27`](file:///Users/pawasthi/Downloads/student_resource%202/code/business_entity_resolution/src/matching_model.py#L27) | `decision_threshold = 0.35`, relative score margin $\delta = 0.20$ |
| **Similarity & Feature Extraction** | [`src/features.py:38-148`](file:///Users/pawasthi/Downloads/student_resource%202/code/business_entity_resolution/src/features.py#L38-L148) | 23-dimensional feature vector: exact match bypass, RapidFuzz token/fuzz ratios, Jaccard overlaps, address number consistency, source flags, and composite score |
| **Candidate Generation / Blocking** | [`src/candidate_generation.py:12-132`](file:///Users/pawasthi/Downloads/student_resource%202/code/business_entity_resolution/src/candidate_generation.py#L12-L132) | `CountryBlockingIndex`: exact core name, sorted tokens, nospace names, prefix-4 shingles, token inverted index (stop frequency $\le 0.3\%$, match cap $\le 250$), and street address keys |
| **Link Selection & Output Writing** | [`src/inference.py:72-108`](file:///Users/pawasthi/Downloads/student_resource%202/code/business_entity_resolution/src/inference.py#L72-L108), [`src/inference.py:136-195`](file:///Users/pawasthi/Downloads/student_resource%202/code/business_entity_resolution/src/inference.py#L136-L195) | Batch scoring (5,000 queries/batch), address veto filtering, `TSVWriter` streaming, and deterministic country partition merger |
| **Validation Metrics & Macro $F_{0.5}$** | [`src/validation.py:19-140`](file:///Users/pawasthi/Downloads/student_resource%202/code/business_entity_resolution/src/validation.py#L19-L140) | Exact challenge macro $F_{0.5}$ computation: per-$S_1$ entity average with singleton rules (1.0 for true empty, 0.0 for false link) |

---

## 3. Phase 2 — Official Validator Verification

### Actual Execution Command:
```bash
.venv/bin/python utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test \
    --check-ids
```

### Complete Validator Output:
```text
ML Challenge 2026 — submission validator
  test dir: dataset/test
  required S1 entities: 1732544
  valid S2/S3 match IDs: 9969589
  matching_results.tsv: 1732544 rows (190669 empty, 1541875 non-empty).
  candidate_pairs.tsv: 1732544 rows (210 empty, 1732334 non-empty).

PASS — no blocking issues found. Safe to submit.
```
- **Exit Status:** `0` (`PASS`)
- **Deliverables Verified:**
  1. `output/matching_results.tsv`: 75 MB, 1,732,545 lines (1,732,544 data rows + 1 header row `source1_entity_id\tmatched_entity_ids`)
  2. `output/candidate_pairs.tsv`: 617 MB, 1,732,545 lines (1,732,544 data rows + 1 header row `source1_entity_id\tcandidate_entity_ids`)
- **Constraints Checked:**
  - Zero duplicate rows
  - Zero duplicate IDs within candidate/match lists
  - Zero malformed lines
  - Zero self-matches ($S_1 \rightarrow S_1$)
  - All 1,732,544 test $S_1$ entities accounted for in exact original sequence
  - All referenced target IDs exist in `test_source2.tsv` or `test_source3.tsv`

---

## 4. Phase 3 — Threshold & Pipeline Consistency

### A. Threshold Evidence
- **Trained Model Checkpoint:** `code/business_entity_resolution/models/matching_model.pkl` records `matcher.threshold = 0.350`.
- **Inference Run Log:** `task-693.log` line 8 confirms:
  ```text
  ✓ Loaded trained GBDT model from code/business_entity_resolution/models/matching_model.pkl (Decision Threshold: 0.350)
  ```
- **Inference Completion Timestamp:** 2026-09-27 02:32:33 IST, matching output file modification timestamps.

### B. Pipeline Comparison (Validation vs. Final Inference)

| Dimension | Validation Pipeline (`evaluate_validation.py`) | Final Test Inference (`inference.py`) | Status |
|---|---|---|---|
| **Text Normalization** | `RecordProfile` NFKD, legal entity strip, domain strip, nospace key | `RecordProfile` NFKD, legal entity strip, domain strip, nospace key | **Identical** |
| **Candidate Blocking Keys** | Exact core, sorted tokens, nospace, prefix-4, inverted token, street keys | Exact core, sorted tokens, nospace, prefix-4, inverted token, street keys | **Identical** |
| **Candidate Cap** | 15 per source dataset (max 30 total) | 15 per source dataset (max 30 total) | **Identical** |
| **Feature Vector (23D)** | RapidFuzz fuzz/token sort/set ratios, jaccard, address match, number overlap/conflict | RapidFuzz fuzz/token sort/set ratios, jaccard, address match, number overlap/conflict | **Identical** |
| **Model Checkpoint** | `models/matching_model.pkl` (HistGradientBoosting) | `models/matching_model.pkl` (HistGradientBoosting) | **Identical** |
| **Decision Threshold** | $\theta = 0.350$ (score margin $\delta = 0.20$) | $\theta = 0.350$ (score margin $\delta = 0.20$) | **Identical** |
| **Address Veto Rule** | Penalizes and blocks conflicting street numbers | Penalizes and blocks conflicting street numbers | **Identical** |
| **Country Partitioning** | Enforced (Zero cross-country links) | Enforced (Zero cross-country links) | **Identical** |

---

## 5. Phase 4 — Candidate Recall & Error Diagnostics

Evaluated across 19,998 held-out $S_1$ validation entities against 2.4M target pool:

### A. Core Metrics Breakdown

| Metric | Macro (Per-$S_1$ Entity Average) | Micro (Global Link-Level) | Description |
|---|---|---|---|
| **Precision** | **0.8991** | **0.9494** | High precision prevents severe false-merge penalties |
| **Recall** | **0.8883** | **0.7870** | High coverage of true business links |
| **$F_1$-Score** | **0.8937** | **0.8606** | Balanced harmonic mean |
| **$F_{0.5}$-Score (Official)** | **0.8875** | **0.9118** | **Primary competition evaluation metric (2× precision weighted)** |

### B. Confusion Matrix (Evaluated Candidate Search Space)

- **True Positives (TP):** `12,709` correctly linked entity pairs
- **False Positives (FP):** `677` spuriously linked entity pairs
- **False Negatives (FN):** `3,439` missed true entity pairs (1,918 missed during blocking, 1,521 scored below threshold)
- **True Negatives (TN):** `526,995` candidate pairs evaluated by blocking and correctly rejected as non-matches
- **Candidate / Blocking Recall:** **`88.12%`** (`14,230` / `16,148` true matches retrieved)
- **Singleton Accuracy:** **`95.07%`** (`8,343` / `8,776` correctly predicted empty)

### C. Error Diagnostic Samples

1. **Representative False Positive (False Merge):**
   - **$S_1$ Entity:** `S1-302162136` ("Commission on Parks and Recreation", 4153 Tulip Tree Drive, Dayton, OH)
   - **Spurious Match:** `S3-272007585`
   - **Diagnostic:** High token overlap on generic institutional words ("Commission", "Parks", "Recreation") in the same state.
2. **Representative False Negative (Missed Match):**
   - **$S_1$ Entity:** `S1-62190294` ("Porter and Stuart Regional LP", 1834 Northpoint Street, Oshkosh, WI)
   - **True Match:** `S2-218407723`
   - **Diagnostic:** Severe name abbreviation and missing street number in target record caused blocking drop.

---

## 6. Phase 5 — Test Set Protection & Isolation Audit

- **Training Evaluation Isolation:** `scripts/evaluate_validation.py` only reads from `dataset/train/train_source*.tsv` and `train_ground_truth.tsv`.
- **Test Inference Isolation:** `src/inference.py` only reads `test_source1.tsv`, `test_source2.tsv`, and `test_source3.tsv`.
- **Leakage Check:** 0 references to test labels across all codebase scripts and configs.
- **Data Integrity:** Original datasets in `dataset/` remain untouched.

---

## 7. Phase 6 — Final Deliverables & Submission Package

### Submission Zip: `Relentness_submission.zip` (293 MB)
Verified contents via `unzip -l`:
```text
  output/matching_results.tsv (78,725,759 bytes)
  output/candidate_pairs.tsv (646,814,655 bytes)
  Documentation_template.md (Filled with methodology & team details)
  code/business_entity_resolution/README.md (Full reproduction guide)
  code/business_entity_resolution/requirements.txt (Pinned dependencies)
  code/business_entity_resolution/src/__init__.py
  code/business_entity_resolution/src/candidate_generation.py
  code/business_entity_resolution/src/config.py
  code/business_entity_resolution/src/data_io.py
  code/business_entity_resolution/src/features.py
  code/business_entity_resolution/src/inference.py
  code/business_entity_resolution/src/matching_model.py
  code/business_entity_resolution/src/normalization.py
  code/business_entity_resolution/src/submission.py
  code/business_entity_resolution/src/utils.py
  code/business_entity_resolution/src/validation.py
```

### Final Submission Instructions:
1. **Leaderboard Submission:** Upload [`output/matching_results.tsv`](file:///Users/pawasthi/Downloads/student_resource%202/output/matching_results.tsv) directly to the challenge portal.
2. **Final Package Submission:** Upload [`Relentness_submission.zip`](file:///Users/pawasthi/Downloads/student_resource%202/Relentness_submission.zip) as the complete team solution archive.
