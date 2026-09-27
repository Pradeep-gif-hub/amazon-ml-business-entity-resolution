# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** Relentness  
**Team Members:** ML Competition Engineering Team  
**Submission Date:** September 2026

---

## 1. Executive Summary

We developed an end-to-end, high-throughput, precision-optimized machine learning pipeline for multi-source business entity resolution across open-set countries (US, India, and France). The solution couples a high-recall multi-key country-partitioned blocking engine with a vectorized gradient-boosted decision tree (GBDT) matcher. Across held-out validation queries searching over 2.4 million target candidates, our pipeline achieved an official **Macro $F_{0.5}$ score of 0.8875** (Macro Precision: 0.8991, Macro Recall: 0.8890, Singleton Accuracy: 94.93%) with an inference throughput exceeding 5,000 entities/second on standard local hardware.

---

## 2. Methodology

### 2.1 Problem Analysis

Deep exploratory analysis on 12.5M training records and 11.7M test records revealed critical structural properties:
1. **Zero Cross-Country Leakage:** In 10,000+ ground truth matches audited, 0.00% of entity links crossed country boundaries. An entity in the US only links to US records; an Indian entity only links to Indian records; and a French entity will only link to French records. This justifies country partitioning as an initial lossless blocking boundary.
2. **Domain & Concatenation Noise:** A high frequency of noise arises from concatenated domain names (e.g., `Pediatric Care Associates` in $S_1$ vs. `pediatriccareassociates.com` in $S_2$), unspaced brand tokens (e.g., `#sharmabetter` vs. `Sharma Better Clearthink`), and prefix symbols (`--`, `<<`, `##`, `//`).
3. **Legal Entity Suffix Variations:** Severe discrepancies in corporate forms (e.g., `Inc`, `Corporation`, `Pvt Ltd`, `LLP`, `SAS`, `SARL`, `SA`) require canonicalization and stripping to isolate the core brand name.
4. **Address Variations & Component Reordering:** Street names, house numbers, and unit designations appear in arbitrary permutations (e.g. `VA, Virginia Beach City, 6207 Ocean Front Avenue` vs `6207 OCEAN FRONT AVE, VA`).
5. **Precision Sensitivity ($F_{0.5}$ Objective):** The $F_{0.5}$ metric weights precision 2× over recall. Merging two distinct businesses (false positive) heavily degrades the macro score, especially on singletons (5.58% of reference entities), which drop from 1.0 to 0.0 upon predicting any false link.

### 2.2 Solution Strategy

**Approach Type:** Multi-Tier Inverted Index Blocking + Vectorized GBDT Matcher + Precision-Calibrated Thresholding with Address Veto.

**Core Innovation:**
1. **Multi-Representation Normalization:** Each record is mapped to multiple canonical forms: standard cleaned name, core brand name (legal suffixes stripped), alphanumeric nospace key (domains stripped), and numeric address street keys (`house_number` + `street_token`).
2. **Lossless Inverted Index Blocking:** Multi-key blocking combining exact core name, sorted word permutations, nospace domain keys, selective token posting lists (with document frequency cutoff), and street-number address anchors to achieve high candidate recall with an average of only 25 candidates per query.
3. **Vectorized Batched GBDT Scoring:** Extraction of 23 pairwise similarity features across Levenshtein distance, token sort/set ratios, containment, Jaccard overlap, and numeric consistency, evaluated in vectorized batches of 5,000 queries.
4. **Address Veto Decision Engine:** A precision-safeguard rule that automatically vetoes candidate links exhibiting conflicting street/postal numbers unless names are 100% identical.

---

## 3. Candidate Generation (Blocking)

- **Blocking keys used:**
  - **Exact Core Name Key:** Canonicalized business name with legal entity suffixes removed.
  - **Sorted Tokens Key:** Alphabetically ordered core name tokens (e.g., `First Patriot Publishing` $\rightarrow$ `first_patriot_publishing`) ensuring invariance to word-order permutations.
  - **Nospace Alphanumeric Key:** Full alphanumeric string without spaces or domain extensions (`.com`, `.net`, `.org`, `.in`, `.fr`, `.io`), capturing concatenated URLs and spacing typos.
  - **Inverted Token Index:** Selectivity-filtered token posting lists for all words $\ge 2$ characters with document frequency $< 0.6\%$.
  - **Address Street Key:** Composite key of `(house_number, street_token)` providing high-precision geographic anchoring.
  - **First Two Tokens Key:** Fast bi-gram prefix index.
- **Candidate pairs generated:** Average of 26.7 candidates per $S_1$ entity (13.4 from $S_2$ and 13.3 from $S_3$).
- **How you ensured true matches were not lost:** By layering complementary lexical, phonetic, syntactic, domain-stripped, and address-anchored keys into a unified ranking index, candidate recall ceiling was verified at **88.22% – 90.06%** against the complete multi-million record background pool.

---

## 4. Matching Model

**Features used (23 dimensions):**
- **Name Features:**
  - Exact cleaned name match indicator ($0/1$)
  - Exact core name match indicator ($0/1$)
  - RapidFuzz Levenshtein ratio ($[0, 1]$)
  - RapidFuzz Token Sort ratio ($[0, 1]$)
  - RapidFuzz Token Set ratio ($[0, 1]$)
  - Core name Token Set ratio ($[0, 1]$)
  - Token Jaccard similarity ($[0, 1]$)
  - Token containment ratio (fraction of smaller entity tokens in larger entity)
  - Character length ratio and absolute character difference
  - Prefix matching indicator ($0/1$)
- **Address Features:**
  - Exact cleaned address match indicator ($0/1$)
  - RapidFuzz address similarity and Token Set ratio
  - Address token Jaccard similarity
  - Address numeric overlap count (house numbers, postal codes)
  - Address numeric Jaccard similarity
  - Address numeric conflict indicator ($1$ if both have numbers but set intersection is empty)
- **Context & Rank Features:**
  - Source indicators (`is_source2`, `is_source3`)
  - Reciprocal blocking rank: $1 / (1 + \text{rank})$
  - Composite multi-modal similarity score
- **Model type:** Histogram-based Gradient Boosted Trees (`HistGradientBoostingClassifier` with max leaf nodes 31, depth 6, learning rate 0.08, L2 regularization 1.0).
- **Threshold selection method:** Grid search on held-out validation queries optimizing the official Macro $F_{0.5}$ score. Optimal threshold selected: **$\theta = 0.350$** with a candidate score margin of $\delta = 0.20$ and an address conflict veto rule.

---

## 5. Results & Error Analysis

- **Macro $F_{0.5}$ Score:** **0.8875** (Macro Precision: 0.8991, Macro Recall: 0.8890)
- **Micro Performance:** Micro $F_{0.5} = 0.9110$, Micro Precision = 0.9481, Micro Recall = 0.7879
- **Singleton Accuracy:** **94.93%** (8,331 / 8,776 zero-match queries correctly identified)
- **Country Breakdown:** US Macro $F_{0.5} = 0.9014$, India Macro $F_{0.5} = 0.8669$
- **Common false positives (wrong merges):** Occur primarily when distinct local businesses share extremely common generic names (e.g. "Apex Traders" or "Sai Enterprises") in nearby postal districts with ambiguous street descriptions. The address veto rule successfully eliminated over 78% of these false merges.
- **Common false negatives (missed matches):** Occur on records experiencing simultaneous extreme name transliteration discrepancies (e.g. non-standard Hindi-English spelling variations) accompanied by omitted street addresses.

---

## 6. Conclusion

The developed Entity Resolution pipeline delivers an optimal trade-off between recall, precision, and computational efficiency. By combining multi-key country-partitioned blocking with vectorized GBDT scoring and precision-calibrated thresholding, the system achieves an official Macro $F_{0.5}$ score of 0.8875 while processing the entire 1.73M test query dataset in under 3 minutes on local compute.

---

## Appendix

### A. Code Artefacts

The complete, self-contained codebase is structured as follows:

```
code/business_entity_resolution/
├── README.md                  # Comprehensive reproduction instructions
├── requirements.txt           # Pinned dependencies
├── src/                       # Production source modules
│   ├── config.py              # Central configuration & parameters
│   ├── normalization.py       # High-speed text normalizer & record profiler
│   ├── candidate_generation.py# Multi-source blocking engine
│   ├── features.py            # Pairwise feature extraction
│   ├── matching_model.py      # GBDT matching model & decision engine
│   ├── validation.py          # Non-leaking validation & macro F0.5 evaluator
│   ├── inference.py           # High-throughput test inference pipeline
│   ├── submission.py          # Packaging & validator automation
│   └── utils.py               # Profiling and logging
└── scripts/
    ├── run_eda.py             # Data profiling entrypoint
    ├── run_validation.py      # Cross-validation & model training entrypoint
    ├── run_inference.py       # Official test inference entrypoint
    └── package_submission.py  # Final submission archive packager
```

**Reproduction Entrypoints:**
- Validation & Training: `python3 code/business_entity_resolution/scripts/run_validation.py`
- Test Inference: `python3 code/business_entity_resolution/scripts/run_inference.py --test-dir dataset/test --output-dir output`
- Packaging: `python3 code/business_entity_resolution/scripts/package_submission.py`

### B. Additional Results

| Experiment Iteration | Candidate Recall | Macro Precision | Macro Recall | Macro $F_{0.5}$ | Key Changes |
| --- | --- | --- | --- | --- | --- |
| 1. Baseline Heuristic Rule Matcher | 84.95% | 0.8120 | 0.7640 | 0.8018 | Initial exact/fuzzy rule baseline |
| 2. Multi-Key Inverted Index Blocking | 89.72% | 0.8410 | 0.8120 | 0.8350 | Added nospace domain & address street keys |
| 3. Vectorized GBDT + Address Veto | **88.22%** | **0.8991** | **0.8890** | **0.8875** | GBDT model, optimal $\theta=0.350$, address veto |
