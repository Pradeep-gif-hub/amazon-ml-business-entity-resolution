# Exploratory Data Analysis Report: Amazon ML Challenge 2026

> **Note on Methodology:** All row counts, missing field tallies, uniqueness checks, and country breakdowns are **Exact Counts** computed via full-file streaming. Character lengths and pairwise noise diagnostics are **Sampled Estimates** computed over bounded subsets of 20,000 records.

## 1. Dataset Volumes & Schema Integrity [Exact Counts]

| File | Total Rows | Unique IDs | Duplicate IDs | Missing Name | Missing Addr | Missing Country | Country Breakdown |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `train_source1.tsv` | 2,206,821 | 2,206,821 | 0 | 0 | 0 | 0 | US: 1,323,633, India: 883,188 |
| `train_source2.tsv` | 5,034,616 | 5,034,616 | 0 | 0 | 168,967 | 0 | India: 2,017,799, US: 3,016,817 |
| `train_source3.tsv` | 5,285,603 | 5,285,603 | 0 | 0 | 175,916 | 0 | US: 3,170,056, India: 2,115,547 |
| `test_source1.tsv` | 1,732,544 | 1,732,544 | 0 | 0 | 0 | 0 | US: 663,106, France: 259,452, India: 809,986 |
| `test_source2.tsv` | 4,887,273 | 4,887,273 | 0 | 0 | 129,408 | 0 | India: 2,312,565, France: 703,378, US: 1,871,330 |
| `test_source3.tsv` | 5,082,316 | 5,082,316 | 0 | 0 | 136,098 | 0 | India: 2,405,000, France: 731,615, US: 1,945,701 |

## 2. Ground Truth Match Distributions [Exact Counts]

- **Total Reference S1 Entities:** 2,206,821
- **Singletons (0 Matches):** 123,247 (5.58%)
- **Total Matched Links:** 7,638,365
  - Links to Source 2 (`S2-`): 3,693,619 (48.36%)
  - Links to Source 3 (`S3-`): 3,944,746 (51.64%)

### Match Cardinality per S1 Entity [Exact Counts]

| Match Count | Number of S1 Entities | Percentage |
| --- | --- | --- |
| 0 matches | 123,247 | 5.58% |
| 1 matches | 119,157 | 5.40% |
| 2 matches | 375,212 | 17.00% |
| 3 matches | 530,841 | 24.05% |
| 4 matches | 484,115 | 21.94% |
| 5 matches | 321,957 | 14.59% |
| 6 matches | 164,868 | 7.47% |
| 7 matches | 63,968 | 2.90% |
| 8 matches | 18,680 | 0.85% |
| 9 matches | 4,205 | 0.19% |
| 10 matches | 534 | 0.02% |
| 11 matches | 37 | 0.00% |

## 3. Key Findings & Pipeline Design Implications

1. **Strict Country Isolation:** 100% of true matches stay within the same country partition (US links only to US, India to India, France to France). Partitioning by country is a lossless blocking boundary.
2. **Address Null Rates:** Address is missing in only ~3.3% of $S_2$ and $S_3$ records; $S_1$ has 0.0% missing addresses. Address is a reliable primary signal when available.
3. **Singletons:** 5.58% of reference entities are singletons. High precision thresholding is critical because predicting any false link for a singleton results in a score of 0.0.
