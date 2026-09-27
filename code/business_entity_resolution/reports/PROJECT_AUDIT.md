# Project Audit: Amazon ML Challenge 2026 — Business Entity Resolution

## 1. Workspace Overview

The workspace contains the challenge dataset, official validator, problem instructions, and methodology template.

- **Workspace Root:** `/Users/pawasthi/Downloads/student_resource 2`
- **Challenge Directory Structure:**
  - `README.md` (14.1 KB): Official problem statement, evaluation metric (macro $F_{0.5}$), guidelines, and rules.
  - `Documentation_template.md` (2.2 KB): Required methodology report template.
  - `dataset/`
    - `train/`
      - `train_source1.tsv`: 200.34 MB (2,206,821 rows) — Reference dataset, deduplicated S1 entities.
      - `train_source2.tsv`: 466.63 MB (5,034,616 rows) — Noisy source S2 entities.
      - `train_source3.tsv`: 480.37 MB (5,285,603 rows) — Noisy source S3 entities.
      - `train_ground_truth.tsv`: 121.13 MB (2,206,821 rows) — Ground truth matches for S1 entities.
    - `test/`
      - `test_source1.tsv`: 166.91 MB (1,732,544 rows) — Test reference dataset.
      - `test_source2.tsv`: 485.86 MB (4,887,273 rows) — Test S2 dataset.
      - `test_source3.tsv`: 482.56 MB (5,082,316 rows) — Test S3 dataset.
  - `utils/`
    - `validate_submission.py` (13.7 KB): Official validation script.

## 2. File Schemas and Data Specifications

### Source Files (`*_source1.tsv`, `*_source2.tsv`, `*_source3.tsv`)
- Delimiter: Tab (`\t`), UTF-8 encoded.
- Columns (4):
  1. `entity_id` (string): Unique identifier prefixed with `S1-`, `S2-`, or `S3-`.
  2. `business_name` (string): Business entity name (may contain abbreviations, legal suffixes, typos, multilingual text).
  3. `business_address` (string): Street address, landmark, city, state, postal code.
  4. `country` (string): Country label (`US` and `India` in train; `US`, `India`, and `France` in test).

### Ground Truth File (`train_ground_truth.tsv`)
- Columns (2):
  1. `source1_entity_id` (string): S1 identifier.
  2. `matched_entity_ids` (string): Comma-separated list of matching S2 and S3 IDs (empty for singletons).

### Expected Output Deliverables
1. `output/matching_results.tsv`:
   - Columns: `source1_entity_id\tmatched_entity_ids`
   - Must contain all 1,732,544 test S1 IDs in exactly one row each.
2. `output/candidate_pairs.tsv`:
   - Columns: `source1_entity_id\tcandidate_entity_ids`
   - Must contain the final blocking candidate set evaluated by the matcher.
   - All IDs in `matching_results.tsv` must appear in `candidate_pairs.tsv`.
3. Self-contained code package under `code/business_entity_resolution/`:
   - `src/`: Modular Python modules.
   - `README.md`: Reproduction and execution guide.
   - `requirements.txt`: Pinned dependencies.
4. Completed `Documentation_template.md`.
5. Packaged submission ZIP.

## 3. Official Validation Criteria

The official validator `utils/validate_submission.py` enforces:
- Tab-separated format.
- Exact header matches (`source1_entity_id\tmatched_entity_ids` and `source1_entity_id\tcandidate_entity_ids`).
- Exact matching of all required $S_1$ entity IDs without duplicates or missing rows.
- No self-matches ($S_1$ IDs inside matched list).
- Only valid prefixes (`S2-`, `S3-`).
- Candidate containment: matched IDs should be subsets of candidates.
- Exit code 0 upon passing all checks.

## 4. Hardware and Computational Constraints

- Environment: macOS (Darwin arm64), 10 CPU cores, 16 GB Physical RAM.
- Scale: ~12.5M records in train, ~11.7M records in test.
- Memory Strategy: In-memory string overhead must be minimized. Use streaming/chunked processing, inverted token indexing with integer identifiers or compact hash tables, PyArrow/polars/custom compact representations, and disk-cached candidate generation if necessary.
- External Data Policy: Strictly forbidden. Only local compute and provided data.
