#!/usr/bin/env python3
"""
Unit and Integration Smoke Test for Amazon ML Challenge 2026 Pipeline.
Performs fast, bounded tests (< 5 seconds) on 1,000-row samples to verify:
1. TSV parsing and schema integrity
2. Text normalization and key extraction
3. Inverted index candidate generation
4. Feature extraction (23 dimensions)
5. Model training and inference
6. Official Macro F0.5 metric computation
7. Output schema compliance
"""

import os
import sys
import numpy as np

# Ensure package root is in sys.path
pkg_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if pkg_root not in sys.path:
    sys.path.insert(0, pkg_root)

from src.normalization import (
    RecordProfile, normalize_business_name, extract_core_name,
    extract_nospace_name, normalize_business_address, extract_address_numbers
)
from src.candidate_generation import CountryBlockingIndex, MultiSourceBlockingEngine
from src.features import compute_pair_features, FEATURE_NAMES
from src.matching_model import MatchingModel
from src.validation import compute_f05_score, evaluate_entity_resolution_predictions
from src.data_io import stream_source_tsv, load_ground_truth

def test_tsv_parsing():
    print("[Smoke Test 1/7] Testing TSV streaming and schema on 1,000-row slices...")
    train_dir = "/Users/pawasthi/Downloads/student_resource 2/dataset/train"
    
    s1_sample = []
    for i, r in enumerate(stream_source_tsv(os.path.join(train_dir, "train_source1.tsv"))):
        assert r.entity_id.startswith("S1-"), f"Invalid S1 prefix: {r.entity_id}"
        assert r.country in {"US", "India"}, f"Unexpected train country: {r.country}"
        assert len(r.raw_name) > 0, "Empty S1 business name"
        s1_sample.append(r)
        if len(s1_sample) >= 1000:
            break
            
    assert len(s1_sample) == 1000, f"Expected 1000 rows, got {len(s1_sample)}"
    print(f"  ✓ S1 parsed 1,000 records successfully (Countries: {set(r.country for r in s1_sample)})")
    
    s2_sample = []
    for i, r in enumerate(stream_source_tsv(os.path.join(train_dir, "train_source2.tsv"))):
        assert r.entity_id.startswith("S2-"), f"Invalid S2 prefix: {r.entity_id}"
        s2_sample.append(r)
        if len(s2_sample) >= 1000:
            break
    assert len(s2_sample) == 1000, f"Expected 1000 rows, got {len(s2_sample)}"
    print(f"  ✓ S2 parsed 1,000 records successfully")
    return s1_sample, s2_sample

def test_normalization():
    print("[Smoke Test 2/7] Testing text normalization, domain stripping, and nospace keys...")
    
    # 1. Accent removal & noise stripping
    raw1 = "<< LLC Moncada Léarning Center --"
    clean1 = normalize_business_name(raw1)
    core1 = extract_core_name(raw1)
    nospace1 = extract_nospace_name(core1)
    assert "learning" in clean1, f"Accent folding failed: {clean1}"
    assert core1 == "moncada learning center", f"Core name unexpected: {core1}"
    assert nospace1 == "moncadalearningcenter", f"Nospace unexpected: {nospace1}"
    
    # 2. Domain stripping
    raw2 = "pediatriccareassociates.com"
    core2 = extract_core_name(raw2)
    nospace2 = extract_nospace_name(core2)
    assert nospace2 == "pediatriccareassociates", f"Domain strip failed: {nospace2}"
    
    # 3. Address normalization
    raw_addr = "13834-A Willowtwist Saint, Houston, Texas"
    clean_addr = normalize_business_address(raw_addr)
    numbers = extract_address_numbers(raw_addr)
    assert "st" in clean_addr.split(), f"Street abbrev failed: {clean_addr}"
    assert "13834a" in numbers or "13834" in numbers, f"Number extraction failed: {numbers}"
    
    print("  ✓ Normalization logic passed all assertions.")

def test_blocking_engine(s1_records, s2_records):
    print("[Smoke Test 3/7] Testing CountryBlockingIndex and candidate retrieval...")
    idx = CountryBlockingIndex("US", max_candidates_per_query=10)
    idx.build(s2_records)
    
    query = s1_records[0]
    cands = idx.retrieve_candidates(query)
    assert isinstance(cands, list), "Candidate output must be a list"
    assert len(cands) <= 10, f"Candidate count {len(cands)} exceeds max 10"
    print(f"  ✓ Retrieved {len(cands)} candidates for query '{query.raw_name}' in {query.country}")

def test_feature_extraction(s1_records, s2_records):
    print("[Smoke Test 4/7] Testing pairwise feature extraction (23 dimensions)...")
    feats = compute_pair_features(s1_records[0], s2_records[0], blocking_score=5.0, blocking_rank=0)
    assert len(feats) == 23, f"Expected 23 features, got {len(feats)}"
    assert all(np.isfinite(f) for f in feats), f"Non-finite feature detected: {feats}"
    print(f"  ✓ 23 features computed successfully: {[round(f, 3) for f in feats[:6]]}...")

def test_matching_model():
    print("[Smoke Test 5/7] Testing GBDT model fit, predict_proba, and vectorized score_batch...")
    # Generate synthetic training pairs
    X = np.random.rand(200, 23).astype(np.float32)
    y = np.random.randint(0, 2, size=200).astype(np.int32)
    
    model = MatchingModel()
    model.fit(X, y)
    probs = model.predict_proba(X[:10])
    assert len(probs) == 10, "Probability output dimension mismatch"
    assert all(0.0 <= p <= 1.0 for p in probs), "Probabilities out of [0, 1] range"
    print("  ✓ Model trained and predicted probabilities successfully.")

def test_metric_computation():
    print("[Smoke Test 6/7] Testing macro F0.5 metric calculation & singletons...")
    gt = {
        "S1-1": {"S2-10", "S3-20"},
        "S1-2": {"S2-30"},
        "S1-3": set(),            # True singleton
        "S1-4": set()             # True singleton
    }
    pred = {
        "S1-1": {"S2-10", "S3-20"},  # Perfect: F0.5 = 1.0
        "S1-2": {"S2-30", "S3-99"},  # Precision 1/2, Recall 1/1 -> F0.5 = (1.25*0.5*1)/(0.25*0.5+1) = 0.5556
        "S1-3": set(),               # Correct singleton: F0.5 = 1.0
        "S1-4": {"S2-88"}            # False merge on singleton: F0.5 = 0.0
    }
    
    # Expected macro score = (1.0 + 0.555555 + 1.0 + 0.0) / 4 = 0.638888
    res = evaluate_entity_resolution_predictions(gt, pred)
    expected_f05 = (1.0 + (1.25 * 0.5 * 1.0) / (0.25 * 0.5 + 1.0) + 1.0 + 0.0) / 4.0
    assert abs(res["macro_f05"] - expected_f05) < 1e-4, f"Metric mismatch: {res['macro_f05']} vs {expected_f05}"
    assert res["singleton_accuracy"] == 0.5, f"Singleton accuracy mismatch: {res['singleton_accuracy']}"
    print(f"  ✓ Macro F0.5 verified exactly: {res['macro_f05']:.4f} (Singleton Acc: {res['singleton_accuracy']:.2f})")

def test_address_veto():
    print("[Smoke Test 7/7] Testing address veto rule for false merge prevention...")
    s1 = RecordProfile("S1-99", "Generic Hardware Traders", "105 Elm Street, City A", "US")
    cand_conflict = RecordProfile("S2-99", "Generic Hardware Traders LLC", "999 Oak Avenue, City B", "US")
    
    matcher = MatchingModel()
    matcher.threshold = 0.50
    # Candidate with address conflict (different numbers: 105 vs 999) and non-exact raw name
    scored = [(cand_conflict, 0.58)]
    mids = matcher.filter_matches(s1, scored, threshold=0.50)
    # The veto rule checks name_exact (core is identical here, so let's test different core)
    s1_diff_name = RecordProfile("S1-99", "Alpha Hardware Solutions", "105 Elm Street, City A", "US")
    cand_diff_name = RecordProfile("S2-99", "Beta Hardware Store", "999 Oak Avenue, City B", "US")
    mids_diff = matcher.filter_matches(s1_diff_name, [(cand_diff_name, 0.58)], threshold=0.50)
    assert len(mids_diff) == 0, f"Address veto failed: {mids_diff}"
    print("  ✓ Address veto successfully blocked conflicting candidates.")

def main():
    print("=" * 65)
    print("STARTING COMPLETE END-TO-END SMOKE TEST")
    print("=" * 65)
    s1_sample, s2_sample = test_tsv_parsing()
    test_normalization()
    test_blocking_engine(s1_sample, s2_sample)
    test_feature_extraction(s1_sample, s2_sample)
    test_matching_model()
    test_metric_computation()
    test_address_veto()
    print("=" * 65)
    print("🎉 ALL 7 SMOKE TESTS PASSED SUCCESSFULLY WITH ZERO ERRORS!")
    print("=" * 65)

if __name__ == "__main__":
    main()
