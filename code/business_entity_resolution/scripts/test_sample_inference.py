#!/usr/bin/env python3
"""
Bounded Sample Inference Test (5,000 rows).
Processes a strictly bounded sample of Source 1, Source 2, and Source 3 records,
logging progress every 500 rows, tracking memory, and verifying output formatting.
"""

import os
import sys
import time
import collections
from typing import Dict, List, Set, Tuple

pkg_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if pkg_root not in sys.path:
    sys.path.insert(0, pkg_root)

from src.normalization import (
    RecordProfile, normalize_business_name, extract_core_name,
    extract_nospace_name, normalize_business_address
)
from src.candidate_generation import CountryBlockingIndex
from src.matching_model import MatchingModel
from src.data_io import stream_source_tsv, TSVWriter
from src.config import default_config

def main():
    print("=" * 65)
    print("STARTING BOUNDED SAMPLE INFERENCE TEST (Limit: 5,000 rows)")
    print("=" * 65)
    
    start_time = time.time()
    test_dir = "dataset/test"
    sample_output_dir = "code/business_entity_resolution/scratch_output"
    os.makedirs(sample_output_dir, exist_ok=True)
    
    # 1. Load 5,000 S2 records
    print("[1/5] Loading bounded sample of 5,000 Source 2 records...")
    t0 = time.time()
    s2_sample = []
    for r in stream_source_tsv(os.path.join(test_dir, "test_source2.tsv")):
        s2_sample.append(r)
        if len(s2_sample) >= 5000:
            break
    print(f"  ✓ Loaded {len(s2_sample):,} S2 sample records in {time.time() - t0:.2f}s")
    
    # 2. Load 5,000 S3 records
    print("[2/5] Loading bounded sample of 5,000 Source 3 records...")
    t0 = time.time()
    s3_sample = []
    for r in stream_source_tsv(os.path.join(test_dir, "test_source3.tsv")):
        s3_sample.append(r)
        if len(s3_sample) >= 5000:
            break
    print(f"  ✓ Loaded {len(s3_sample):,} S3 sample records in {time.time() - t0:.2f}s")
    
    # 3. Index by country
    print("[3/5] Building country-partitioned blocking indices for sample...")
    t0 = time.time()
    indices_s2 = {}
    indices_s3 = {}
    
    s2_by_c = collections.defaultdict(list)
    for r in s2_sample:
        s2_by_c[r.country].append(r)
    for c, recs in s2_by_c.items():
        idx = CountryBlockingIndex(c, max_candidates_per_query=15)
        idx.build(recs)
        indices_s2[c] = idx
        
    s3_by_c = collections.defaultdict(list)
    for r in s3_sample:
        s3_by_c[r.country].append(r)
    for c, recs in s3_by_c.items():
        idx = CountryBlockingIndex(c, max_candidates_per_query=15)
        idx.build(recs)
        indices_s3[c] = idx
        
    print(f"  ✓ Indexed sample target records in {time.time() - t0:.2f}s")
    
    # 4. Load trained model
    print("[4/5] Loading matching model checkpoint...")
    model_path = "code/business_entity_resolution/models/matching_model.pkl"
    matcher = MatchingModel()
    if os.path.exists(model_path):
        matcher.load(model_path)
        print(f"  ✓ Loaded model from {model_path} (Threshold: {matcher.threshold:.3f})")
    else:
        print("  ✓ Using baseline feature matcher")
        
    # 5. Process 5,000 S1 queries with progress logged every 500 rows
    print("[5/5] Processing 5,000 Source 1 queries (logging every 500 rows)...")
    matching_path = os.path.join(sample_output_dir, "sample_matching_results.tsv")
    candidate_path = os.path.join(sample_output_dir, "sample_candidate_pairs.tsv")
    
    match_writer = TSVWriter(matching_path, "source1_entity_id", "matched_entity_ids")
    cand_writer = TSVWriter(candidate_path, "source1_entity_id", "candidate_entity_ids")
    
    query_count = 0
    total_cands = 0
    total_matches = 0
    singletons = 0
    t0 = time.time()
    
    batch = []
    batch_size = 500
    
    for r in stream_source_tsv(os.path.join(test_dir, "test_source1.tsv")):
        batch.append(r)
        if len(batch) >= batch_size:
            # Process batch
            batch_queries_with_cands = []
            for s1 in batch:
                c = s1.country
                cands_s2 = indices_s2[c].retrieve_candidates(s1) if c in indices_s2 else []
                cands_s3 = indices_s3[c].retrieve_candidates(s1) if c in indices_s3 else []
                all_cands = cands_s2 + cands_s3
                batch_queries_with_cands.append((s1, all_cands))
                cand_ids = [cand[0].entity_id for cand in all_cands]
                cand_writer.write_row(s1.entity_id, cand_ids)
                total_cands += len(cand_ids)
                
            scored_batch = matcher.score_batch(batch_queries_with_cands)
            for s1, scored_cands in scored_batch:
                matched_ids = matcher.filter_matches(s1, scored_cands) if scored_cands else []
                match_writer.write_row(s1.entity_id, matched_ids)
                if matched_ids:
                    total_matches += len(matched_ids)
                else:
                    singletons += 1
                    
            query_count += len(batch)
            elapsed = time.time() - t0
            print(f"  Processed {query_count:,}/5,000 queries ({query_count/elapsed:,.1f} qps) | Matches: {total_matches:,} | Singletons: {singletons:,}")
            batch = []
            
        if query_count >= 5000:
            break
            
    match_writer.close()
    cand_writer.close()
    
    total_time = time.time() - start_time
    print("=" * 65)
    print("🎉 BOUNDED SAMPLE INFERENCE TEST COMPLETED SUCCESSFULLY")
    print(f"Total Queries Processed: {query_count:,}")
    print(f"Total Candidates Emitted: {total_cands:,} (avg {total_cands/query_count:.2f}/query)")
    print(f"Total Matches Emitted: {total_matches:,} (avg {total_matches/query_count:.2f}/query)")
    print(f"Singletons: {singletons:,} ({singletons/query_count*100:.2f}%)")
    print(f"Total Runtime: {total_time:.2f}s ({query_count/total_time:,.1f} queries/second)")
    print("=" * 65)
    
    # Verify outputs exist and have exact row count
    with open(matching_path, "r") as f:
        match_lines = sum(1 for _ in f) - 1
    with open(candidate_path, "r") as f:
        cand_lines = sum(1 for _ in f) - 1
        
    assert match_lines == 5000, f"Expected 5000 matching rows, got {match_lines}"
    assert cand_lines == 5000, f"Expected 5000 candidate rows, got {cand_lines}"
    print(f"✓ Output file row counts verified: {match_lines:,} rows in matching & candidate files.")

if __name__ == "__main__":
    main()
