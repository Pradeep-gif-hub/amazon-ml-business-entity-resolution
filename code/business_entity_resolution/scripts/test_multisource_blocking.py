#!/usr/bin/env python3
"""
Test candidate generation on S2 and S3 simultaneously and measure combined recall.
"""

import os
import sys
import time
import collections

pkg_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if pkg_root not in sys.path:
    sys.path.insert(0, pkg_root)

from src.normalization import RecordProfile
from src.candidate_generation import MultiSourceBlockingEngine
from src.data_io import stream_source_tsv, load_ground_truth

def main():
    train_dir = "/Users/pawasthi/Downloads/student_resource 2/dataset/train"
    print("Testing MultiSourceBlockingEngine on S2 and S3...")
    
    # Load 10,000 S1 records (sample across US and India)
    s1_sample = []
    for r in stream_source_tsv(os.path.join(train_dir, "train_source1.tsv")):
        s1_sample.append(r)
        if len(s1_sample) >= 10000:
            break
            
    gt_map = load_ground_truth(os.path.join(train_dir, "train_ground_truth.tsv"))
    
    # Load S2 and S3 records
    s2_by_c = collections.defaultdict(list)
    for r in stream_source_tsv(os.path.join(train_dir, "train_source2.tsv")):
        s2_by_c[r.country].append(r)
        if len(s2_by_c["US"]) >= 400000 and len(s2_by_c["India"]) >= 300000:
            break
            
    s3_by_c = collections.defaultdict(list)
    for r in stream_source_tsv(os.path.join(train_dir, "train_source3.tsv")):
        s3_by_c[r.country].append(r)
        if len(s3_by_c["US"]) >= 400000 and len(s3_by_c["India"]) >= 300000:
            break
            
    engine = MultiSourceBlockingEngine(max_cands_per_source=15)
    t0 = time.time()
    engine.index_target_records(s2_by_c, s3_by_c)
    print(f"Built MultiSourceBlockingEngine in {time.time() - t0:.2f}s")
    
    # Known IDs in target pool
    target_ids = {r.entity_id for recs in s2_by_c.values() for r in recs} | {r.entity_id for recs in s3_by_c.values() for r in recs}
    
    total_true_matches_in_pool = 0
    retrieved_matches = 0
    cand_count_per_s1 = []
    
    t0 = time.time()
    for s1 in s1_sample:
        true_mids = gt_map.get(s1.entity_id, set())
        true_in_pool = true_mids & target_ids
        total_true_matches_in_pool += len(true_in_pool)
        
        cands = engine.retrieve_for_s1(s1)
        cand_ids = {c[0].entity_id for c in cands}
        cand_count_per_s1.append(len(cand_ids))
        
        if true_in_pool:
            retrieved_matches += len(true_in_pool & cand_ids)
            
    query_time = time.time() - t0
    recall = (retrieved_matches / total_true_matches_in_pool * 100) if total_true_matches_in_pool else 0
    avg_cands = sum(cand_count_per_s1) / len(cand_count_per_s1) if cand_count_per_s1 else 0
    qps = len(s1_sample) / query_time
    
    print("\n--- Multi-Source Blocking Engine Results ---")
    print(f"Evaluated Queries: {len(s1_sample):,}")
    print(f"True Matches in Pool: {total_true_matches_in_pool:,}")
    print(f"Retrieved Matches: {retrieved_matches:,}")
    print(f"Candidate Recall (Ceiling): {recall:.2f}%")
    print(f"Avg Candidates per S1: {avg_cands:.2f}")
    print(f"Throughput: {qps:,.1f} queries/second")

if __name__ == "__main__":
    main()
