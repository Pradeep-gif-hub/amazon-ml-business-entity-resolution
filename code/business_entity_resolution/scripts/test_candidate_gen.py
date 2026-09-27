#!/usr/bin/env python3
"""
Test script to benchmark candidate generation recall, index build time,
and query throughput on a representative sample of training data.
"""

import os
import sys
import time
import collections
from typing import Dict, List, Set, Tuple

# Ensure package root is in sys.path
pkg_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if pkg_root not in sys.path:
    sys.path.insert(0, pkg_root)

from src.normalization import (
    RecordProfile, normalize_business_name, extract_core_name,
    extract_name_tokens, normalize_business_address, extract_address_numbers
)
from src.data_io import stream_source_tsv, load_ground_truth

class BlockingIndex:
    def __init__(self, country: str):
        self.country = country
        self.records: List[RecordProfile] = []
        self.exact_core_index = collections.defaultdict(list)
        self.exact_clean_index = collections.defaultdict(list)
        self.first_two_tokens_index = collections.defaultdict(list)
        self.token_inverted_index = collections.defaultdict(list)
        self.addr_num_token_index = collections.defaultdict(list)
        self.token_freq = collections.Counter()
        self.stop_tokens = set()

    def add_records(self, records: List[RecordProfile]):
        start_idx = len(self.records)
        self.records.extend(records)
        
        # Build token frequencies first
        for i in range(start_idx, len(self.records)):
            r = self.records[i]
            for t in r.name_tokens:
                self.token_freq[t] += 1
                
        # Determine stop tokens (> 1% frequency or common words)
        total_recs = len(self.records)
        max_freq = max(200, int(total_recs * 0.01))
        self.stop_tokens = {t for t, count in self.token_freq.items() if count > max_freq}
        
        # Populate indexes
        for i in range(start_idx, len(self.records)):
            r = self.records[i]
            if r.core_name:
                self.exact_core_index[r.core_name].append(i)
                tokens = r.core_name.split()
                if len(tokens) >= 2:
                    k2 = f"{tokens[0]}_{tokens[1]}"
                    self.first_two_tokens_index[k2].append(i)
                elif len(tokens) == 1:
                    self.first_two_tokens_index[tokens[0]].append(i)
                    
            if r.clean_name and r.clean_name != r.core_name:
                self.exact_clean_index[r.clean_name].append(i)
                
            for t in r.name_tokens:
                if t not in self.stop_tokens and len(t) >= 3:
                    self.token_inverted_index[t].append(i)
                    
            # Address number + first 3 chars of core name
            first3 = r.core_name[:3] if len(r.core_name) >= 3 else r.core_name
            if first3:
                for num in r.addr_numbers:
                    self.addr_num_token_index[f"{num}_{first3}"].append(i)

    def retrieve_candidates(self, query: RecordProfile, max_candidates: int = 15) -> List[Tuple[RecordProfile, float]]:
        candidate_scores = collections.defaultdict(float)
        
        # 1. Exact core name match
        if query.core_name and query.core_name in self.exact_core_index:
            for idx in self.exact_core_index[query.core_name]:
                candidate_scores[idx] += 10.0
                
        # 2. Exact clean name match
        if query.clean_name and query.clean_name in self.exact_clean_index:
            for idx in self.exact_clean_index[query.clean_name]:
                candidate_scores[idx] += 8.0
                
        # 3. First two tokens match
        tokens = query.core_name.split()
        if len(tokens) >= 2:
            k2 = f"{tokens[0]}_{tokens[1]}"
            if k2 in self.first_two_tokens_index:
                for idx in self.first_two_tokens_index[k2]:
                    candidate_scores[idx] += 4.0
        elif len(tokens) == 1 and tokens[0] in self.first_two_tokens_index:
            for idx in self.first_two_tokens_index[tokens[0]]:
                candidate_scores[idx] += 3.0
                
        # 4. Token Inverted Index overlap
        q_tokens = [t for t in query.name_tokens if t not in self.stop_tokens and len(t) >= 3]
        if q_tokens:
            token_candidates = collections.Counter()
            for t in q_tokens:
                matches = self.token_inverted_index.get(t, [])
                if len(matches) <= 500:  # Avoid ultra-wide posting lists
                    for idx in matches:
                        token_candidates[idx] += 1
            for idx, match_cnt in token_candidates.most_common(50):
                overlap_ratio = match_cnt / max(1, len(q_tokens))
                candidate_scores[idx] += 3.0 * overlap_ratio
                
        # 5. Address Number + Prefix
        first3 = query.core_name[:3] if len(query.core_name) >= 3 else query.core_name
        if first3:
            for num in query.addr_numbers:
                k = f"{num}_{first3}"
                if k in self.addr_num_token_index:
                    for idx in self.addr_num_token_index[k]:
                        candidate_scores[idx] += 2.0
                        
        # Rank candidates by score
        if not candidate_scores:
            return []
            
        ranked = sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)[:max_candidates]
        return [(self.records[idx], score) for idx, score in ranked]

def main():
    print("Testing Candidate Generation on Training Sample...")
    train_dir = "/Users/pawasthi/Downloads/student_resource 2/dataset/train"
    
    # Load 50,000 S1 records
    print("Loading 20,000 S1 sample...")
    s1_samples = []
    for i, r in enumerate(stream_source_tsv(os.path.join(train_dir, "train_source1.tsv"))):
        if r.country == "US":  # Test on US partition first
            s1_samples.append(r)
        if len(s1_samples) >= 10000:
            break
            
    # Load Ground Truth
    gt_map = load_ground_truth(os.path.join(train_dir, "train_ground_truth.tsv"))
    
    # Load S2 and S3 for US
    print("Loading S2 records for US...")
    t0 = time.time()
    s2_us = []
    for r in stream_source_tsv(os.path.join(train_dir, "train_source2.tsv")):
        if r.country == "US":
            s2_us.append(r)
            if len(s2_us) >= 500000:  # 500k sample for fast benchmark
                break
    print(f"Loaded {len(s2_us):,} S2 records in {time.time() - t0:.2f}s")
    
    index_s2 = BlockingIndex("US")
    t0 = time.time()
    index_s2.add_records(s2_us)
    print(f"Built BlockingIndex on {len(s2_us):,} records in {time.time() - t0:.2f}s")
    
    # Evaluate recall on S1 queries
    s2_id_set = {r.entity_id for r in s2_us}
    
    total_true_s2_links = 0
    retrieved_true_s2_links = 0
    candidate_counts = []
    
    t0 = time.time()
    for s1 in s1_samples:
        true_mids = gt_map.get(s1.entity_id, set())
        # Filter true mids to those present in our S2 sample
        true_s2_in_sample = {m for m in true_mids if m.startswith("S2-") and m in s2_id_set}
        total_true_s2_links += len(true_s2_in_sample)
        
        candidates = index_s2.retrieve_candidates(s1, max_candidates=15)
        cand_ids = {c[0].entity_id for c in candidates}
        candidate_counts.append(len(cand_ids))
        
        if true_s2_in_sample:
            hits = len(true_s2_in_sample & cand_ids)
            retrieved_true_s2_links += hits
            
    query_time = time.time() - t0
    recall = (retrieved_true_s2_links / total_true_s2_links * 100) if total_true_s2_links else 0
    avg_cands = sum(candidate_counts) / len(candidate_counts) if candidate_counts else 0
    qps = len(s1_samples) / query_time
    
    print("\n--- Benchmark Results ---")
    print(f"Evaluated Queries: {len(s1_samples):,}")
    print(f"True S2 Matches in Sample Pool: {total_true_s2_links:,}")
    print(f"Retrieved S2 Matches: {retrieved_true_s2_links:,}")
    print(f"Candidate Recall (Ceiling): {recall:.2f}%")
    print(f"Avg Candidates per S1: {avg_cands:.2f}")
    print(f"Retrieval Throughput: {qps:,.1f} queries/second")

if __name__ == "__main__":
    main()
