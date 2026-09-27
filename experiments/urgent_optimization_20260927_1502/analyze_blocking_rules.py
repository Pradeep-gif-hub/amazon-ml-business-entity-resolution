#!/usr/bin/env python3
"""
Phase 2: Comprehensive Candidate Generation & Blocking Rule Breakdown.
Analyzes each blocking channel's unique true match yield, redundancy,
and candidate count impact on the held-out validation split.
"""

import os
import sys
import time
import collections
import numpy as np

# Add project root to sys.path
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
ber_root = os.path.join(repo_root, "code/business_entity_resolution")
if ber_root not in sys.path:
    sys.path.insert(0, ber_root)

from src.config import default_config
from src.normalization import RecordProfile
from src.data_io import stream_source_tsv, load_ground_truth
from src.validation import create_stratified_validation_split

def main():
    print("=" * 80)
    print("PHASE 2: CANDIDATE GENERATION & BLOCKING RULE DEEP DIVE")
    print("=" * 80)
    
    cfg = default_config
    train_dir = cfg.train_dir
    
    # 1. Load Ground Truth
    print("\n[1/4] Loading ground truth matching labels...")
    gt_map = load_ground_truth(os.path.join(train_dir, "train_ground_truth.tsv"))
    
    # 2. Load Reference S1 Entities & create validation split
    print("\n[2/4] Loading Reference S1 entities...")
    s1_all = []
    s1_countries = {}
    for r in stream_source_tsv(os.path.join(train_dir, "train_source1.tsv")):
        s1_all.append(r)
        s1_countries[r.entity_id] = r.country
        if len(s1_all) >= 100000:
            break
            
    s1_dict = {r.entity_id: r for r in s1_all}
    all_s1_ids = list(s1_dict.keys())
    train_s1_ids, val_s1_ids = create_stratified_validation_split(
        all_s1_ids, s1_countries, gt_map, val_ratio=0.20, random_seed=cfg.random_seed
    )
    val_queries = [s1_dict[sid] for sid in val_s1_ids]
    print(f"Validation S1 Queries: {len(val_queries):,}")
    
    # 3. Load S2 and S3 target pools
    print("\n[3/4] Loading S2 and S3 target pools (2.4M records)...")
    s2_by_c = collections.defaultdict(list)
    s2_count = 0
    for r in stream_source_tsv(os.path.join(train_dir, "train_source2.tsv")):
        s2_by_c[r.country].append(r)
        s2_count += 1
        if s2_count >= 1200000:
            break
            
    s3_by_c = collections.defaultdict(list)
    s3_count = 0
    for r in stream_source_tsv(os.path.join(train_dir, "train_source3.tsv")):
        s3_by_c[r.country].append(r)
        s3_count += 1
        if s3_count >= 1200000:
            break
            
    target_pool_ids = {r.entity_id for recs in s2_by_c.values() for r in recs} | {r.entity_id for recs in s3_by_c.values() for r in recs}
    val_gt = {sid: (gt_map.get(sid, set()) & target_pool_ids) for sid in val_s1_ids}
    total_true_in_pool = sum(len(mids) for mids in val_gt.values())
    print(f"Total True Match Pairs in Pool for Validation: {total_true_in_pool:,}")
    
    # 4. Detailed Per-Rule Retrieval Instrumentation
    print("\n[4/4] Evaluating individual blocking rules on validation set...")
    
    # Build per-channel index structures for S2 and S3
    class InstrumentableCountryIndex:
        def __init__(self, country, records):
            self.country = country
            self.records = records
            total_recs = len(records)
            
            self.exact_core_index = collections.defaultdict(list)
            self.exact_nospace_index = collections.defaultdict(list)
            self.sorted_tokens_index = collections.defaultdict(list)
            self.first_two_tokens_index = collections.defaultdict(list)
            self.prefix4_index = collections.defaultdict(list)
            self.token_inverted_index = collections.defaultdict(list)
            self.addr_street_key_index = collections.defaultdict(list)
            
            token_freq = collections.Counter()
            for r in records:
                for t in r.name_tokens:
                    token_freq[t] += 1
            max_freq = max(100, int(total_recs * 0.003))
            self.stop_tokens = {t for t, count in token_freq.items() if count > max_freq}
            
            for i, r in enumerate(records):
                if r.core_name:
                    self.exact_core_index[r.core_name].append(i)
                    tokens = r.core_name.split()
                    if len(tokens) >= 2:
                        k2 = f"{tokens[0]}_{tokens[1]}"
                        self.first_two_tokens_index[k2].append(i)
                        sorted_k = "_".join(sorted(tokens))
                        self.sorted_tokens_index[sorted_k].append(i)
                    elif len(tokens) == 1:
                        self.first_two_tokens_index[tokens[0]].append(i)
                        
                    if len(r.core_name) >= 4:
                        self.prefix4_index[r.core_name[:4]].append(i)
                        
                if r.nospace_name and len(r.nospace_name) >= 3:
                    self.exact_nospace_index[r.nospace_name].append(i)
                    
                for t in r.name_tokens:
                    if t not in self.stop_tokens and len(t) >= 2:
                        self.token_inverted_index[t].append(i)
                        
                for sk in r.addr_street_keys:
                    self.addr_street_key_index[sk].append(i)
                    
        def retrieve_by_channel(self, query):
            """Returns {channel_name: {cand_id: score}}"""
            channels = {}
            
            # Rule 1: Exact Core Name
            r1 = {}
            if query.core_name and query.core_name in self.exact_core_index:
                for idx in self.exact_core_index[query.core_name]:
                    r1[self.records[idx].entity_id] = 14.0
            channels["Rule 1: Exact Core"] = r1
            
            # Rule 2: Sorted Tokens
            r2 = {}
            q_words = query.core_name.split()
            if len(q_words) >= 2:
                sorted_q = "_".join(sorted(q_words))
                if sorted_q in self.sorted_tokens_index:
                    for idx in self.sorted_tokens_index[sorted_q]:
                        r2[self.records[idx].entity_id] = 12.0
            channels["Rule 2: Sorted Tokens"] = r2
            
            # Rule 3: Exact Nospace Name
            r3 = {}
            if query.nospace_name and len(query.nospace_name) >= 4 and query.nospace_name in self.exact_nospace_index:
                for idx in self.exact_nospace_index[query.nospace_name]:
                    r3[self.records[idx].entity_id] = 11.0
            channels["Rule 3: Nospace Domain/Key"] = r3
            
            # Rule 4: First Two Tokens
            r4 = {}
            if len(q_words) >= 2:
                k2 = f"{q_words[0]}_{q_words[1]}"
                if k2 in self.first_two_tokens_index:
                    for idx in self.first_two_tokens_index[k2]:
                        r4[self.records[idx].entity_id] = 5.0
            elif len(q_words) == 1 and q_words[0] in self.first_two_tokens_index:
                for idx in self.first_two_tokens_index[q_words[0]]:
                    r4[self.records[idx].entity_id] = 3.5
            channels["Rule 4: First Two Tokens"] = r4
            
            # Rule 5: Token Inverted Index Overlap
            r5 = {}
            q_tokens = [t for t in query.name_tokens if t not in self.stop_tokens and len(t) >= 2]
            if q_tokens:
                token_cands = collections.Counter()
                for t in q_tokens:
                    matches = self.token_inverted_index.get(t, [])
                    if len(matches) <= 250:
                        for idx in matches:
                            token_cands[idx] += 1
                for idx, match_cnt in token_cands.most_common(25):
                    overlap_ratio = match_cnt / len(q_tokens)
                    r5[self.records[idx].entity_id] = 4.5 * overlap_ratio
            channels["Rule 5: Token Inverted Index"] = r5
            
            # Rule 6: Address Street Key Match
            r6 = {}
            for sk in query.addr_street_keys:
                if sk in self.addr_street_key_index:
                    matches = self.addr_street_key_index[sk]
                    if len(matches) <= 80:
                        for idx in matches:
                            r6[self.records[idx].entity_id] = 6.5
            channels["Rule 6: Street Address Keys"] = r6
            
            return channels
            
    # Build instrumented indices
    inst_indices_s2 = {}
    inst_indices_s3 = {}
    for c, recs in s2_by_c.items():
        inst_indices_s2[c] = InstrumentableCountryIndex(c, recs)
    for c, recs in s3_by_c.items():
        inst_indices_s3[c] = InstrumentableCountryIndex(c, recs)
        
    rule_names = [
        "Rule 1: Exact Core",
        "Rule 2: Sorted Tokens",
        "Rule 3: Nospace Domain/Key",
        "Rule 4: First Two Tokens",
        "Rule 5: Token Inverted Index",
        "Rule 6: Street Address Keys"
    ]
    
    rule_total_cands = collections.defaultdict(int)
    rule_true_matches = collections.defaultdict(int)
    rule_unique_true = collections.defaultdict(int)
    
    # Store for each query: true matches caught by each rule
    # and all candidate sets
    query_rule_cands = []
    
    for s1 in val_queries:
        c = s1.country
        true_mids = val_gt[s1.entity_id]
        
        cands_by_rule = {r: set() for r in rule_names}
        
        if c in inst_indices_s2:
            ch_s2 = inst_indices_s2[c].retrieve_by_channel(s1)
            for r in rule_names:
                cands_by_rule[r] |= set(ch_s2[r].keys())
                
        if c in inst_indices_s3:
            ch_s3 = inst_indices_s3[c].retrieve_by_channel(s1)
            for r in rule_names:
                cands_by_rule[r] |= set(ch_s3[r].keys())
                
        # Track statistics
        all_other_rules_matches = {}
        for r in rule_names:
            cands = cands_by_rule[r]
            rule_total_cands[r] += len(cands)
            caught = cands & true_mids
            rule_true_matches[r] += len(caught)
            
        for r in rule_names:
            caught_this = cands_by_rule[r] & true_mids
            caught_other = set()
            for other_r in rule_names:
                if other_r != r:
                    caught_other |= (cands_by_rule[other_r] & true_mids)
            unique_caught = caught_this - caught_other
            rule_unique_true[r] += len(unique_caught)
            
    print("\n" + "=" * 105)
    print(f"{'Blocking Channel / Rule':<30} | {'Total Pairs':<14} | {'True Caught':<14} | {'Recall %':<10} | {'Unique True':<14} | {'Avg Pairs/S1':<12}")
    print("-" * 105)
    for r in rule_names:
        tot_p = rule_total_cands[r]
        tc = rule_true_matches[r]
        rec = (tc / total_true_in_pool * 100) if total_true_in_pool else 0.0
        uniq = rule_unique_true[r]
        avg_p = tot_p / len(val_queries)
        print(f"{r:<30} | {tot_p:<14,} | {tc:<14,} | {rec:<10.2f}% | {uniq:<14,} | {avg_p:<12.2f}")
    print("=" * 105)
    
if __name__ == "__main__":
    main()
