#!/usr/bin/env python3
"""
Diagnostic script to inspect missed matches in candidate generation
and engineer targeted blocking keys to push recall > 95%.
"""

import os
import sys
import collections

pkg_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if pkg_root not in sys.path:
    sys.path.insert(0, pkg_root)

from src.normalization import (
    RecordProfile, normalize_business_name, extract_core_name,
    extract_name_tokens, normalize_business_address, extract_address_numbers
)
from src.data_io import stream_source_tsv, load_ground_truth

def main():
    train_dir = "/Users/pawasthi/Downloads/student_resource 2/dataset/train"
    
    # Load 5,000 S1 records
    s1_dict = {}
    for r in stream_source_tsv(os.path.join(train_dir, "train_source1.tsv")):
        if r.country == "US":
            s1_dict[r.entity_id] = r
        if len(s1_dict) >= 5000:
            break
            
    gt_map = load_ground_truth(os.path.join(train_dir, "train_ground_truth.tsv"))
    
    # Load 300k S2 records
    s2_dict = {}
    for r in stream_source_tsv(os.path.join(train_dir, "train_source2.tsv")):
        if r.country == "US":
            s2_dict[r.entity_id] = r
        if len(s2_dict) >= 300000:
            break
            
    # Find all true pairs in this subset
    true_pairs = []
    for s1_id, s1 in s1_dict.items():
        mids = gt_map.get(s1_id, set())
        for m in mids:
            if m in s2_dict:
                true_pairs.append((s1, s2_dict[m]))
                
    print(f"Total true S1-S2 pairs in sample: {len(true_pairs)}")
    
    # Analyze why pairs match or don't match standard keys
    exact_core_matches = 0
    token_overlap_matches = 0
    first_two_matches = 0
    addr_num_matches = 0
    any_token_matches = 0
    clean_addr_matches = 0
    first_token_matches = 0
    char_3gram_matches = 0
    
    unmatched_pairs = []
    
    for s1, s2 in true_pairs:
        matched = False
        if s1.core_name == s2.core_name:
            exact_core_matches += 1
            matched = True
            
        t1, t2 = s1.name_tokens, s2.name_tokens
        if t1 & t2:
            any_token_matches += 1
            matched = True
            
        s1_words = s1.core_name.split()
        s2_words = s2.core_name.split()
        if s1_words and s2_words and s1_words[0] == s2_words[0]:
            first_token_matches += 1
            matched = True
            
        if s1.clean_addr and s2.clean_addr and s1.clean_addr == s2.clean_addr:
            clean_addr_matches += 1
            matched = True
            
        if s1.addr_numbers & s2.addr_numbers:
            addr_num_matches += 1
            
        if not matched:
            unmatched_pairs.append((s1, s2))
            
    print(f"Exact Core Match: {exact_core_matches}/{len(true_pairs)} ({exact_core_matches/len(true_pairs)*100:.2f}%)")
    print(f"Any Name Token Overlap: {any_token_matches}/{len(true_pairs)} ({any_token_matches/len(true_pairs)*100:.2f}%)")
    print(f"First Word Equal: {first_token_matches}/{len(true_pairs)} ({first_token_matches/len(true_pairs)*100:.2f}%)")
    print(f"Exact Address Match: {clean_addr_matches}/{len(true_pairs)} ({clean_addr_matches/len(true_pairs)*100:.2f}%)")
    print(f"Address Number Overlap: {addr_num_matches}/{len(true_pairs)} ({addr_num_matches/len(true_pairs)*100:.2f}%)")
    print(f"Unmatched Pairs Remaining: {len(unmatched_pairs)} ({len(unmatched_pairs)/len(true_pairs)*100:.2f}%)")
    
    print("\n--- Sample Unmatched Pairs Inspection ---")
    for s1, s2 in unmatched_pairs[:15]:
        print(f"S1 Name: '{s1.raw_name}' | Core: '{s1.core_name}' | Addr: '{s1.raw_addr}'")
        print(f"S2 Name: '{s2.raw_name}' | Core: '{s2.core_name}' | Addr: '{s2.raw_addr}'")
        print("-" * 60)

if __name__ == "__main__":
    main()
