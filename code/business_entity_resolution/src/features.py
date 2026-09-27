"""
Feature extraction pipeline for candidate entity pairs.
Computes fast string similarity metrics (Levenshtein, token sort, token set),
Jaccard overlap, numeric address consistency, and blocking signals.
"""

import math
from typing import List, Dict, Set, Tuple, Optional
import rapidfuzz.fuzz as rf_fuzz
from .normalization import RecordProfile

FEATURE_NAMES = [
    "name_exact_clean",
    "name_exact_core",
    "name_fuzz_ratio",
    "name_token_sort_ratio",
    "name_token_set_ratio",
    "name_core_token_set",
    "name_jaccard",
    "name_token_containment",
    "name_length_ratio",
    "name_length_diff",
    "name_prefix_equal",
    "addr_exact_clean",
    "addr_fuzz_ratio",
    "addr_token_set_ratio",
    "addr_jaccard",
    "addr_num_overlap_cnt",
    "addr_num_jaccard",
    "addr_num_conflict",
    "is_source2",
    "is_source3",
    "blocking_score",
    "blocking_rank_recip",
    "name_addr_composite"
]

def compute_pair_features(
    s1: RecordProfile,
    cand: RecordProfile,
    blocking_score: float = 0.0,
    blocking_rank: int = 0
) -> List[float]:
    """Extract numeric feature vector for an (S1, Candidate) pair."""
    # 1. Exact Name Matches
    name_exact_clean = 1.0 if s1.clean_name and s1.clean_name == cand.clean_name else 0.0
    name_exact_core = 1.0 if s1.core_name and s1.core_name == cand.core_name else 0.0
    
    # 2. Name Fuzzy Similarities & Overlap (Fast-path if exact match)
    if name_exact_clean == 1.0:
        name_fuzz_ratio = 1.0
        name_token_sort_ratio = 1.0
        name_token_set_ratio = 1.0
        name_core_token_set = 1.0
        name_jaccard = 1.0
        name_token_containment = 1.0
        name_length_ratio = 1.0
        name_length_diff = 0.0
        name_prefix_equal = 1.0
    else:
        # Use rapidfuzz for C++ accelerated Levenshtein / Token ratios
        name_fuzz_ratio = rf_fuzz.ratio(s1.clean_name, cand.clean_name) / 100.0 if s1.clean_name and cand.clean_name else 0.0
        name_token_sort_ratio = rf_fuzz.token_sort_ratio(s1.clean_name, cand.clean_name) / 100.0 if s1.clean_name and cand.clean_name else 0.0
        name_token_set_ratio = rf_fuzz.token_set_ratio(s1.clean_name, cand.clean_name) / 100.0 if s1.clean_name and cand.clean_name else 0.0
        name_core_token_set = 1.0 if name_exact_core == 1.0 else (rf_fuzz.token_set_ratio(s1.core_name, cand.core_name) / 100.0 if s1.core_name and cand.core_name else 0.0)
        
        n1, n2 = s1.name_tokens, cand.name_tokens
        if n1 and n2:
            inter = len(n1 & n2)
            union = len(n1 | n2)
            name_jaccard = inter / union if union > 0 else 0.0
            name_token_containment = inter / min(len(n1), len(n2))
        else:
            name_jaccard = 0.0
            name_token_containment = 0.0
            
        l1, l2 = len(s1.clean_name), len(cand.clean_name)
        max_l = max(l1, l2)
        name_length_ratio = min(l1, l2) / max_l if max_l > 0 else 0.0
        name_length_diff = float(abs(l1 - l2))
        
        c1, c2 = s1.core_name, cand.core_name
        name_prefix_equal = 1.0 if (len(c1) >= 4 and len(c2) >= 4 and (c1.startswith(c2) or c2.startswith(c1))) else 0.0
    
    # 5. Address Exact Match
    addr_exact_clean = 1.0 if s1.clean_addr and s1.clean_addr == cand.clean_addr else 0.0
    
    # 6. Address Fuzzy Similarities & Overlap
    if addr_exact_clean == 1.0:
        addr_fuzz_ratio = 1.0
        addr_token_set_ratio = 1.0
        addr_jaccard = 1.0
    elif s1.clean_addr and cand.clean_addr:
        addr_fuzz_ratio = rf_fuzz.ratio(s1.clean_addr, cand.clean_addr) / 100.0
        addr_token_set_ratio = rf_fuzz.token_set_ratio(s1.clean_addr, cand.clean_addr) / 100.0
        a1, a2 = s1.addr_tokens, cand.addr_tokens
        if a1 and a2:
            a_inter = len(a1 & a2)
            a_union = len(a1 | a2)
            addr_jaccard = a_inter / a_union if a_union > 0 else 0.0
        else:
            addr_jaccard = 0.0
    else:
        addr_fuzz_ratio = 0.0
        addr_token_set_ratio = 0.0
        addr_jaccard = 0.0
        
    # 8. Address Number Consistency (House number / postal code)
    num1, num2 = s1.addr_numbers, cand.addr_numbers
    if num1 and num2:
        num_inter = len(num1 & num2)
        num_union = len(num1 | num2)
        addr_num_overlap_cnt = float(num_inter)
        addr_num_jaccard = num_inter / num_union if num_union > 0 else 0.0
        addr_num_conflict = 1.0 if num_inter == 0 else 0.0
    else:
        addr_num_overlap_cnt = 0.0
        addr_num_jaccard = 0.0
        addr_num_conflict = 0.0
        
    # 9. Source Indicators
    is_source2 = 1.0 if cand.entity_id.startswith("S2-") else 0.0
    is_source3 = 1.0 if cand.entity_id.startswith("S3-") else 0.0
    
    # 10. Blocking Rank
    blocking_rank_recip = 1.0 / (1.0 + blocking_rank)
    
    # 11. Composite Score
    name_score = 0.4 * name_token_set_ratio + 0.3 * name_fuzz_ratio + 0.3 * name_core_token_set
    addr_score = 0.6 * addr_token_set_ratio + 0.4 * addr_fuzz_ratio if s1.clean_addr and cand.clean_addr else 0.5
    # If number conflict exists, penalize address score
    if addr_num_conflict > 0:
        addr_score *= 0.6
    name_addr_composite = name_score * (0.7 + 0.3 * addr_score)
    
    return [
        name_exact_clean,
        name_exact_core,
        name_fuzz_ratio,
        name_token_sort_ratio,
        name_token_set_ratio,
        name_core_token_set,
        name_jaccard,
        name_token_containment,
        name_length_ratio,
        name_length_diff,
        name_prefix_equal,
        addr_exact_clean,
        addr_fuzz_ratio,
        addr_token_set_ratio,
        addr_jaccard,
        addr_num_overlap_cnt,
        addr_num_jaccard,
        addr_num_conflict,
        is_source2,
        is_source3,
        float(blocking_score),
        blocking_rank_recip,
        name_addr_composite
    ]
