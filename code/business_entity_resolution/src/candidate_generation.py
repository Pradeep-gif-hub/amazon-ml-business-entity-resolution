"""
High-recall, high-throughput candidate generation and blocking engine.
Partitions records by country, indexes multiple high-precision keys,
including sorted core name tokens, nospace keys, prefix-4 shingles,
address street numbers, and token inverted indices.
"""

import collections
from typing import Dict, List, Set, Tuple, Optional
from .normalization import RecordProfile

class CountryBlockingIndex:
    """Inverted index and multi-key blocking engine for a single country partition."""
    
    def __init__(self, country: str, max_candidates_per_query: int = 20):
        self.country = country
        self.max_candidates = max_candidates_per_query
        self.records: List[RecordProfile] = []
        
        # Key to list of record indices
        self.exact_core_index = collections.defaultdict(list)
        self.exact_nospace_index = collections.defaultdict(list)
        self.sorted_tokens_index = collections.defaultdict(list)
        self.first_two_tokens_index = collections.defaultdict(list)
        self.prefix4_index = collections.defaultdict(list)
        self.token_inverted_index = collections.defaultdict(list)
        self.addr_street_key_index = collections.defaultdict(list)
        
        self.token_freq = collections.Counter()
        self.stop_tokens = set()
        
    def build(self, records: List[RecordProfile]) -> None:
        """Build blocking indices over the target records."""
        self.records = records
        total_recs = len(records)
        
        # 1. Compute token frequencies
        for r in records:
            for t in r.name_tokens:
                self.token_freq[t] += 1
                
        # Cut off tokens appearing in > 0.3% of corpus (stop words like 'center', 'shop', 'services')
        max_freq = max(100, int(total_recs * 0.003))
        self.stop_tokens = {t for t, count in self.token_freq.items() if count > max_freq}
        
        # 2. Populate inverted indices
        for i, r in enumerate(records):
            if r.core_name:
                self.exact_core_index[r.core_name].append(i)
                tokens = r.core_name.split()
                if len(tokens) >= 2:
                    k2 = f"{tokens[0]}_{tokens[1]}"
                    self.first_two_tokens_index[k2].append(i)
                    # Sorted tokens key for word-order permutation invariance
                    sorted_k = "_".join(sorted(tokens))
                    self.sorted_tokens_index[sorted_k].append(i)
                elif len(tokens) == 1:
                    self.first_two_tokens_index[tokens[0]].append(i)
                    
                # Prefix 4-gram
                if len(r.core_name) >= 4:
                    self.prefix4_index[r.core_name[:4]].append(i)
                    
            if r.nospace_name and len(r.nospace_name) >= 3:
                self.exact_nospace_index[r.nospace_name].append(i)
                
            for t in r.name_tokens:
                if t not in self.stop_tokens and len(t) >= 2:
                    self.token_inverted_index[t].append(i)
                    
            for sk in r.addr_street_keys:
                self.addr_street_key_index[sk].append(i)
                
    def retrieve_candidates(self, query: RecordProfile) -> List[Tuple[RecordProfile, float]]:
        """Retrieve top ranked candidate records for a query entity."""
        candidate_scores = collections.defaultdict(float)
        
        # 1. Exact Core Name Match
        if query.core_name and query.core_name in self.exact_core_index:
            for idx in self.exact_core_index[query.core_name]:
                candidate_scores[idx] += 14.0
                
        # 2. Sorted Name Tokens Match (word-order transpositions)
        q_words = query.core_name.split()
        if len(q_words) >= 2:
            sorted_q = "_".join(sorted(q_words))
            if sorted_q in self.sorted_tokens_index:
                for idx in self.sorted_tokens_index[sorted_q]:
                    candidate_scores[idx] += 12.0
                
        # 3. Exact Nospace Name Match (catches domains, concatenated names, spacing differences)
        if query.nospace_name and len(query.nospace_name) >= 4 and query.nospace_name in self.exact_nospace_index:
            for idx in self.exact_nospace_index[query.nospace_name]:
                candidate_scores[idx] += 11.0
                
        # 4. First Two Tokens Match
        if len(q_words) >= 2:
            k2 = f"{q_words[0]}_{q_words[1]}"
            if k2 in self.first_two_tokens_index:
                for idx in self.first_two_tokens_index[k2]:
                    candidate_scores[idx] += 5.0
        elif len(q_words) == 1 and q_words[0] in self.first_two_tokens_index:
            for idx in self.first_two_tokens_index[q_words[0]]:
                candidate_scores[idx] += 3.5
                
        # 5. Token Inverted Index Overlap
        q_tokens = [t for t in query.name_tokens if t not in self.stop_tokens and len(t) >= 2]
        if q_tokens:
            token_candidates = collections.Counter()
            for t in q_tokens:
                matches = self.token_inverted_index.get(t, [])
                if len(matches) <= 250:
                    for idx in matches:
                        token_candidates[idx] += 1
            for idx, match_cnt in token_candidates.most_common(25):
                overlap_ratio = match_cnt / len(q_tokens)
                candidate_scores[idx] += 4.5 * overlap_ratio
                
        # 6. Address Street Key Match (House Number + Street Token)
        for sk in query.addr_street_keys:
            if sk in self.addr_street_key_index:
                matches = self.addr_street_key_index[sk]
                if len(matches) <= 80:  # High-precision address bucket
                    for idx in matches:
                        candidate_scores[idx] += 6.5
                        
        if not candidate_scores:
            return []
            
        ranked = sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)[:self.max_candidates]
        return [(self.records[idx], score) for idx, score in ranked]

class MultiSourceBlockingEngine:
    """Manages separate country-partitioned blocking indices for Source 2 and Source 3."""
    
    def __init__(self, max_cands_per_source: int = 15):
        self.max_cands_per_source = max_cands_per_source
        self.indices_s2: Dict[str, CountryBlockingIndex] = {}
        self.indices_s3: Dict[str, CountryBlockingIndex] = {}
        
    def index_target_records(self, s2_records_by_country: Dict[str, List[RecordProfile]], s3_records_by_country: Dict[str, List[RecordProfile]]) -> None:
        """Build indices for all countries in S2 and S3."""
        for country, recs in s2_records_by_country.items():
            idx = CountryBlockingIndex(country, max_candidates_per_query=self.max_cands_per_source)
            idx.build(recs)
            self.indices_s2[country] = idx
            
        for country, recs in s3_records_by_country.items():
            idx = CountryBlockingIndex(country, max_candidates_per_query=self.max_cands_per_source)
            idx.build(recs)
            self.indices_s3[country] = idx
            
    def retrieve_for_s1(self, s1: RecordProfile) -> List[Tuple[RecordProfile, float]]:
        """Retrieve candidates from both S2 and S3 for a given S1 entity."""
        country = s1.country
        cands_s2 = self.indices_s2[country].retrieve_candidates(s1) if country in self.indices_s2 else []
        cands_s3 = self.indices_s3[country].retrieve_candidates(s1) if country in self.indices_s3 else []
        return cands_s2 + cands_s3
