"""
High-throughput, memory-bounded, country-partitioned test inference pipeline.
Processes France, US, and India sequentially with isolated memory buffers and fast country filtering,
generating output/matching_results.tsv and output/candidate_pairs.tsv.
"""

import os
import sys
import gc
import time
import csv
import collections
from typing import Dict, List, Set, Tuple, Optional

from .normalization import RecordProfile
from .candidate_generation import CountryBlockingIndex
from .matching_model import MatchingModel
from .data_io import stream_source_tsv, TSVWriter
from .config import Config, default_config

DELIM = "\t"

COUNTRY_EXPECTED_S1 = {
    "France": 259452,
    "US": 663106,
    "India": 809986
}

def count_file_lines(path: str) -> int:
    """Fast line count check for partition files."""
    if not os.path.exists(path):
        return 0
    cnt = 0
    with open(path, "rb") as f:
        for _ in f:
            cnt += 1
    return cnt

def run_country_partition_inference(
    country: str,
    test_dir: str,
    output_dir: str,
    matcher: MatchingModel,
    max_cands_per_source: int = 15,
    batch_size: int = 5000,
    force_recompute: bool = False
) -> Tuple[str, str, int, int, int, int]:
    """Process a single country partition (France, US, or India) with isolated memory."""
    print(f"\n{'='*30} PROCESSING COUNTRY: {country} {'='*30}", flush=True)
    t_start = time.time()
    
    part_matching_path = os.path.join(output_dir, f"matching_part_{country}.tsv")
    part_candidate_path = os.path.join(output_dir, f"candidate_part_{country}.tsv")
    expected_rows = COUNTRY_EXPECTED_S1.get(country, None)
    
    # Check if complete partition files already exist
    if not force_recompute and expected_rows is not None:
        m_lines = count_file_lines(part_matching_path)
        c_lines = count_file_lines(part_candidate_path)
        if m_lines == expected_rows + 1 and c_lines == expected_rows + 1:
            print(f"[{country}] ✓ Existing complete partition found ({expected_rows:,} rows). Skipping recomputation.", flush=True)
            return part_matching_path, part_candidate_path, expected_rows, 0, 0, 0
    
    # 1. Load target records for this country only with fast country filter
    print(f"[{country}] Loading Source 2 records for {country}...", flush=True)
    t0 = time.time()
    s2_records = list(stream_source_tsv(os.path.join(test_dir, "test_source2.tsv"), filter_country=country))
    print(f"[{country}] Loaded {len(s2_records):,} S2 records in {time.time() - t0:.2f}s", flush=True)
    
    print(f"[{country}] Loading Source 3 records for {country}...", flush=True)
    t0 = time.time()
    s3_records = list(stream_source_tsv(os.path.join(test_dir, "test_source3.tsv"), filter_country=country))
    print(f"[{country}] Loaded {len(s3_records):,} S3 records in {time.time() - t0:.2f}s", flush=True)
    
    # 2. Build blocking indices
    print(f"[{country}] Building blocking indices for S2 and S3...", flush=True)
    t0 = time.time()
    idx_s2 = CountryBlockingIndex(country, max_candidates_per_query=max_cands_per_source)
    idx_s2.build(s2_records)
    
    idx_s3 = CountryBlockingIndex(country, max_candidates_per_query=max_cands_per_source)
    idx_s3.build(s3_records)
    print(f"[{country}] Built blocking indices on {len(s2_records) + len(s3_records):,} target records in {time.time() - t0:.2f}s", flush=True)
    
    # 3. Stream S1 queries for this country and perform batched scoring
    match_writer = TSVWriter(part_matching_path, "source1_entity_id", "matched_entity_ids")
    cand_writer = TSVWriter(part_candidate_path, "source1_entity_id", "candidate_entity_ids")
    
    print(f"[{country}] Streaming S1 queries and performing batched matching inference...", flush=True)
    t0 = time.time()
    query_count = 0
    total_cands = 0
    total_matches = 0
    singletons = 0
    
    batch = []
    
    def process_query_batch(q_batch):
        nonlocal total_cands, total_matches, singletons
        if not q_batch:
            return
        batch_with_cands = []
        for s1 in q_batch:
            cands_s2 = idx_s2.retrieve_candidates(s1)
            cands_s3 = idx_s3.retrieve_candidates(s1)
            all_cands = cands_s2 + cands_s3
            batch_with_cands.append((s1, all_cands))
            cand_ids = [c[0].entity_id for c in all_cands]
            cand_writer.write_row(s1.entity_id, cand_ids)
            total_cands += len(cand_ids)
            
        scored_batch = matcher.score_batch(batch_with_cands)
        for s1, scored_cands in scored_batch:
            matched_ids = matcher.filter_matches(s1, scored_cands) if scored_cands else []
            match_writer.write_row(s1.entity_id, matched_ids)
            if matched_ids:
                total_matches += len(matched_ids)
            else:
                singletons += 1
                
    for s1 in stream_source_tsv(os.path.join(test_dir, "test_source1.tsv"), filter_country=country):
        query_count += 1
        batch.append(s1)
        if len(batch) >= batch_size:
            process_query_batch(batch)
            batch = []
            if query_count % 100000 == 0:
                elapsed = time.time() - t0
                print(f"  [{country}] Processed {query_count:,} queries ({query_count/elapsed:,.1f} qps) | Matches: {total_matches:,} | Singletons: {singletons:,}", flush=True)
                
    if batch:
        process_query_batch(batch)
        
    match_writer.close()
    cand_writer.close()
    
    # 4. Clean up memory for next country
    del s2_records, s3_records, idx_s2, idx_s3
    gc.collect()
    
    elapsed_total = time.time() - t_start
    print(f"[{country}] Done: {query_count:,} queries processed in {elapsed_total:.2f}s ({query_count/elapsed_total:,.1f} qps)", flush=True)
    return part_matching_path, part_candidate_path, query_count, total_cands, total_matches, singletons

def merge_country_partitions(
    partition_files: List[Tuple[str, str]],
    test_source1_path: str,
    final_matching_path: str,
    final_candidate_path: str
) -> None:
    """Merge per-country partition files into the final ordered submission TSVs."""
    print("\n[Merging] Combining country partition outputs into final submission files...")
    t0 = time.time()
    
    match_map = {}
    cand_map = {}
    
    for m_part, c_part in partition_files:
        with open(m_part, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter=DELIM)
            next(reader, None)  # header
            for row in reader:
                if row:
                    match_map[row[0].strip()] = row[1].strip() if len(row) > 1 else ""
                    
        with open(c_part, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter=DELIM)
            next(reader, None)  # header
            for row in reader:
                if row:
                    cand_map[row[0].strip()] = row[1].strip() if len(row) > 1 else ""
                    
        try:
            os.remove(m_part)
            os.remove(c_part)
        except OSError:
            pass
            
    # Write final files following exact test_source1.tsv entity order
    os.makedirs(os.path.dirname(final_matching_path), exist_ok=True)
    os.makedirs(os.path.dirname(final_candidate_path), exist_ok=True)
    
    total_written = 0
    with open(final_matching_path, "w", encoding="utf-8", newline="\n") as f_m, \
         open(final_candidate_path, "w", encoding="utf-8", newline="\n") as f_c:
        f_m.write(f"source1_entity_id\tmatched_entity_ids\n")
        f_c.write(f"source1_entity_id\tcandidate_entity_ids\n")
        
        with open(test_source1_path, "r", encoding="utf-8") as f_s1:
            reader = csv.reader(f_s1, delimiter=DELIM)
            next(reader, None)
            for row in reader:
                if not row:
                    continue
                s1_id = row[0].strip()
                m_str = match_map.get(s1_id, "")
                c_str = cand_map.get(s1_id, "")
                f_m.write(f"{s1_id}\t{m_str}\n")
                f_c.write(f"{s1_id}\t{c_str}\n")
                total_written += 1
                
    print(f"[Merging] Successfully wrote {total_written:,} ordered rows to final deliverables in {time.time() - t0:.2f}s")

def run_test_inference(
    test_dir: str = "dataset/test",
    output_dir: str = "output",
    model_path: Optional[str] = None,
    config: Optional[Config] = None,
    batch_size: int = 5000
) -> Tuple[str, str]:
    """Execute complete, memory-bounded, country-partitioned test inference pipeline."""
    cfg = config or default_config
    start_total_time = time.time()
    
    print("=" * 75)
    print("STARTING HIGH-THROUGHPUT COUNTRY-PARTITIONED TEST INFERENCE PIPELINE")
    print(f"Test Directory: {test_dir}")
    print(f"Output Directory: {output_dir}")
    print("=" * 75)
    
    # 1. Load trained matching model
    print("\n[Setup] Loading matching model checkpoint...")
    matcher = MatchingModel(cfg.gbd_params)
    m_path = model_path or os.path.join(cfg.models_dir, "matching_model.pkl")
    if os.path.exists(m_path):
        matcher.load(m_path)
        print(f"  ✓ Loaded trained GBDT model from {m_path} (Decision Threshold: {matcher.threshold:.3f})")
    else:
        print("  ✓ Using baseline feature matcher")
        
    # 2. Process countries sequentially in isolated memory passes
    countries = ["France", "US", "India"]
    partition_files = []
    
    total_all_queries = 0
    total_all_cands = 0
    total_all_matches = 0
    total_all_singletons = 0
    
    for country in countries:
        m_part, c_part, q_cnt, c_cnt, m_cnt, s_cnt = run_country_partition_inference(
            country=country,
            test_dir=test_dir,
            output_dir=output_dir,
            matcher=matcher,
            max_cands_per_source=cfg.max_candidates_per_source,
            batch_size=batch_size
        )
        partition_files.append((m_part, c_part))
        total_all_queries += q_cnt
        total_all_cands += c_cnt
        total_all_matches += m_cnt
        total_all_singletons += s_cnt
        
    # 3. Merge partition outputs into final files
    final_matching_path = os.path.join(output_dir, "matching_results.tsv")
    final_candidate_path = os.path.join(output_dir, "candidate_pairs.tsv")
    test_s1_path = os.path.join(test_dir, "test_source1.tsv")
    
    merge_country_partitions(
        partition_files=partition_files,
        test_source1_path=test_s1_path,
        final_matching_path=final_matching_path,
        final_candidate_path=final_candidate_path
    )
    
    total_elapsed = time.time() - start_total_time
    print("\n" + "=" * 75)
    print("🎉 INFERENCE COMPLETED SUCCESSFULLY ACROSS ALL TEST RECORDS")
    print(f"Total S1 Queries Processed: {total_all_queries:,}")
    print(f"Total Candidates Generated: {total_all_cands:,} (avg {total_all_cands/total_all_queries:.2f}/query)")
    print(f"Total Matches Predicted: {total_all_matches:,} (avg {total_all_matches/total_all_queries:.2f}/query)")
    print(f"Singletons (0 Matches): {total_all_singletons:,} ({total_all_singletons/total_all_queries*100:.2f}%)")
    print(f"Total Pipeline Runtime: {total_elapsed:.2f} seconds ({total_elapsed/60:.2f} minutes)")
    print(f"Overall Inference Throughput: {total_all_queries/total_elapsed:,.1f} queries/second")
    print(f"Deliverables Saved to:\n  1. {final_matching_path}\n  2. {final_candidate_path}")
    print("=" * 75)
    
    return final_matching_path, final_candidate_path
