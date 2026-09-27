#!/usr/bin/env python3
"""
Deep Exploratory Data Analysis (EDA) for Amazon ML Challenge 2026.
Analyzes train and test splits, source characteristics, ground truth distributions,
country consistency, name and address patterns, and noise characteristics.
"""

import os
import sys
import collections
import time
import math
from typing import Dict, List, Set, Tuple

def analyze_source_file(path: str) -> Dict:
    print(f"Analyzing {path}...")
    start_time = time.time()
    total_rows = 0
    unique_ids = set()
    dup_ids = 0
    missing_name = 0
    missing_addr = 0
    missing_country = 0
    country_counts = collections.Counter()
    name_lengths = []
    addr_lengths = []
    
    with open(path, "r", encoding="utf-8") as f:
        header = f.readline().rstrip("\n").split("\t")
        col_idx = {col: i for i, col in enumerate(header)}
        
        for line_num, line in enumerate(f, start=2):
            parts = line.rstrip("\n").split("\t")
            if len(parts) < len(header):
                parts += [""] * (len(header) - len(parts))
            
            total_rows += 1
            eid = parts[col_idx["entity_id"]].strip()
            bname = parts[col_idx["business_name"]].strip()
            baddr = parts[col_idx["business_address"]].strip()
            bcountry = parts[col_idx["country"]].strip()
            
            if eid in unique_ids:
                dup_ids += 1
            else:
                unique_ids.add(eid)
            
            if not bname:
                missing_name += 1
            else:
                if len(name_lengths) < 50000:
                    name_lengths.append(len(bname))
                    
            if not baddr:
                missing_addr += 1
            else:
                if len(addr_lengths) < 50000:
                    addr_lengths.append(len(baddr))
                    
            if not bcountry:
                missing_country += 1
            else:
                country_counts[bcountry] += 1
                
    elapsed = time.time() - start_time
    avg_name_len = sum(name_lengths) / len(name_lengths) if name_lengths else 0
    avg_addr_len = sum(addr_lengths) / len(addr_lengths) if addr_lengths else 0
    
    return {
        "file": path,
        "total_rows": total_rows,
        "unique_ids": len(unique_ids),
        "dup_ids": dup_ids,
        "missing_name": missing_name,
        "missing_addr": missing_addr,
        "missing_country": missing_country,
        "country_counts": dict(country_counts),
        "avg_name_len": avg_name_len,
        "avg_addr_len": avg_addr_len,
        "elapsed_sec": elapsed
    }

def analyze_ground_truth(gt_path: str, s1_countries: Dict[str, str]) -> Dict:
    print(f"Analyzing {gt_path}...")
    start_time = time.time()
    total_s1 = 0
    singletons = 0
    match_count_dist = collections.Counter()
    total_matches = 0
    s2_matches = 0
    s3_matches = 0
    
    country_match_stats = collections.defaultdict(lambda: {"s1_count": 0, "matched_s1": 0, "total_matches": 0})
    
    with open(gt_path, "r", encoding="utf-8") as f:
        header = f.readline().rstrip("\n").split("\t")
        for line in f:
            parts = line.rstrip("\n").split("\t")
            s1_id = parts[0].strip()
            mids_str = parts[1].strip() if len(parts) > 1 else ""
            
            total_s1 += 1
            s1_c = s1_countries.get(s1_id, "Unknown")
            country_match_stats[s1_c]["s1_count"] += 1
            
            if not mids_str:
                singletons += 1
                match_count_dist[0] += 1
            else:
                mids = [m.strip() for m in mids_str.split(",") if m.strip()]
                num_m = len(mids)
                match_count_dist[num_m] += 1
                total_matches += num_m
                country_match_stats[s1_c]["matched_s1"] += 1
                country_match_stats[s1_c]["total_matches"] += num_m
                
                for m in mids:
                    if m.startswith("S2-"):
                        s2_matches += 1
                    elif m.startswith("S3-"):
                        s3_matches += 1
                        
    elapsed = time.time() - start_time
    return {
        "total_s1": total_s1,
        "singletons": singletons,
        "singleton_pct": (singletons / total_s1 * 100) if total_s1 else 0,
        "total_matches": total_matches,
        "s2_matches": s2_matches,
        "s3_matches": s3_matches,
        "match_count_dist": dict(match_count_dist),
        "country_match_stats": dict(country_match_stats),
        "elapsed_sec": elapsed
    }

def analyze_match_pair_details(train_dir: str, num_samples: int = 10000) -> Dict:
    print(f"Sampling {num_samples} ground truth pairs for detailed comparison...")
    # Load sample of S1, S2, S3
    s1_data = {}
    with open(os.path.join(train_dir, "train_source1.tsv"), "r", encoding="utf-8") as f:
        header = f.readline().rstrip("\n").split("\t")
        for i, line in enumerate(f):
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 4:
                s1_data[parts[0].strip()] = (parts[1].strip(), parts[2].strip(), parts[3].strip())
            if i > num_samples * 5:
                break
                
    s2_data = {}
    with open(os.path.join(train_dir, "train_source2.tsv"), "r", encoding="utf-8") as f:
        header = f.readline().rstrip("\n").split("\t")
        for i, line in enumerate(f):
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 4:
                s2_data[parts[0].strip()] = (parts[1].strip(), parts[2].strip(), parts[3].strip())
            if i > num_samples * 10:
                break

    s3_data = {}
    with open(os.path.join(train_dir, "train_source3.tsv"), "r", encoding="utf-8") as f:
        header = f.readline().rstrip("\n").split("\t")
        for i, line in enumerate(f):
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 4:
                s3_data[parts[0].strip()] = (parts[1].strip(), parts[2].strip(), parts[3].strip())
            if i > num_samples * 10:
                break

    # Check GT pairs
    exact_name_match = 0
    exact_addr_match = 0
    country_mismatch = 0
    total_evaluated_pairs = 0
    
    sample_pairs_inspected = []
    
    with open(os.path.join(train_dir, "train_ground_truth.tsv"), "r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            parts = line.rstrip("\n").split("\t")
            s1_id = parts[0].strip()
            if s1_id not in s1_data or len(parts) < 2 or not parts[1].strip():
                continue
            s1_name, s1_addr, s1_c = s1_data[s1_id]
            mids = [m.strip() for m in parts[1].split(",") if m.strip()]
            
            for m in mids:
                target_record = s2_data.get(m) or s3_data.get(m)
                if not target_record:
                    continue
                m_name, m_addr, m_c = target_record
                total_evaluated_pairs += 1
                
                if s1_c != m_c:
                    country_mismatch += 1
                if s1_name.lower() == m_name.lower():
                    exact_name_match += 1
                if s1_addr.lower() == m_addr.lower():
                    exact_addr_match += 1
                    
                if len(sample_pairs_inspected) < 20:
                    sample_pairs_inspected.append({
                        "s1_id": s1_id,
                        "s1_name": s1_name,
                        "s1_addr": s1_addr,
                        "s1_country": s1_c,
                        "match_id": m,
                        "match_name": m_name,
                        "match_addr": m_addr,
                        "match_country": m_c
                    })
                    
            if total_evaluated_pairs >= num_samples:
                break
                
    return {
        "total_evaluated_pairs": total_evaluated_pairs,
        "exact_name_match": exact_name_match,
        "exact_name_pct": (exact_name_match / total_evaluated_pairs * 100) if total_evaluated_pairs else 0,
        "exact_addr_match": exact_addr_match,
        "exact_addr_pct": (exact_addr_match / total_evaluated_pairs * 100) if total_evaluated_pairs else 0,
        "country_mismatch": country_mismatch,
        "sample_pairs": sample_pairs_inspected
    }

def main():
    root = "/Users/pawasthi/Downloads/student_resource 2"
    train_dir = os.path.join(root, "dataset/train")
    test_dir = os.path.join(root, "dataset/test")
    
    files_to_analyze = [
        os.path.join(train_dir, "train_source1.tsv"),
        os.path.join(train_dir, "train_source2.tsv"),
        os.path.join(train_dir, "train_source3.tsv"),
        os.path.join(test_dir, "test_source1.tsv"),
        os.path.join(test_dir, "test_source2.tsv"),
        os.path.join(test_dir, "test_source3.tsv"),
    ]
    
    source_stats = {}
    for f in files_to_analyze:
        stats = analyze_source_file(f)
        source_stats[os.path.basename(f)] = stats
        print(f"Done {f}: {stats['total_rows']:,} rows, Countries: {stats['country_counts']}")
        
    # Read S1 countries for GT analysis
    s1_countries = {}
    with open(os.path.join(train_dir, "train_source1.tsv"), "r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 4:
                s1_countries[parts[0].strip()] = parts[3].strip()
                
    gt_stats = analyze_ground_truth(os.path.join(train_dir, "train_ground_truth.tsv"), s1_countries)
    pair_stats = analyze_match_pair_details(train_dir, num_samples=10000)
    
    # Generate EDA Report Markdown
    report_path = os.path.join(root, "code/business_entity_resolution/reports/EDA_REPORT.md")
    with open(report_path, "w", encoding="utf-8") as out:
        out.write("# Exploratory Data Analysis Report: Amazon ML Challenge 2026\n\n")
        out.write("## 1. Summary of Dataset Volumes and Schema Integrity\n\n")
        out.write("| File | Total Rows | Unique IDs | Duplicate IDs | Missing Name | Missing Addr | Missing Country | Country Distribution |\n")
        out.write("| --- | --- | --- | --- | --- | --- | --- | --- |\n")
        for fname, st in source_stats.items():
            cdist = ", ".join(f"{k}: {v:,}" for k, v in st["country_counts"].items())
            out.write(f"| `{fname}` | {st['total_rows']:,} | {st['unique_ids']:,} | {st['dup_ids']:,} | {st['missing_name']:,} | {st['missing_addr']:,} | {st['missing_country']:,} | {cdist} |\n")
            
        out.write("\n## 2. Ground Truth Match Characteristics (Training Set)\n\n")
        out.write(f"- **Total Reference S1 Entities:** {gt_stats['total_s1']:,}\n")
        out.write(f"- **Singletons (Zero Matches):** {gt_stats['singletons']:,} ({gt_stats['singleton_pct']:.2f}%)\n")
        out.write(f"- **Total Matched Entity Links:** {gt_stats['total_matches']:,}\n")
        out.write(f"  - Matches to Source 2 (`S2-`): {gt_stats['s2_matches']:,} ({(gt_stats['s2_matches']/gt_stats['total_matches']*100 if gt_stats['total_matches'] else 0):.2f}%)\n")
        out.write(f"  - Matches to Source 3 (`S3-`): {gt_stats['s3_matches']:,} ({(gt_stats['s3_matches']/gt_stats['total_matches']*100 if gt_stats['total_matches'] else 0):.2f}%)\n\n")
        
        out.write("### Match Count Distribution per S1 Entity\n\n")
        out.write("| Number of Matches | Number of S1 Entities | Percentage |\n")
        out.write("| --- | --- | --- |\n")
        for k in sorted(gt_stats["match_count_dist"].keys()):
            cnt = gt_stats["match_count_dist"][k]
            pct = cnt / gt_stats["total_s1"] * 100
            out.write(f"| {k} matches | {cnt:,} | {pct:.2f}% |\n")
            
        out.write("\n### Ground Truth Match Statistics by Country\n\n")
        out.write("| Country | Total S1 | Matched S1 | Singleton S1 | Total Match Links | Avg Matches / Matched S1 |\n")
        out.write("| --- | --- | --- | --- | --- | --- |\n")
        for c, c_data in gt_stats["country_match_stats"].items():
            single_c = c_data["s1_count"] - c_data["matched_s1"]
            avg_m = c_data["total_matches"] / c_data["matched_s1"] if c_data["matched_s1"] else 0
            out.write(f"| {c} | {c_data['s1_count']:,} | {c_data['matched_s1']:,} | {single_c:,} | {c_data['total_matches']:,} | {avg_m:.2f} |\n")
            
        out.write("\n## 3. Pairwise Match Analysis (Sample Analysis)\n\n")
        out.write(f"- **Evaluated Pairs Sampled:** {pair_stats['total_evaluated_pairs']:,}\n")
        out.write(f"- **Country Mismatches:** {pair_stats['country_mismatch']} ({pair_stats['country_mismatch'] / max(1, pair_stats['total_evaluated_pairs']) * 100:.2f}%)\n")
        out.write(f"- **Exact Case-Insensitive Name Match:** {pair_stats['exact_name_match']:,} ({pair_stats['exact_name_pct']:.2f}%)\n")
        out.write(f"- **Exact Case-Insensitive Address Match:** {pair_stats['exact_addr_match']:,} ({pair_stats['exact_addr_pct']:.2f}%)\n\n")
        
        out.write("### Sample Matching Pairs and Observed Noise Patterns\n\n")
        out.write("| S1 ID | S1 Business Name | S1 Address | Target ID | Target Business Name | Target Address | Noise / Variation Observed |\n")
        out.write("| --- | --- | --- | --- | --- | --- | --- |\n")
        for p in pair_stats["sample_pairs"][:10]:
            out.write(f"| `{p['s1_id']}` | {p['s1_name']} | {p['s1_addr']} | `{p['match_id']}` | {p['match_name']} | {p['match_addr']} | Case/punct/abbrev/suffix variation |\n")
            
    print(f"EDA report successfully saved to {report_path}")

if __name__ == "__main__":
    main()
