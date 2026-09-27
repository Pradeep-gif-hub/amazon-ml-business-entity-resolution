#!/usr/bin/env python3
"""
Phase 3: Systematic Controlled Experiments for Candidate Generation & Decision Optimization.
Evaluates:
- Exp 1: Candidate set size & score pruning (K per source, min blocking score)
- Exp 2: Fine-grained Threshold and Margin sweep
- Exp 3: Singleton protection & Precision veto enhancement
- Exp 4: Combined Best Configuration vs Verified Baseline
"""

import os
import sys
import time
import collections
import numpy as np

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
ber_root = os.path.join(repo_root, "code/business_entity_resolution")
if ber_root not in sys.path:
    sys.path.insert(0, ber_root)

from src.config import default_config
from src.normalization import RecordProfile
from src.data_io import stream_source_tsv, load_ground_truth
from src.candidate_generation import MultiSourceBlockingEngine, CountryBlockingIndex
from src.matching_model import MatchingModel
from src.validation import (
    evaluate_entity_resolution_predictions,
    create_stratified_validation_split,
    compute_f05_score
)

def compute_f1_score(precision: float, recall: float) -> float:
    if precision <= 0.0 or recall <= 0.0:
        return 0.0
    return (2.0 * precision * recall) / (precision + recall)

def main():
    print("=" * 85)
    print("PHASE 3: CONTROLLED VALIDATION EXPERIMENTS (REVERSIBLE & ISOLATED)")
    print("=" * 85)
    
    cfg = default_config
    train_dir = cfg.train_dir
    models_dir = cfg.models_dir
    
    # 1. Load Ground Truth
    print("\n[1/5] Loading ground truth matching labels...")
    gt_map = load_ground_truth(os.path.join(train_dir, "train_ground_truth.tsv"))
    
    # 2. Load Reference S1 Entities & create validation split
    print("\n[2/5] Loading Reference S1 entities (seed=42, 80/20 split)...")
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
    print(f"Validation Queries: {len(val_queries):,}")
    
    # 3. Load S2 and S3 target pools
    print("\n[3/5] Loading and indexing S2 and S3 target pools (2.4M records)...")
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
    print(f"Total True Match Pairs in Pool: {total_true_in_pool:,}")
    
    # 4. Load Trained Matching Model
    print("\n[4/5] Loading trained matching model checkpoint...")
    matcher = MatchingModel(cfg.gbd_params)
    model_path = os.path.join(models_dir, "matching_model.pkl")
    matcher.load(model_path)
    print(f"Loaded GBDT model checkpoint from {model_path} (threshold={matcher.threshold:.3f})")
    
    # Build full blocking index with cap = 20 for flexible offline pruning evaluation
    print("\n[5/5] Building full candidate blocking engine...")
    engine_max20 = MultiSourceBlockingEngine(max_cands_per_source=20)
    engine_max20.index_target_records(s2_by_c, s3_by_c)
    
    # Pre-retrieve max 20 candidates per source for all validation queries
    print("Pre-retrieving candidate pools for validation queries...")
    t0 = time.time()
    val_cands_raw = {}
    for s1 in val_queries:
        c = s1.country
        cands_s2 = engine_max20.indices_s2[c].retrieve_candidates(s1) if c in engine_max20.indices_s2 else []
        cands_s3 = engine_max20.indices_s3[c].retrieve_candidates(s1) if c in engine_max20.indices_s3 else []
        val_cands_raw[s1.entity_id] = (cands_s2, cands_s3)
    print(f"Pre-retrieved in {time.time() - t0:.2f}s")
    
    # Evaluation Helper Function
    def evaluate_configuration(
        k_per_source=15,
        min_score=0.0,
        threshold=0.35,
        margin=0.20,
        enhanced_veto=False
    ):
        cand_map = {}
        total_cands = 0
        true_retrieved = 0
        
        for s1 in val_queries:
            s2_c, s3_c = val_cands_raw[s1.entity_id]
            
            # Filter by min_score and cap K
            s2_f = [c for c in s2_c if c[1] >= min_score][:k_per_source]
            s3_f = [c for c in s3_c if c[1] >= min_score][:k_per_source]
            all_c = s2_f + s3_f
            
            cand_map[s1.entity_id] = all_c
            total_cands += len(all_c)
            
            cand_ids = {c[0].entity_id for c in all_c}
            true_mids = val_gt[s1.entity_id]
            true_retrieved += len(cand_ids & true_mids)
            
        cand_recall = (true_retrieved / total_true_in_pool * 100) if total_true_in_pool else 0.0
        avg_cands = total_cands / len(val_queries)
        
        # Batch score
        batch_items = [(s1, cand_map[s1.entity_id]) for s1 in val_queries]
        scored_batch = matcher.score_batch(batch_items)
        
        # Filter matches with decision rules
        preds = {}
        for s1, scored in scored_batch:
            if not scored:
                preds[s1.entity_id] = set()
                continue
                
            scored.sort(key=lambda x: x[1], reverse=True)
            max_prob = scored[0][1]
            if max_prob < threshold:
                preds[s1.entity_id] = set()
                continue
                
            matched_ids = []
            for cand, prob in scored:
                if prob >= threshold and prob >= (max_prob - margin):
                    name_exact = (s1.clean_name == cand.clean_name) or (s1.core_name == cand.core_name) or (s1.nospace_name == cand.nospace_name)
                    num_conflict = len(s1.addr_numbers) > 0 and len(cand.addr_numbers) > 0 and len(s1.addr_numbers & cand.addr_numbers) == 0
                    
                    if not name_exact and num_conflict and prob < (threshold + 0.15):
                        continue
                        
                    if enhanced_veto:
                        # Extra protection against single-word name false merges when addresses have 0 token overlap
                        s1_words = s1.core_name.split()
                        if len(s1_words) <= 1 and not name_exact and len(s1.addr_tokens & cand.addr_tokens) == 0 and prob < (threshold + 0.20):
                            continue
                            
                    matched_ids.append(cand.entity_id)
            preds[s1.entity_id] = set(matched_ids)
            
        metrics = evaluate_entity_resolution_predictions(val_gt, preds, s1_countries)
        metrics["cand_recall"] = cand_recall
        metrics["avg_cands"] = avg_cands
        metrics["total_cands"] = total_cands
        return metrics

    # =========================================================================
    # EXPERIMENT 1: CANDIDATE SET SIZE & MIN BLOCKING SCORE OPTIMIZATION
    # =========================================================================
    print("\n" + "=" * 95)
    print("EXPERIMENT 1: CANDIDATE SET SIZE & PRUNING SWEEP (Fixed θ=0.35, δ=0.20)")
    print("=" * 95)
    print(f"{'K/Src':<6} | {'MinScore':<9} | {'Avg Cands':<10} | {'Cand Recall':<12} | {'Macro F0.5':<11} | {'Macro Prec':<11} | {'Macro Rec':<10} | {'Singl Acc':<10}")
    print("-" * 95)
    
    exp1_configs = [
        (15, 0.0),  # Baseline
        (12, 0.0),
        (10, 0.0),
        (8, 0.0),
        (6, 0.0),
        (15, 3.5),
        (12, 3.5),
        (10, 3.5),
        (8, 3.5),
        (15, 4.5),
        (10, 4.5),
    ]
    
    exp1_results = []
    for k, ms in exp1_configs:
        m = evaluate_configuration(k_per_source=k, min_score=ms, threshold=0.35, margin=0.20)
        exp1_results.append((k, ms, m))
        is_baseline = " (Baseline)" if (k==15 and ms==0.0) else ""
        print(f"{k:<6} | {ms:<9.1f} | {m['avg_cands']:<10.2f} | {m['cand_recall']:<11.2f}% | {m['macro_f05']:<11.4f} | {m['macro_precision']:<11.4f} | {m['macro_recall']:<10.4f} | {m['singleton_accuracy']:<10.4f}{is_baseline}")
        
    # =========================================================================
    # EXPERIMENT 2: FINE-GRAINED THRESHOLD & MARGIN SWEEP
    # =========================================================================
    print("\n" + "=" * 95)
    print("EXPERIMENT 2: FINE-GRAINED THRESHOLD & MARGIN SWEEP (K=15, MinScore=0.0)")
    print("=" * 95)
    print(f"{'Threshold (θ)':<14} | {'Margin (δ)':<12} | {'Macro F0.5':<12} | {'Macro Prec':<12} | {'Macro Rec':<12} | {'Singl Acc':<12}")
    print("-" * 95)
    
    thresholds = [0.30, 0.32, 0.34, 0.35, 0.36, 0.37, 0.38, 0.40]
    margins = [0.15, 0.18, 0.20, 0.22, 0.25]
    
    best_th_cfg = None
    best_th_f05 = -1.0
    
    for th in thresholds:
        for mg in margins:
            m = evaluate_configuration(k_per_source=15, min_score=0.0, threshold=th, margin=mg)
            if m['macro_f05'] > best_th_f05:
                best_th_f05 = m['macro_f05']
                best_th_cfg = (th, mg, m)
            if mg == 0.20 or th == 0.35:
                is_base = " (Baseline)" if (th==0.35 and mg==0.20) else ""
                print(f"{th:<14.2f} | {mg:<12.2f} | {m['macro_f05']:<12.4f} | {m['macro_precision']:<12.4f} | {m['macro_recall']:<12.4f} | {m['singleton_accuracy']:<12.4f}{is_base}")
                
    print("-" * 95)
    print(f"Optimal Threshold / Margin from Sweep: θ = {best_th_cfg[0]:.2f}, δ = {best_th_cfg[1]:.2f} -> Macro F0.5 = {best_th_cfg[2]['macro_f05']:.4f}")
    
    # =========================================================================
    # EXPERIMENT 3: PRECISION VETO ENHANCEMENT
    # =========================================================================
    print("\n" + "=" * 95)
    print("EXPERIMENT 3: ENHANCED SINGLETON & PRECISION VETO")
    print("=" * 95)
    base_m = evaluate_configuration(k_per_source=15, min_score=0.0, threshold=0.35, margin=0.20, enhanced_veto=False)
    veto_m = evaluate_configuration(k_per_source=15, min_score=0.0, threshold=0.35, margin=0.20, enhanced_veto=True)
    
    print(f"Baseline Standard Veto : Macro F0.5 = {base_m['macro_f05']:.4f} | Prec = {base_m['macro_precision']:.4f} | Rec = {base_m['macro_recall']:.4f} | Singl Acc = {base_m['singleton_accuracy']:.4f}")
    print(f"Enhanced Precision Veto: Macro F0.5 = {veto_m['macro_f05']:.4f} | Prec = {veto_m['macro_precision']:.4f} | Rec = {veto_m['macro_recall']:.4f} | Singl Acc = {veto_m['singleton_accuracy']:.4f}")
    
    # =========================================================================
    # EXPERIMENT 4: COMBINED OPTIMAL CONFIGURATIONS
    # =========================================================================
    print("\n" + "=" * 95)
    print("EXPERIMENT 4: COMBINED CANDIDATE-EFFICIENCY & DECISION CONFIGURATIONS")
    print("=" * 95)
    print(f"{'Configuration':<35} | {'Avg Cands':<10} | {'Cand Rec':<10} | {'Macro F0.5':<11} | {'Macro Prec':<11} | {'Macro Rec':<10} | {'Singl Acc':<10}")
    print("-" * 95)
    
    combos = [
        ("Baseline (K=15, MinScore=0, θ=0.35)", 15, 0.0, 0.35, 0.20, False),
        ("Candidate-Pruned (K=10, MinScore=0, θ=0.35)", 10, 0.0, 0.35, 0.20, False),
        ("Candidate-Pruned (K=8, MinScore=0, θ=0.35)", 8, 0.0, 0.35, 0.20, False),
        ("Score-Filtered (K=15, MinScore=3.5, θ=0.35)", 15, 3.5, 0.35, 0.20, False),
        ("Score-Filtered (K=10, MinScore=3.5, θ=0.35)", 10, 3.5, 0.35, 0.20, False),
        ("Threshold-Tuned (K=15, θ=best, δ=best)", 15, 0.0, best_th_cfg[0], best_th_cfg[1], False),
        ("Combined Lean (K=10, MinScore=3.5, θ=best)", 10, 3.5, best_th_cfg[0], best_th_cfg[1], False),
    ]
    
    for name, k, ms, th, mg, ev in combos:
        m = evaluate_configuration(k_per_source=k, min_score=ms, threshold=th, margin=mg, enhanced_veto=ev)
        print(f"{name:<35} | {m['avg_cands']:<10.2f} | {m['cand_recall']:<9.2f}% | {m['macro_f05']:<11.4f} | {m['macro_precision']:<11.4f} | {m['macro_recall']:<10.4f} | {m['singleton_accuracy']:<10.4f}")
        
    print("=" * 95)

if __name__ == "__main__":
    main()
