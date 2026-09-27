"""
Validation strategy and official macro F0.5 metric evaluation.
Implements non-leaking train/val splits and detailed performance diagnostics.
"""

import random
from typing import Dict, List, Set, Tuple, Optional
import collections

def compute_f05_score(precision: float, recall: float) -> float:
    """Compute F_0.5 score from precision and recall: (1.25 * P * R) / (0.25 * P + R)."""
    if precision <= 0.0 or recall <= 0.0:
        return 0.0
    denom = 0.25 * precision + recall
    if denom <= 0.0:
        return 0.0
    return (1.25 * precision * recall) / denom

def evaluate_entity_resolution_predictions(
    ground_truth: Dict[str, Set[str]],
    predictions: Dict[str, Set[str]],
    s1_countries: Optional[Dict[str, str]] = None
) -> Dict[str, float]:
    """
    Compute official macro-averaged F0.5 score and diagnostic sub-metrics.
    Handles singletons (0 matches) with exact official competition logic:
    - True singleton + predicted empty = 1.0
    - True singleton + predicted non-empty = 0.0
    - True matched + predicted empty = 0.0
    - True matched + predicted non-empty = (1.25 * P * R) / (0.25 * P + R)
    """
    all_s1_ids = list(ground_truth.keys())
    total_entities = len(all_s1_ids)
    
    if total_entities == 0:
        return {
            "macro_f05": 0.0,
            "macro_precision": 0.0,
            "macro_recall": 0.0,
            "singleton_accuracy": 0.0,
            "matched_f05": 0.0,
            "total_entities": 0
        }
        
    f05_scores = []
    precisions = []
    recalls = []
    
    singleton_total = 0
    singleton_correct = 0
    
    matched_f05_scores = []
    
    # Per country tracking
    country_f05 = collections.defaultdict(list)
    
    # Global TP, FP, FN for micro metrics
    global_tp = 0
    global_fp = 0
    global_fn = 0
    
    for s1_id in all_s1_ids:
        y_true = ground_truth[s1_id]
        y_pred = predictions.get(s1_id, set())
        country = s1_countries.get(s1_id, "Unknown") if s1_countries else "Unknown"
        
        # Singleton case
        if len(y_true) == 0:
            singleton_total += 1
            if len(y_pred) == 0:
                s_f05 = 1.0
                s_prec = 1.0
                s_rec = 1.0
                singleton_correct += 1
            else:
                s_f05 = 0.0
                s_prec = 0.0
                s_rec = 1.0
                global_fp += len(y_pred)
        else:
            # Non-singleton case
            if len(y_pred) == 0:
                s_f05 = 0.0
                s_prec = 0.0
                s_rec = 0.0
                global_fn += len(y_true)
            else:
                tp = len(y_true & y_pred)
                fp = len(y_pred - y_true)
                fn = len(y_true - y_pred)
                
                global_tp += tp
                global_fp += fp
                global_fn += fn
                
                if tp == 0:
                    s_f05 = 0.0
                    s_prec = 0.0
                    s_rec = 0.0
                else:
                    s_prec = tp / len(y_pred)
                    s_rec = tp / len(y_true)
                    s_f05 = compute_f05_score(s_prec, s_rec)
                    
            matched_f05_scores.append(s_f05)
            
        f05_scores.append(s_f05)
        precisions.append(s_prec)
        recalls.append(s_rec)
        country_f05[country].append(s_f05)
        
    macro_f05 = sum(f05_scores) / total_entities
    macro_prec = sum(precisions) / total_entities
    macro_rec = sum(recalls) / total_entities
    
    micro_prec = global_tp / (global_tp + global_fp) if (global_tp + global_fp) > 0 else 0.0
    micro_rec = global_tp / (global_tp + global_fn) if (global_tp + global_fn) > 0 else 0.0
    micro_f05 = compute_f05_score(micro_prec, micro_rec)
    
    singleton_acc = (singleton_correct / singleton_total) if singleton_total > 0 else 1.0
    matched_f05 = (sum(matched_f05_scores) / len(matched_f05_scores)) if matched_f05_scores else 0.0
    
    results = {
        "macro_f05": macro_f05,
        "macro_precision": macro_prec,
        "macro_recall": macro_rec,
        "micro_f05": micro_f05,
        "micro_precision": micro_prec,
        "micro_recall": micro_rec,
        "singleton_accuracy": singleton_acc,
        "singleton_count": singleton_total,
        "matched_f05": matched_f05,
        "matched_count": len(matched_f05_scores),
        "total_entities": total_entities
    }
    
    for c, scores in country_f05.items():
        results[f"macro_f05_{c}"] = sum(scores) / len(scores)
        
    return results

def create_stratified_validation_split(
    s1_ids: List[str],
    s1_countries: Dict[str, str],
    gt_map: Dict[str, Set[str]],
    val_ratio: float = 0.1,
    random_seed: int = 42
) -> Tuple[List[str], List[str]]:
    """Create a stratified train / validation split of S1 IDs without leakage."""
    rng = random.Random(random_seed)
    
    # Group by (country, is_singleton)
    strata = collections.defaultdict(list)
    for s1_id in s1_ids:
        country = s1_countries.get(s1_id, "Unknown")
        is_singleton = len(gt_map.get(s1_id, set())) == 0
        strata[(country, is_singleton)].append(s1_id)
        
    train_ids = []
    val_ids = []
    
    for key, ids in strata.items():
        shuffled = list(ids)
        rng.shuffle(shuffled)
        n_val = max(1, int(len(shuffled) * val_ratio))
        val_ids.extend(shuffled[:n_val])
        train_ids.extend(shuffled[n_val:])
        
    rng.shuffle(train_ids)
    rng.shuffle(val_ids)
    return train_ids, val_ids
