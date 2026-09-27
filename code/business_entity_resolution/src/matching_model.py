"""
Machine learning matching models and macro F0.5-optimized decision engine.
Uses scikit-learn HistGradientBoostingClassifier with batched vectorized scoring,
feature extraction, and precision-weighted threshold tuning with address veto.
"""

import os
import pickle
import numpy as np
from typing import List, Dict, Set, Tuple, Optional
from sklearn.ensemble import HistGradientBoostingClassifier

from .normalization import RecordProfile
from .features import compute_pair_features, FEATURE_NAMES
from .validation import evaluate_entity_resolution_predictions

class MatchingModel:
    """Gradient boosted decision tree model for scoring candidate entity pairs."""
    
    def __init__(self, gbd_params: Optional[Dict] = None):
        self.params = gbd_params or {
            "learning_rate": 0.08,
            "max_iter": 200,
            "max_leaf_nodes": 31,
            "max_depth": 6,
            "min_samples_leaf": 20,
            "l2_regularization": 1.0,
            "random_state": 42
        }
        self.model = None
        self.threshold: float = 0.52
        self.margin: float = 0.20
        self.feature_names = FEATURE_NAMES
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> "MatchingModel":
        """Train HistGradientBoostingClassifier on feature matrix X and binary labels y."""
        self.model = HistGradientBoostingClassifier(**self.params)
        self.model.fit(X, y)
        return self
        
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict match probability for feature matrix X."""
        if self.model is None or len(X) == 0:
            return np.array([f[-1] for f in X], dtype=np.float32) if len(X) > 0 else np.array([], dtype=np.float32)
        probs = self.model.predict_proba(X)
        return probs[:, 1]

    def score_batch(
        self,
        batch_queries: List[Tuple[RecordProfile, List[Tuple[RecordProfile, float]]]]
    ) -> List[Tuple[RecordProfile, List[Tuple[RecordProfile, float]]]]:
        """
        Score a batch of queries with vectorized feature extraction and inference.
        batch_queries: [(s1, [(cand, blk_score), ...]), ...]
        Returns: [(s1, [(cand, pred_prob), ...]), ...]
        """
        all_features = []
        slice_indices = []
        
        curr_idx = 0
        for s1, cands in batch_queries:
            n_c = len(cands)
            if n_c == 0:
                slice_indices.append((0, 0))
                continue
            for rank, (cand, blk_score) in enumerate(cands):
                feats = compute_pair_features(s1, cand, blk_score, rank)
                all_features.append(feats)
            slice_indices.append((curr_idx, curr_idx + n_c))
            curr_idx += n_c
            
        if not all_features:
            return [(s1, []) for s1, _ in batch_queries]
            
        X = np.array(all_features, dtype=np.float32)
        all_probs = self.predict_proba(X)
        
        results = []
        for i, (s1, cands) in enumerate(batch_queries):
            start_i, end_i = slice_indices[i]
            if start_i == end_i:
                results.append((s1, []))
            else:
                scored = [(cands[j][0], float(all_probs[start_i + j])) for j in range(end_i - start_i)]
                results.append((s1, scored))
                
        return results
        
    def score_candidates(
        self,
        s1: RecordProfile,
        candidates: List[Tuple[RecordProfile, float]]
    ) -> List[Tuple[RecordProfile, float]]:
        """Score candidate records for a single query S1 entity."""
        if not candidates:
            return []
        res = self.score_batch([(s1, candidates)])
        return res[0][1]
        
    def filter_matches(
        self,
        s1: RecordProfile,
        scored_candidates: List[Tuple[RecordProfile, float]],
        threshold: Optional[float] = None,
        margin: Optional[float] = None
    ) -> List[str]:
        """Apply macro F0.5-optimized decision rules to emit final matched IDs."""
        if not scored_candidates:
            return []
            
        th = threshold if threshold is not None else self.threshold
        mg = margin if margin is not None else self.margin
        
        # Sort by predicted probability descending
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        max_prob = scored_candidates[0][1]
        
        # If best candidate is below threshold -> singleton (empty match)
        if max_prob < th:
            return []
            
        matched_ids = []
        for cand, prob in scored_candidates:
            if prob >= th and prob >= (max_prob - mg):
                # Address veto check:
                # If name is not exact and address numbers directly conflict, veto
                name_exact = (s1.clean_name == cand.clean_name) or (s1.core_name == cand.core_name) or (s1.nospace_name == cand.nospace_name)
                num_conflict = len(s1.addr_numbers) > 0 and len(cand.addr_numbers) > 0 and len(s1.addr_numbers & cand.addr_numbers) == 0
                
                if not name_exact and num_conflict and prob < (th + 0.15):
                    continue  # Veto false merge
                    
                matched_ids.append(cand.entity_id)
                
        return matched_ids

    def tune_thresholds(
        self,
        val_s1_records: List[RecordProfile],
        val_candidate_map: Dict[str, List[Tuple[RecordProfile, float]]],
        val_gt_map: Dict[str, Set[str]],
        threshold_range: Tuple[float, float, int] = (0.35, 0.75, 17)
    ) -> float:
        """Grid search optimal decision threshold maximizing macro F0.5 on validation data."""
        best_th = self.threshold
        best_score = -1.0
        
        thresholds = np.linspace(threshold_range[0], threshold_range[1], threshold_range[2])
        
        # Batched scoring for validation records
        batch_items = [(s1, val_candidate_map.get(s1.entity_id, [])) for s1 in val_s1_records]
        scored_batch = self.score_batch(batch_items)
        scored_map = {s1.entity_id: scored for s1, scored in scored_batch}
            
        for th in thresholds:
            preds = {}
            for s1 in val_s1_records:
                mids = self.filter_matches(s1, scored_map[s1.entity_id], threshold=th, margin=self.margin)
                preds[s1.entity_id] = set(mids)
                
            res = evaluate_entity_resolution_predictions(val_gt_map, preds)
            f05 = res["macro_f05"]
            if f05 > best_score:
                best_score = f05
                best_th = float(th)
                
        self.threshold = best_th
        return best_th

    def save(self, path: str) -> None:
        """Save model and metadata to disk."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "model": self.model,
                "threshold": self.threshold,
                "margin": self.margin,
                "params": self.params,
                "feature_names": self.feature_names
            }, f)

    def load(self, path: str) -> "MatchingModel":
        """Load trained model and metadata from disk."""
        with open(path, "rb") as f:
            data = pickle.load(f)
            self.model = data["model"]
            self.threshold = data["threshold"]
            self.margin = data["margin"]
            self.params = data["params"]
            self.feature_names = data["feature_names"]
        return self
