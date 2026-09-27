"""
Configuration and hyperparameters for Business Entity Resolution.
"""

import os
from dataclasses import dataclass, field
from typing import List, Dict

@dataclass
class Config:
    # Directories
    workspace_root: str = "/Users/pawasthi/Downloads/student_resource 2"
    train_dir: str = os.path.join(workspace_root, "dataset/train")
    test_dir: str = os.path.join(workspace_root, "dataset/test")
    output_dir: str = os.path.join(workspace_root, "output")
    reports_dir: str = os.path.join(workspace_root, "code/business_entity_resolution/reports")
    models_dir: str = os.path.join(workspace_root, "code/business_entity_resolution/models")
    
    # Random seed
    random_seed: int = 42
    
    # Candidate Generation / Blocking hyperparameters
    max_candidates_per_source: int = 15  # Max candidates from S2 and S3 each
    max_total_candidates: int = 30       # Max total candidates per S1
    min_token_len: int = 2               # Min token length for inverted index
    max_token_doc_freq_ratio: float = 0.006 # Stop words cutoff ratio
    
    # GBDT Classifier hyperparameters
    gbd_params: Dict = field(default_factory=lambda: {
        "learning_rate": 0.08,
        "max_iter": 200,
        "max_leaf_nodes": 31,
        "max_depth": 6,
        "min_samples_leaf": 20,
        "l2_regularization": 1.0,
        "random_state": 42
    })
    
    # Decision thresholds
    default_threshold: float = 0.52
    top_score_margin: float = 0.20
    
    # Multiprocessing
    num_workers: int = 8
    chunk_size: int = 100000

default_config = Config()
