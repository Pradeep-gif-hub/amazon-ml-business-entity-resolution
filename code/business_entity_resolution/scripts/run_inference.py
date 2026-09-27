#!/usr/bin/env python3
"""
Official test inference entrypoint script.
Generates output/matching_results.tsv and output/candidate_pairs.tsv,
then verifies them against the official challenge validator.
"""

import os
import sys
import argparse
import time

pkg_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if pkg_root not in sys.path:
    sys.path.insert(0, pkg_root)

from src.config import Config, default_config
from src.inference import run_test_inference
from src.submission import validate_submission_files

def main():
    parser = argparse.ArgumentParser(description="Run test inference for Amazon ML Challenge 2026 Entity Resolution")
    parser.add_argument("--test-dir", default="dataset/test", help="Path to directory containing test TSV files")
    parser.add_argument("--output-dir", default="output", help="Directory where matching_results.tsv and candidate_pairs.tsv are saved")
    parser.add_argument("--model-path", default="code/business_entity_resolution/models/matching_model.pkl", help="Path to trained model checkpoint")
    parser.add_argument("--check-ids", action="store_true", help="Enable deep ID-existence validation against test source2/3 files")
    args = parser.parse_args()
    
    cfg = default_config
    cfg.test_dir = args.test_dir
    cfg.output_dir = args.output_dir
    
    # 1. Run inference
    matching_path, candidate_path = run_test_inference(
        test_dir=args.test_dir,
        output_dir=args.output_dir,
        model_path=args.model_path,
        config=cfg
    )
    
    # 2. Run official validator
    print("\n" + "=" * 70)
    print("RUNNING OFFICIAL VALIDATOR ON GENERATED SUBMISSION FILES")
    print("=" * 70)
    is_valid = validate_submission_files(
        matching_path=matching_path,
        candidate_path=candidate_path,
        test_dir=args.test_dir,
        validator_script="utils/validate_submission.py",
        check_ids=args.check_ids
    )
    
    if is_valid:
        print("\n🎉 ALL CHECKS PASSED: Submission outputs are 100% valid and ready for submission!")
        sys.exit(0)
    else:
        print("\n❌ VALIDATION ERRORS DETECTED: Please review issues above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
