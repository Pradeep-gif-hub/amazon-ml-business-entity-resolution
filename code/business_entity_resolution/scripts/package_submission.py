#!/usr/bin/env python3
"""
Submission packager script.
Packages the final submission zip archive in the exact official structure.
"""

import os
import sys
import argparse

pkg_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if pkg_root not in sys.path:
    sys.path.insert(0, pkg_root)

from src.submission import package_final_submission, validate_submission_files

def main():
    parser = argparse.ArgumentParser(description="Package official Amazon ML Challenge 2026 submission zip")
    parser.add_argument("--team-name", default="Relentness", help="Team name prefix for zip archive")
    parser.add_argument("--output-zip", default=None, help="Custom path for output zip archive")
    args = parser.parse_args()
    
    workspace_root = "/Users/pawasthi/Downloads/student_resource 2"
    
    # Check validator first
    print("Validating outputs prior to packaging...")
    is_valid = validate_submission_files(
        matching_path=os.path.join(workspace_root, "output/matching_results.tsv"),
        candidate_path=os.path.join(workspace_root, "output/candidate_pairs.tsv"),
        test_dir=os.path.join(workspace_root, "dataset/test"),
        validator_script=os.path.join(workspace_root, "utils/validate_submission.py")
    )
    
    if not is_valid:
        print("Warning: Validator returned non-zero code. Proceeding with packaging.")
        
    zip_path = package_final_submission(
        team_name=args.team_name,
        workspace_root=workspace_root,
        output_zip_path=args.output_zip
    )
    print(f"\n✅ Final submission package created successfully at:\n  {zip_path}")

if __name__ == "__main__":
    main()
