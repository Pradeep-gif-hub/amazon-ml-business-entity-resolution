"""
Submission validation and packaging utilities.
Enforces official competition packaging format and runs validate_submission.py.
"""

import os
import sys
import subprocess
import zipfile
from typing import List, Optional

def validate_submission_files(
    matching_path: str = "output/matching_results.tsv",
    candidate_path: str = "output/candidate_pairs.tsv",
    test_dir: str = "dataset/test",
    validator_script: str = "utils/validate_submission.py",
    check_ids: bool = False
) -> bool:
    """Run the official validator script on generated submission files."""
    if not os.path.exists(validator_script):
        print(f"Error: Validator script not found at {validator_script}")
        return False
        
    cmd = [
        sys.executable,
        validator_script,
        "--matching", matching_path,
        "--candidate", candidate_path,
        "--test-dir", test_dir
    ]
    if check_ids:
        cmd.append("--check-ids")
        
    print(f"Running official validator command: {' '.join(cmd)}")
    res = subprocess.run(cmd, capture_output=True, text=True)
    print("--- Official Validator Output ---")
    print(res.stdout)
    if res.stderr:
        print("--- Validator Errors / Warnings ---")
        print(res.stderr)
        
    if res.returncode == 0:
        print("\n✅ VALIDATION PASSED: Files are 100% compliant with competition specifications.")
        return True
    else:
        print(f"\n❌ VALIDATION FAILED with exit code {res.returncode}.")
        return False

def package_final_submission(
    team_name: str = "Relentness",
    workspace_root: str = "/Users/pawasthi/Downloads/student_resource 2",
    output_zip_path: Optional[str] = None
) -> str:
    """
    Package the final submission zip with exact required structure:
    <team_name>_submission.zip
    ├── output/
    │   ├── matching_results.tsv
    │   └── candidate_pairs.tsv
    ├── code/
    │   └── business_entity_resolution/
    │       ├── src/
    │       ├── README.md
    │       └── requirements.txt
    └── Documentation_template.md
    """
    zip_path = output_zip_path or os.path.join(workspace_root, f"{team_name}_submission.zip")
    
    # Required files to include
    files_to_pack = [
        ("output/matching_results.tsv", "output/matching_results.tsv"),
        ("output/candidate_pairs.tsv", "output/candidate_pairs.tsv"),
        ("Documentation_template.md", "Documentation_template.md"),
        ("code/business_entity_resolution/README.md", "code/business_entity_resolution/README.md"),
        ("code/business_entity_resolution/requirements.txt", "code/business_entity_resolution/requirements.txt"),
    ]
    
    # Collect all src/ files
    src_dir = os.path.join(workspace_root, "code/business_entity_resolution/src")
    for fname in sorted(os.listdir(src_dir)):
        if fname.endswith(".py"):
            rel_src = os.path.join("code/business_entity_resolution/src", fname)
            files_to_pack.append((rel_src, rel_src))
            
    print(f"Creating submission package at: {zip_path}")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for local_rel, arcname in files_to_pack:
            abs_local = os.path.join(workspace_root, local_rel)
            if os.path.exists(abs_local):
                zf.write(abs_local, arcname=arcname)
                print(f"  Added: {arcname} ({os.path.getsize(abs_local) / 1024:.1f} KB)")
            else:
                print(f"  Warning: File missing, cannot pack: {abs_local}")
                
    # Verify archive
    print("\nVerifying submission zip archive contents:")
    with zipfile.ZipFile(zip_path, "r") as zf:
        namelist = zf.namelist()
        for name in sorted(namelist):
            info = zf.getinfo(name)
            print(f"  [OK] {name} (uncompressed: {info.file_size / 1024:.1f} KB, compressed: {info.compress_size / 1024:.1f} KB)")
            
    return zip_path
