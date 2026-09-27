"""
High-throughput, memory-conscious I/O utilities for TSV datasets and submission files.
Supports instant country-level fast filtering before record profile allocation.
"""

import os
import csv
from typing import Generator, Dict, List, Set, Tuple, Optional
from .normalization import RecordProfile

DELIM = "\t"

def stream_source_tsv(path: str, filter_country: Optional[str] = None) -> Generator[RecordProfile, None, None]:
    """
    Stream records from a source TSV file as lightweight RecordProfile objects.
    If filter_country is specified, non-matching lines are skipped instantly without profile overhead.
    """
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        header_line = f.readline()
        if not header_line:
            return
        header = [c.strip().lower() for c in header_line.rstrip("\n").split(DELIM)]
        col_idx = {col: i for i, col in enumerate(header)}
        
        eid_idx = col_idx.get("entity_id", 0)
        name_idx = col_idx.get("business_name", 1)
        addr_idx = col_idx.get("business_address", 2)
        country_idx = col_idx.get("country", 3)
        
        for line in f:
            line_str = line.rstrip("\n")
            if not line_str:
                continue
            parts = line_str.split(DELIM)
            while len(parts) < 4:
                parts.append("")
                
            country = parts[country_idx].strip()
            if filter_country is not None and country != filter_country:
                continue
                
            eid = parts[eid_idx].strip()
            name = parts[name_idx].strip()
            addr = parts[addr_idx].strip()
            
            yield RecordProfile(eid, name, addr, country)

def load_ground_truth(path: str) -> Dict[str, Set[str]]:
    """Load ground truth mappings: {s1_id: {matched_s2_or_s3_ids}}."""
    gt_map = {}
    with open(path, "r", encoding="utf-8") as f:
        header = f.readline()
        for line in f:
            line_str = line.rstrip("\n")
            if not line_str:
                continue
            parts = line_str.split(DELIM)
            s1_id = parts[0].strip()
            if len(parts) > 1 and parts[1].strip():
                mids = {m.strip() for m in parts[1].split(",") if m.strip()}
                gt_map[s1_id] = mids
            else:
                gt_map[s1_id] = set()
    return gt_map

class TSVWriter:
    """Streaming TSV writer to write matching and candidate files directly without memory overhead."""
    def __init__(self, path: str, header_col1: str, header_col2: str):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.f = open(path, "w", encoding="utf-8", newline="\n")
        self.f.write(f"{header_col1}{DELIM}{header_col2}\n")
        self.count = 0
        
    def write_row(self, s1_id: str, target_ids: List[str]) -> None:
        seen = set()
        deduped = []
        for tid in target_ids:
            if tid not in seen and tid:
                seen.add(tid)
                deduped.append(tid)
        joined = ",".join(deduped)
        self.f.write(f"{s1_id}{DELIM}{joined}\n")
        self.count += 1
        
    def close(self) -> None:
        self.f.close()
