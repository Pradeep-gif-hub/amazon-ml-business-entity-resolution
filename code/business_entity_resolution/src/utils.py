"""
Logging, profiling, and helper utilities.
"""

import time
import sys
from contextlib import contextmanager

@contextmanager
def timer(name: str):
    """Context manager for timing code execution blocks."""
    t0 = time.time()
    print(f"[{name}] Starting...")
    try:
        yield
    finally:
        elapsed = time.time() - t0
        print(f"[{name}] Finished in {elapsed:.2f} seconds ({elapsed/60:.2f} mins).")

def format_number(val: int) -> str:
    """Format integer with commas."""
    return f"{val:,}"
