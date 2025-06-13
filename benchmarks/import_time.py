#!/usr/bin/env python
"""Benchmark import times for datawrangler package."""

import time
import subprocess
import sys
import statistics

def measure_import_time(package, runs=5):
    """Measure the import time of a package."""
    times = []
    
    for _ in range(runs):
        cmd = [sys.executable, "-c", f"import time; start=time.time(); import {package}; print(time.time()-start)"]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            times.append(float(result.stdout.strip()))
        else:
            print(f"Error importing {package}: {result.stderr}")
            return None
    
    return {
        'mean': statistics.mean(times),
        'median': statistics.median(times),
        'min': min(times),
        'max': max(times),
        'stdev': statistics.stdev(times) if len(times) > 1 else 0
    }

def profile_imports():
    """Profile import times for various packages."""
    packages = [
        'numpy',
        'pandas', 
        'sklearn',
        'torch',
        'transformers',
        'sentence_transformers',
        'datawrangler'
    ]
    
    print("Import Time Analysis")
    print("=" * 50)
    
    for package in packages:
        print(f"\nProfiling {package}...")
        stats = measure_import_time(package)
        
        if stats:
            print(f"  Mean:   {stats['mean']:.3f}s")
            print(f"  Median: {stats['median']:.3f}s")
            print(f"  Min:    {stats['min']:.3f}s")
            print(f"  Max:    {stats['max']:.3f}s")
            print(f"  Stdev:  {stats['stdev']:.3f}s")

if __name__ == "__main__":
    profile_imports()