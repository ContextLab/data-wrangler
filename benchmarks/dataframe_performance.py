#!/usr/bin/env python
"""Benchmark DataFrame performance between pandas and Polars backends."""

import time
import numpy as np
import pandas as pd
import polars as pl
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import datawrangler as dw


def generate_test_data(size='small'):
    """Generate test data of different sizes."""
    if size == 'small':
        n = 1000
    elif size == 'medium':
        n = 100000
    elif size == 'large':
        n = 1000000
    else:
        raise ValueError(f"Unknown size: {size}")
    
    return {
        'array': np.random.randn(n, 10),
        'text': [f"Sample text {i}" for i in range(min(n, 1000))],  # Limit text for performance
        'mixed': [np.random.randn(100), ["text", "data"], None, pd.DataFrame({'a': [1, 2, 3]})]
    }


def time_operation(func, *args, **kwargs):
    """Time a single operation."""
    start = time.time()
    result = func(*args, **kwargs)
    end = time.time()
    return result, end - start


def benchmark_array_wrangling(data, runs=5):
    """Benchmark array wrangling with both backends."""
    results = {'pandas': [], 'polars': []}
    
    for _ in range(runs):
        # Pandas backend
        _, time_pandas = time_operation(dw.wrangle, data['array'], backend='pandas')
        results['pandas'].append(time_pandas)
        
        # Polars backend
        _, time_polars = time_operation(dw.wrangle, data['array'], backend='polars')
        results['polars'].append(time_polars)
    
    return {
        'pandas': {
            'mean': np.mean(results['pandas']),
            'std': np.std(results['pandas']),
            'min': np.min(results['pandas']),
            'max': np.max(results['pandas'])
        },
        'polars': {
            'mean': np.mean(results['polars']),
            'std': np.std(results['polars']),
            'min': np.min(results['polars']),
            'max': np.max(results['polars'])
        }
    }


def benchmark_dataframe_operations(size='medium', runs=5):
    """Benchmark common DataFrame operations."""
    n = 100000 if size == 'medium' else 1000000
    
    # Create test DataFrames
    data = {
        'A': np.random.randn(n),
        'B': np.random.randn(n),
        'C': np.random.choice(['X', 'Y', 'Z'], n),
        'D': np.random.randint(0, 100, n)
    }
    
    df_pandas = pd.DataFrame(data)
    df_polars = pl.DataFrame(data)
    
    operations = {
        'groupby_mean': lambda df: df.groupby('C').mean() if isinstance(df, pd.DataFrame) else df.group_by('C').mean(),
        'filter': lambda df: df[df['A'] > 0] if isinstance(df, pd.DataFrame) else df.filter(pl.col('A') > 0),
        'sort': lambda df: df.sort_values('B') if isinstance(df, pd.DataFrame) else df.sort('B'),
        'join': lambda df: df.merge(df, on='C', suffixes=('_left', '_right')) if isinstance(df, pd.DataFrame) else df.join(df, on='C', suffix='_right')
    }
    
    results = {}
    
    for op_name, op_func in operations.items():
        results[op_name] = {'pandas': [], 'polars': []}
        
        for _ in range(runs):
            # Pandas
            _, time_pandas = time_operation(op_func, df_pandas)
            results[op_name]['pandas'].append(time_pandas)
            
            # Polars
            _, time_polars = time_operation(op_func, df_polars)
            results[op_name]['polars'].append(time_polars)
    
    # Calculate statistics
    for op_name in results:
        for backend in ['pandas', 'polars']:
            times = results[op_name][backend]
            results[op_name][backend] = {
                'mean': np.mean(times),
                'std': np.std(times),
                'speedup': np.mean(results[op_name]['pandas']) / np.mean(times) if backend == 'polars' else 1.0
            }
    
    return results


def format_results(results, title):
    """Format benchmark results for display."""
    print(f"\n{title}")
    print("=" * len(title))
    
    if 'pandas' in results and 'polars' in results:
        # Simple comparison
        print(f"Pandas: {results['pandas']['mean']:.4f}s (±{results['pandas']['std']:.4f}s)")
        print(f"Polars: {results['polars']['mean']:.4f}s (±{results['polars']['std']:.4f}s)")
        speedup = results['pandas']['mean'] / results['polars']['mean']
        print(f"Speedup: {speedup:.2f}x")
    else:
        # Detailed operations
        for op_name, op_results in results.items():
            print(f"\n{op_name}:")
            print(f"  Pandas: {op_results['pandas']['mean']:.4f}s (±{op_results['pandas']['std']:.4f}s)")
            print(f"  Polars: {op_results['polars']['mean']:.4f}s (±{op_results['polars']['std']:.4f}s)")
            print(f"  Speedup: {op_results['polars']['speedup']:.2f}x")


def main():
    """Run all benchmarks."""
    print("Data Wrangler DataFrame Performance Benchmarks")
    print("=" * 50)
    
    # Test data sizes
    sizes = ['small', 'medium']
    
    for size in sizes:
        print(f"\n\nTesting with {size} data...")
        data = generate_test_data(size)
        
        # Array wrangling benchmark
        array_results = benchmark_array_wrangling(data, runs=5)
        format_results(array_results, f"Array Wrangling ({size})")
        
        # DataFrame operations benchmark
        if size in ['medium']:  # Only run intensive operations on medium data
            df_results = benchmark_dataframe_operations(size, runs=3)
            format_results(df_results, f"DataFrame Operations ({size})")
    
    # Memory usage comparison
    print("\n\nMemory Usage Comparison")
    print("=" * 30)
    
    # Create large array
    large_array = np.random.randn(1000000, 10)
    
    # Pandas
    df_pandas = dw.wrangle(large_array, backend='pandas')
    pandas_memory = df_pandas.memory_usage(deep=True).sum() / 1024 / 1024  # MB
    
    # Polars
    df_polars = dw.wrangle(large_array, backend='polars')
    polars_memory = df_polars.estimated_size() / 1024 / 1024  # MB
    
    print(f"Pandas: {pandas_memory:.2f} MB")
    print(f"Polars: {polars_memory:.2f} MB")
    print(f"Memory saved: {(1 - polars_memory/pandas_memory) * 100:.1f}%")


if __name__ == "__main__":
    main()