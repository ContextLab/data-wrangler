# Data Wrangler Speed & Efficiency Enhancement Plan

**Target Phase**: Performance Optimization and Modern DataFrame Support  
**Focus Areas**: Lazy imports, performance optimization, and Polars integration

## 🎯 **Phase Objectives**

Transform data-wrangler from a functional but potentially slow library into a high-performance, efficient data processing toolkit that:

1. **Lazy Import Architecture**: Only load heavy dependencies when actually needed
2. **Polars Integration**: Support for the fastest DataFrame library available
3. **Performance Optimization**: Minimize overhead and maximize throughput
4. **Memory Efficiency**: Reduce memory footprint and improve scaling

## 🚀 **Current Performance Issues to Address**

### Import Time Problems
```python
# Current issue: Heavy imports happen immediately
import datawrangler as dw  # This currently loads sentence-transformers, sklearn, etc.
```

**Problem**: Users pay import cost even if they only use basic array operations.

### Missing Fast DataFrame Support
- **Current**: Only pandas DataFrames supported
- **Missing**: Polars (often 5-30x faster than pandas)
- **Opportunity**: Auto-detect and optimize for different DataFrame types

### Potential Inefficiencies
- Heavy dependencies loaded eagerly
- No optimization for different data sizes
- Missing parallelization opportunities
- No lazy evaluation patterns

## 📋 **Detailed Implementation Plan**

### 1. Lazy Import Architecture 🔧

#### Current State Analysis
```python
# Current imports in __init__.py and modules
from sentence_transformers import SentenceTransformer  # Heavy import
import sklearn  # Heavy import
import transformers  # Heavy import
```

#### Target Implementation
```python
# New pattern: Import only when needed
def _get_sentence_transformer():
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer
    except ImportError:
        raise ImportError("sentence-transformers not installed. Install with: pip install 'datawrangler[hf]'")

# Use in functions:
def apply_sentence_transformer(model_name, texts):
    SentenceTransformer = _get_sentence_transformer()
    model = SentenceTransformer(model_name)
    return model.encode(texts)
```

#### Files to Modify
- `datawrangler/__init__.py` - Remove heavy imports
- `datawrangler/zoo/text.py` - Add lazy import functions
- `datawrangler/zoo/format.py` - Update import patterns
- Create `datawrangler/util/lazy_imports.py` - Centralized lazy import utilities

#### Implementation Steps
1. **Audit all imports** - Identify heavy dependencies
2. **Create lazy import utilities** - Centralized import management
3. **Refactor text processing** - Lazy load sentence-transformers
4. **Refactor sklearn integration** - Lazy load sklearn models
5. **Update tests** - Ensure lazy imports work correctly
6. **Benchmark import times** - Measure improvement

### 2. Polars Integration 🚀

#### Research Phase
- **Polars capabilities**: What DataFrame operations are available?
- **Performance characteristics**: When is Polars faster than pandas?
- **API compatibility**: How similar is Polars to pandas?
- **Integration patterns**: How to detect and convert between formats?

#### Design Decisions
```python
# Auto-detection pattern
def detect_dataframe_type(df):
    if hasattr(df, '__module__') and 'polars' in df.__module__:
        return 'polars'
    elif hasattr(df, '__module__') and 'pandas' in df.__module__:
        return 'pandas'
    else:
        return 'unknown'

# Unified interface
def wrangle_dataframe_unified(df):
    df_type = detect_dataframe_type(df)
    if df_type == 'polars':
        return wrangle_polars_dataframe(df)
    else:
        return wrangle_pandas_dataframe(df)
```

#### Files to Create/Modify
- `datawrangler/zoo/polars_dataframe.py` - New module for Polars support
- `datawrangler/zoo/dataframe.py` - Update to detect and route DataFrame types
- `datawrangler/zoo/format.py` - Add Polars to format detection
- `requirements_polars.txt` - Optional Polars dependencies

#### Implementation Steps
1. **Add Polars detection utilities**
2. **Create polars_dataframe.py module**
3. **Implement basic Polars operations** (is_polars_dataframe, wrangle_polars_dataframe)
4. **Update format detection** to include Polars
5. **Create conversion utilities** (polars ↔ pandas)
6. **Add Polars examples** to tutorials
7. **Performance benchmarking** vs pandas

### 3. Performance Optimization 🔥

#### Profiling and Benchmarking
```python
# Create benchmarking suite
def benchmark_text_processing():
    """Compare sklearn vs sentence-transformers performance"""
    
def benchmark_dataframe_operations():
    """Compare pandas vs polars performance"""
    
def benchmark_import_times():
    """Measure lazy vs eager import performance"""
```

#### Optimization Targets
- **Import time**: Target <100ms for basic imports
- **Text processing**: Optimize for batch operations
- **Memory usage**: Minimize peak memory consumption
- **Parallelization**: Use multiprocessing where beneficial

#### Files to Create
- `benchmarks/import_performance.py` - Import time benchmarks
- `benchmarks/dataframe_performance.py` - DataFrame operation benchmarks
- `benchmarks/text_processing.py` - Text processing benchmarks
- `benchmarks/memory_usage.py` - Memory profiling

### 4. User Experience Enhancements 🎯

#### Progressive Feature Loading
```python
# Basic usage: Fast import, basic functionality
import datawrangler as dw
df = dw.wrangle([1, 2, 3])  # Fast, no heavy imports

# Advanced usage: Load heavy features on demand
text_df = dw.wrangle(texts, text_kwargs={'model': 'all-MiniLM-L6-v2'})  # Lazy loads sentence-transformers

# High performance: Use Polars automatically
import polars as pl
polars_df = pl.DataFrame({'A': [1, 2, 3]})
result = dw.wrangle(polars_df)  # Automatically uses Polars optimizations
```

#### Configuration for Performance
```python
# New configuration options
dw.config.set_performance_mode('fast')  # Prefer speed over accuracy
dw.config.set_dataframe_backend('polars')  # Force Polars usage
dw.config.set_lazy_imports(True)  # Enable lazy imports (default)
```

## 🔬 **Implementation Priority**

### Phase 1: Foundation (Week 1)
1. **Lazy import infrastructure** - Core utilities and patterns
2. **Import time optimization** - Immediate user experience improvement
3. **Basic benchmarking** - Establish performance baselines

### Phase 2: Polars Integration (Week 2)
1. **Polars detection and basic operations**
2. **Integration with existing format system**
3. **Conversion utilities**
4. **Performance comparisons**

### Phase 3: Advanced Optimization (Week 3)
1. **Memory usage optimization**
2. **Parallelization opportunities**
3. **Caching strategies**
4. **Advanced benchmarking**

### Phase 4: Documentation and Testing (Week 4)
1. **Performance documentation**
2. **Polars tutorial examples**
3. **Benchmarking results**
4. **Migration guide updates**

## 📊 **Success Metrics**

### Performance Targets
- **Import time**: <100ms (vs current ~2-5 seconds with heavy imports)
- **Basic operations**: 90% of current performance (minimal regression)
- **Polars operations**: 2-10x faster than pandas equivalents
- **Memory usage**: 20-50% reduction for large datasets

### User Experience Targets
- **Zero breaking changes** for existing users
- **Automatic optimization** without configuration changes
- **Clear performance guidance** in documentation
- **Smooth Polars adoption path**

## 🧪 **Testing Strategy**

### Performance Tests
```python
# Continuous performance monitoring
def test_import_time():
    """Ensure import stays under 100ms"""
    
def test_polars_performance():
    """Verify Polars is faster than pandas for target operations"""
    
def test_memory_usage():
    """Monitor memory consumption doesn't increase"""

def test_lazy_import_correctness():
    """Verify lazy imports work identically to eager imports"""
```

### Compatibility Tests
- **Existing functionality**: All current features work unchanged
- **Cross-backend compatibility**: Operations work with both pandas and Polars
- **Lazy import robustness**: Handle missing dependencies gracefully

## 🔧 **Technical Implementation Notes**

### Lazy Import Pattern
```python
# Utility function pattern
def _lazy_import(module_name, package=None, error_msg=None):
    def _import():
        try:
            return importlib.import_module(module_name, package)
        except ImportError:
            if error_msg:
                raise ImportError(error_msg)
            raise
    return _import

# Usage
_get_polars = _lazy_import('polars', error_msg="Polars not installed. Install with: pip install polars")

def use_polars():
    pl = _get_polars()
    return pl.DataFrame({'A': [1, 2, 3]})
```

### Polars Integration Pattern
```python
# Unified DataFrame interface
class DataFrameHandler:
    @staticmethod
    def is_dataframe(obj):
        return (hasattr(obj, 'shape') and 
                hasattr(obj, 'columns') and
                (is_pandas_dataframe(obj) or is_polars_dataframe(obj)))
    
    @staticmethod
    def to_pandas(df):
        if is_polars_dataframe(df):
            return df.to_pandas()
        return df
    
    @staticmethod
    def from_pandas(df, target_type='polars'):
        if target_type == 'polars':
            pl = _get_polars()
            return pl.from_pandas(df)
        return df
```

## 🎯 **Next Session Preparation**

### Code Review Checklist
1. **Current import patterns** - What gets imported when?
2. **Performance bottlenecks** - Where are the slow points?
3. **DataFrame usage patterns** - How are DataFrames currently handled?
4. **Test coverage** - What needs performance testing?

### Research Tasks
1. **Polars API study** - Learn Polars patterns and capabilities
2. **Benchmarking tools** - Set up performance measurement infrastructure
3. **Lazy import examples** - Study other libraries' lazy import patterns
4. **Memory profiling** - Tools and techniques for memory optimization

### Environment Setup
1. **Install Polars** - `pip install polars`
2. **Install benchmarking tools** - `pip install memory_profiler pytest-benchmark`
3. **Set up profiling** - Tools for measuring import and execution time

## 🚀 **Expected Outcomes**

After this optimization phase, data-wrangler will:

✅ **Import in <100ms** instead of 2-5 seconds  
✅ **Support Polars** for 2-10x performance improvements  
✅ **Use memory efficiently** with 20-50% reduction for large datasets  
✅ **Maintain full backward compatibility** with existing code  
✅ **Provide automatic optimization** without user configuration  
✅ **Include comprehensive benchmarks** showing performance improvements

This will position data-wrangler as not just a convenient tool, but a **high-performance** solution suitable for production data pipelines and large-scale data processing.

---

**Next Session Focus**: Begin with lazy import implementation, starting with the most impactful heavy dependencies (sentence-transformers, sklearn) and establishing the performance measurement infrastructure.