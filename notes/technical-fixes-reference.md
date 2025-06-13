# Technical Fixes Reference - v0.3.0

Quick reference for the critical fixes implemented in this release.

## 🔧 Fix #1: Sklearn Model Detection Bug

**File**: `datawrangler/zoo/text.py:79`

**Problem**: Strings were incorrectly identified as HuggingFace models because they have `.encode()` method

**Before**:
```python
def is_hugging_face_model(x):
    # ... other checks ...
    
    # Check for encode method (sentence-transformers interface)
    return hasattr(x, 'encode')
```

**After**:
```python
def is_hugging_face_model(x):
    # ... other checks ...
    
    # Check for encode method (sentence-transformers interface) but not strings
    return hasattr(x, 'encode') and not isinstance(x, str)
```

**Impact**: 
- ✅ Sklearn models like 'CountVectorizer' now correctly detected
- ✅ CI test `test_wrangle_text_sklearn` now passes
- ✅ No impact on legitimate sentence-transformers usage

---

## 🔧 Fix #2: Pandas 2.0+ Compatibility

**File**: `datawrangler/util/helpers.py:50`

**Problem**: `iteritems` method removed in pandas 2.0+

**Before**:
```python
required_attributes = ['values', 'index', 'columns', 'shape', 'stack', 'unstack', 'loc', 'iloc', 'size', 'copy',
                       'head', 'tail', 'items', 'iteritems', 'keys', 'iterrows', 'itertuples',
                       # ... more attributes ...
                       ]
```

**After**:
```python
required_attributes = ['values', 'index', 'columns', 'shape', 'stack', 'unstack', 'loc', 'iloc', 'size', 'copy',
                       'head', 'tail', 'items', 'keys', 'iterrows', 'itertuples',
                       # ... more attributes ...
                       ]
```

**Impact**:
- ✅ `dataframe_like()` now works with pandas 2.0+
- ✅ Tests `test_dataframe_like` now pass
- ✅ Maintains functionality (items() is the replacement for iteritems())

---

## 🧪 Testing Commands

**Test specific fixes**:
```bash
# Test sklearn model detection
python -m pytest tests/wrangler/test_zoo.py::test_wrangle_text_sklearn -v

# Test pandas compatibility  
python -m pytest tests/wrangler/test_util.py::test_dataframe_like -v
python -m pytest tests/wrangler/test_zoo.py::test_dataframe_like -v

# Test sentence-transformers still works
python -m pytest tests/wrangler/test_zoo.py::test_wrangle_text_hugging_face -v
```

**Full test suite**:
```bash
python -m pytest tests/ -v
```

---

## 🔍 Debugging Techniques Used

**For model detection bug**:
```python
# Check what get_text_model returns
from datawrangler.zoo.text import get_text_model
result = get_text_model('CountVectorizer')
print(f"Result: {result}, Type: {type(result)}")

# Check individual function behavior
from datawrangler.zoo.text import is_sklearn_model, is_hugging_face_model
print(f"is_sklearn_model: {is_sklearn_model(result)}")
print(f"is_hugging_face_model: {is_hugging_face_model(result)}")
```

**For pandas compatibility**:
```python
# Check what's missing from DataFrames
from datawrangler.util.helpers import dataframe_like
import pandas as pd

data = pd.DataFrame({'A': [1, 2, 3]})
print(f"Is dataframe_like: {dataframe_like(data, debug=True)}")
```

---

## 📦 Verification in Built Package

**Check fixes are included**:
```python
import zipfile

# Verify sklearn fix
with zipfile.ZipFile('dist/pydata_wrangler-0.3.0-py2.py3-none-any.whl', 'r') as z:
    content = z.read('datawrangler/zoo/text.py').decode('utf-8')
    assert 'not isinstance(x, str)' in content
    
# Verify pandas fix  
with zipfile.ZipFile('dist/pydata_wrangler-0.3.0-py2.py3-none-any.whl', 'r') as z:
    content = z.read('datawrangler/util/helpers.py').decode('utf-8')
    assert 'iteritems' not in content
```

---

## ⚠️ Important Notes

1. **Order matters**: Model detection checks sklearn first, then sentence-transformers
2. **String encoding**: The `.encode()` method exists on all Python strings for character encoding
3. **Pandas deprecation**: `iteritems()` → `items()` is the official migration path
4. **Testing**: Always verify fixes work in built packages, not just source code

---

## 🔗 Related Files

**Primary fixes**:
- `datawrangler/zoo/text.py` - Model detection logic
- `datawrangler/util/helpers.py` - DataFrame utilities

**Testing**:
- `tests/wrangler/test_zoo.py` - Text processing tests
- `tests/wrangler/test_util.py` - Utility function tests

**Configuration**:
- `setup.py` - Package metadata and dependencies
- `requirements_hf.txt` - HuggingFace optional dependencies