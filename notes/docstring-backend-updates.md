# Docstring Backend Updates Summary

## Date: 2025-06-14

### Task
Updated docstrings in datawrangler/zoo/*.py files to be backend-agnostic, removing pandas-specific language and adding examples showing both pandas and Polars backends.

### Files Updated

1. **text.py**
   - Updated `wrangle_text` return type description to mention "pandas or Polars based on backend"
   - Added examples showing both pandas and Polars usage
   - Updated `to_str_list` docstring to capitalize "DataFrames"

2. **null.py**
   - Added examples showing both pandas and Polars usage for `wrangle_null`

3. **array.py**
   - Added examples showing both pandas and Polars usage for `wrangle_array`
   - Included example with custom column names

4. **format.py**
   - Updated parameter description to be more generic: "DataFrames (pandas or Polars)" instead of "Pandas DataFrames"
   - Enhanced examples with better comments explaining each usage

5. **dataframe.py**
   - Updated `wrangle_dataframe` kwargs description to mention both backends
   - Added comprehensive examples showing type preservation and conversion
   - Updated `is_multiindex_dataframe` return description to be backend-agnostic

### Key Patterns Applied

1. **Consistent Language**: Changed "pandas DataFrame" to "DataFrame (pandas or Polars)"
2. **Backend Examples**: Added examples for each function showing both `backend='pandas'` (default) and `backend='polars'`
3. **Clear Comments**: Added descriptive comments in examples to explain what each example demonstrates
4. **Return Descriptions**: Updated return type descriptions to explicitly mention backend dependency

### Notes
- The docstrings now properly reflect the dual-backend support
- Examples demonstrate the simplified API introduced in recent updates
- All changes maintain backward compatibility while promoting backend flexibility