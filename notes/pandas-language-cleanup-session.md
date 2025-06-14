# Pandas Language Cleanup Session

Date: 2025-06-14

## Objective
Search for references to "pandas DataFrame", "pd.DataFrame", or other pandas-specific language in Python files and documentation that should be made backend-agnostic.

## Key Findings

### Files with Pandas-Specific Language

1. **Main Module Docstring** (`datawrangler/__init__.py`)
   - Line 2: "Transform messy data into clean pandas DataFrames" 
   - Line 19: `df = dw.wrangle(your_data)  # pandas DataFrame (default)`
   - Should be: "Transform messy data into clean DataFrames (pandas or Polars)"

2. **setup.py**
   - Line 33: `description="Wrangle messy data into pandas DataFrames..."`
   - Should be: "Wrangle messy data into DataFrames (pandas or Polars)..."

3. **Function Docstrings** - Multiple files have docstrings that mention converting to "pandas DataFrame" when they now support both backends:
   - `datawrangler/zoo/array.py` - `wrangle_array()`: "Turn an Array into a DataFrame (pandas or Polars)"
   - `datawrangler/zoo/null.py` - `wrangle_null()`: "Turn a null object (None or empty) into an empty DataFrame (pandas or Polars)"
   - `datawrangler/zoo/text.py` - `wrangle_text()`: Similar patterns
   - `datawrangler/decorate/decorate.py` - Various decorators mention pandas specifically

4. **Comments and Function Names**
   - `datawrangler/decorate/decorate.py`:
     - Functions `pandas_stack()` and `pandas_unstack()` - Names are pandas-specific but could work with Polars
     - Line 229: "coerces any data passed into the function into a DataFrame (pandas or Polars)"
     - Line 287: Reference to pandas.DataFrame.interpolate() documentation

5. **Documentation References**
   - Several files reference pandas-specific documentation URLs
   - Comments about "MultiIndex DataFrame" which is pandas-specific terminology

### Backend-Agnostic Language Already Present
Some files already use good backend-agnostic language:
- `datawrangler/zoo/format.py` - wrangle() docstring properly mentions both backends
- `datawrangler/zoo/dataframe.py` - is_dataframe() properly checks for both types

### Recommendations

1. **Update Module-Level Documentation**
   - Main `__init__.py` docstring should emphasize dual backend support from the start
   - setup.py description should be backend-agnostic

2. **Standardize Docstring Language**
   - Use "DataFrame (pandas or Polars)" when referring to output types
   - Use "DataFrame" alone when the backend is determined by parameter
   - Avoid "pandas DataFrame" unless specifically referring to pandas-only features

3. **Consider Function Naming**
   - `pandas_stack()` and `pandas_unstack()` could potentially be renamed to just `stack()` and `unstack()` if they support both backends
   - Or create backend-agnostic aliases while keeping originals for backward compatibility

4. **Update Examples**
   - Ensure examples show both backends where appropriate
   - Use comments to clarify which backend is being used

## Files to Review in Detail
Priority files that need backend-agnostic language updates:
1. `/Users/jmanning/data-wrangler/datawrangler/__init__.py`
2. `/Users/jmanning/data-wrangler/setup.py`
3. `/Users/jmanning/data-wrangler/datawrangler/decorate/decorate.py`
4. `/Users/jmanning/data-wrangler/datawrangler/zoo/array.py`
5. `/Users/jmanning/data-wrangler/datawrangler/zoo/null.py`
6. `/Users/jmanning/data-wrangler/datawrangler/zoo/text.py`

## Next Steps
- Update docstrings and comments to use backend-agnostic language
- Ensure all examples demonstrate both backends where relevant
- Consider creating aliases for pandas-specific function names
- Review and update any remaining documentation files