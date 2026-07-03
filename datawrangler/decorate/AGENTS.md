<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-07-03 | Updated: 2026-07-03 -->

# decorate

## Purpose
Function decorators that make ordinary functions "DataFrame-aware." The headline decorator `@funnel` auto-wrangles a function's first argument into a DataFrame before the function runs, so functions can be written to assume clean DataFrame input. Also provides missing-value interpolation, list generalization, stack/unstack helpers, and sklearn-model application utilities.

## Key Files
| File | Description |
|-|-|
| `decorate.py` | All decorators and helpers. Public (via `__init__.py`): `list_generalizer`, `funnel`, `interpolate`, `apply_stacked`, `apply_unstacked`. Also `pandas_stack` / `pandas_unstack` (re-exported at top level as `dw.stack` / `dw.unstack`). Internal helpers: `import_sklearn_models`, `get_sklearn_model`, `apply_sklearn_model`, and lazy model-list builders (`_get_reduce_models`, `_get_text_vectorizers`, `_get_impute_models`). |
| `__init__.py` | Re-exports `list_generalizer`, `funnel`, `interpolate`, `apply_stacked`, `apply_unstacked`. |

## For AI Agents

### Working In This Directory
- `funnel` calls `zoo.wrangle` on the incoming data, so decorator behavior depends on the zoo dispatch order and `*_kwargs` conventions — keep them consistent.
- `interpolate` fills missing values using sklearn imputers / pandas interpolation whose defaults come from `core/config.ini` (`[impute]`, `[interpolate]`, `[SimpleImputer]`, etc.).
- `pandas_stack` / `pandas_unstack` operate on lists of DataFrames using a MultiIndex to concatenate/split them; `apply_stacked` / `apply_unstacked` run a function on the stacked vs per-frame view. Polars support in these decorators is more limited than pandas (see package docstring).
- There is a known `FIXME` noting `apply_sklearn_model` partially duplicates `zoo.text.apply_text_model`; prefer consolidating over further divergence.

### Testing Requirements
- Covered by `tests/wrangler/test_decorate.py`. Test decorated functions against arrays, text, DataFrames, and lists thereof; cover both backends where a DataFrame is produced.

### Common Patterns
- Decorators wrap with `functools.wraps`, normalize input via `zoo.wrangle`, and support a `return_model` passthrough so fitted models (imputers, vectorizers) can be reused.

## Dependencies

### Internal
- `zoo` (`wrangle`, type detection), `core` (`get_default_options`, `apply_defaults`, `update_dict`), `util` (helpers)

### External
- pandas, numpy; scikit-learn (imputers, decomposition, feature-extraction — lazily loaded)

<!-- MANUAL: Any manually added notes below this line are preserved on regeneration -->
