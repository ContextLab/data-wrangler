<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-07-03 | Updated: 2026-07-03 -->

# zoo

## Purpose
The heart of the package: the "zoo" of data-type handlers plus the `wrangle()` orchestrator. Each supported data type provides an `is_<type>` detector and a `wrangle_<type>` converter; `format.py` dispatches to them in priority order. This is where new data types are added.

## Key Files
| File | Description |
|-|-|
| `format.py` | The dispatcher. Defines `wrangle(x, return_dtype=False, backend=None, **kwargs)`. Reads the priority list `format_checkers = ['dataframe','text','array','null']` from `config.ini`, then for each item calls the first matching `is_<type>` and runs `wrangle_<type>`. Handles per-type `*_kwargs`, model pre-fitting/reuse across a list of inputs, and list/nested-list inputs. |
| `dataframe.py` | `is_dataframe`, `is_multiindex_dataframe`, `wrangle_dataframe`. Detects pandas/Polars/modin DataFrames and dataframe-like objects; routes to the Polars handler or converts between backends per `backend`. |
| `array.py` | `is_array`, `is_number`, `wrangle_array`. Coerces numbers/arrays/sparse matrices/files into 2-D DataFrames (stacking >2-D arrays); builds pandas or Polars output based on `backend`. |
| `text.py` | **Active** text handler (735 lines, lazy imports). Public: `is_text`, `wrangle_text`, `get_corpus`, `apply_text_model`, `get_text_model`, `to_str_list`, `get_text`. Embeds text via sklearn vectorizers/decomposition and sentence-transformers / HuggingFace models; supports the simplified string/list `model` API. |
| `polars_dataframe.py` | Polars support: `is_polars_dataframe`, `is_polars_lazyframe`, `wrangle_polars_dataframe`, `create_polars_dataframe`, and `pandas_to_polars` / `polars_to_pandas` converters. |
| `null.py` | `is_null`, `wrangle_null`. Turns `None`/empty inputs into an empty pandas or Polars DataFrame. |
| `__init__.py` | Curates the public zoo namespace: `wrangle`, the `is_*`/`wrangle_*` pairs, and text helpers. |

## For AI Agents

### Working In This Directory
- **To add a data type**: implement `is_<type>(obj)` and `wrangle_<type>(obj, return_model=False, backend=None, **kwargs)`, export them in `__init__.py`, and add `<type>` to `supported_formats.types` in `core/config.ini` at the right priority. **Order matters** — earlier checkers win, so put more specific types first.
- `wrangle_<type>` functions share a contract: accept `return_model` (return `(df, model)` when True, where `model` is a `{'model', 'args', 'kwargs'}` dict reusable on new data) and `backend` (`'pandas'`/`'polars'`/`None`).
- `text.py` is the sole text handler (the former `text_lazy.py` / `text_original.py` duplicates were removed in v0.5.0).
- Circular-import guard: `array.py` imports `is_array` lazily elsewhere; keep new cross-module imports cycle-free.

### Testing Requirements
- `tests/wrangler/test_zoo.py` holds the bulk of the suite (~21 tests), parameterized over both backends. Text tests download real sklearn corpora and sentence-transformers models. Add coverage for any new detector/converter across both backends.

### Common Patterns
- Priority-based `is_`/`wrangle_` dispatch; model dicts (`{'model', 'args', 'kwargs'}`) for reproducible re-application; lazy imports for heavy NLP deps.

## Dependencies

### Internal
- `core` (defaults, `update_dict`, backend state), `io` (`load`, `load_dataframe`), `util` (`array_like`, `depth`, lazy importers)

### External
- pandas, numpy, polars; scikit-learn, sentence-transformers, transformers, torch, datasets (text, lazily loaded / optional)

<!-- MANUAL: Any manually added notes below this line are preserved on regeneration -->
