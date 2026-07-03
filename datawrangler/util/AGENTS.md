<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-07-03 | Updated: 2026-07-03 -->

# util

## Purpose
Shared low-level utilities: duck-typing predicates used by the zoo detectors, and the lazy-import infrastructure that keeps `import datawrangler` fast by deferring heavy dependencies (torch, transformers, sklearn submodules, polars) until first use.

## Key Files
| File | Description |
|-|-|
| `helpers.py` | Predicate/utility helpers: `btwn` (inclusive range test), `dataframe_like` (duck-types an object against ~50 DataFrame methods), `array_like` (array/DataFrame/list detection, optionally resolving strings as files/URLs), `depth` (max nesting depth of a list/array). |
| `lazy_imports.py` | Lazy-loading engine: `LazyModule`, `lazy_import`, `lazy_import_with_fallback` (custom ImportError message), `requires_import` (decorator asserting deps), plus pre-built importers `get_sklearn`, `get_numpy`, `get_pandas`, `get_polars`, `get_torch`, `get_transformers`, `get_sentence_transformers`, `get_datasets`, and sklearn-submodule importers. Importers cache their result after first call. |
| `__init__.py` | Re-exports the helpers and all `get_*` lazy importers. |

## For AI Agents

### Working In This Directory
- `array_like` / `dataframe_like` are **duck-typing** checks (attribute presence), not `isinstance` checks — this is deliberate so pandas-like and Polars-like objects both pass. Preserve that when editing.
- `array_like(x)` for a string will attempt to `load` it as a file/URL unless `force_literal=True`. Be mindful of the network/filesystem side-effect.
- Add heavy dependencies as new `get_*` lazy importers here rather than importing them at module top-level anywhere in the package — this is what keeps startup fast (see `benchmarks/import_time.py`).
- `helpers.py` uses a local lazy function (`_get_is_array`) to avoid a circular import with `zoo.array`.

### Testing Requirements
- Covered by `tests/wrangler/test_util.py`. Test predicates against edge cases (empty, nested, scalar, string-as-path) and confirm lazy importers return the real modules.

### Common Patterns
- Duck-typing over `isinstance`; cached lazy importers with optional install-hint fallback messages pointing to the `[hf]` extra.

## Dependencies

### Internal
- `io.load` (used by `array_like` to resolve string paths). Imported broadly by `zoo` and `decorate`.

### External
- numpy, pandas (predicates); everything else is imported lazily on demand

<!-- MANUAL: Any manually added notes below this line are preserved on regeneration -->
