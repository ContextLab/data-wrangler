<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-07-03 | Updated: 2026-07-03 -->

# datawrangler

## Purpose
The importable Python package (`import datawrangler as dw`). This top-level module re-exports the public API and organizes the implementation into four subpackages: `core` (config + backend state), `zoo` (data-type detection/conversion), `decorate` (function decorators), `io` (file/URL loading), and `util` (helpers + lazy imports).

## Key Files
| File | Description |
|-|-|
| `__init__.py` | Public API. Re-exports `wrangle` (from `zoo`), `funnel`/`stack`/`unstack` (from `decorate.decorate`), and `__version__` (from `core`). Contains the package-level docstring documenting pandas vs Polars trade-offs. |

## Subdirectories
| Directory | Purpose |
|-|-|
| `core/` | Config parsing (`config.ini`), default injection, and global backend state (see `core/AGENTS.md`) |
| `zoo/` | Data-type handlers — `is_<type>` / `wrangle_<type>` pairs and the `wrangle()` orchestrator (see `zoo/AGENTS.md`) |
| `decorate/` | The `@funnel` decorator family and stacking helpers (see `decorate/AGENTS.md`) |
| `io/` | `load` / `save` with local caching + remote fetch, and format-specific readers (see `io/AGENTS.md`) |
| `util/` | Predicate helpers (`array_like`, `dataframe_like`, `depth`) and lazy-import infrastructure (see `util/AGENTS.md`) |

## For AI Agents

### Working In This Directory
- `__init__.py` defines the entire public API. Adding a new public symbol means exporting it here and documenting it in `docs/`.
- Note the aliases: `dw.stack` == `decorate.decorate.pandas_stack`, `dw.unstack` == `pandas_unstack`.
- Import order matters: `core` is imported by nearly everything; avoid introducing circular imports (see `util/helpers.py` and `zoo/array.py` for existing lazy-import workarounds against cycles).

### Testing Requirements
- Every subpackage has matching tests in `tests/wrangler/test_<subpackage>.py`. Changing a subpackage requires updating/running its test file.

### Common Patterns
- Subpackage `__init__.py` files curate a flat public namespace (e.g. `zoo/__init__.py` surfaces `is_*`/`wrangle_*` functions from the type modules).

## Dependencies

### Internal
- `core` ← imported by `zoo`, `decorate`, `io`
- `util` ← imported broadly for predicates and lazy imports
- `zoo.format` ← the central dispatcher wiring the type handlers together

### External
- pandas, numpy, polars (backends); scikit-learn, sentence-transformers (text embedding, optional)

<!-- MANUAL: Any manually added notes below this line are preserved on regeneration -->
