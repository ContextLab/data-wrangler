<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-07-03 | Updated: 2026-07-03 -->

# wrangler (tests)

## Purpose
The pytest modules for `data-wrangler`, one per package subpackage, plus the shared fixtures in `conftest.py`. Tests run against real models, files, and network resources and are largely parameterized across the pandas and Polars backends.

## Key Files
| File | Description |
|-|-|
| `conftest.py` | Shared fixtures: `resources` dir, local + remote CSV (`data_file`/`data_url`), image (`img_file`/`img_url`), and text (`text_file`/`text_url`); a parsed `data` DataFrame; a `backend` fixture parameterized over `['pandas','polars']`; and helpers `assert_backend_type` and `assert_dataframes_equivalent`. |
| `test_zoo.py` | Largest module (~21 tests): array/text/dataframe/null detection and wrangling across both backends. |
| `test_decorate.py` | Tests for `funnel`, `interpolate`, stacking, and list generalization (~6 tests). |
| `test_util.py` | Tests for `btwn`, `array_like`, `dataframe_like`, `depth`, and lazy importers (~4 tests). |
| `test_core.py` | Tests for config parsing, defaults injection, and backend state (~3 tests). |
| `test_io.py` | Tests for `load`/`save` with local and remote files (~2 tests). |
| `__init__.py` | Marks the test package. |

## For AI Agents

### Working In This Directory
- Run all: `pytest` (or `make test`). Single test: `pytest tests/wrangler/test_zoo.py::test_name`.
- Reuse `conftest.py` fixtures instead of hardcoding paths/URLs. Use the `backend` fixture + `assert_backend_type` / `assert_dataframes_equivalent` to cover both backends.
- Do **not** weaken or mock a failing test to make it pass (repo `CLAUDE.md`). Fix the code; if stuck, note the problem and commit first.

### Testing Requirements
- Network access is needed for the `*_url` fixtures and for downloading sklearn corpora / sentence-transformers models used by text tests.

### Common Patterns
- Backend-parameterized tests; equivalence assertions that normalize Polars→pandas before comparing values.

## Dependencies

### Internal
- `datawrangler` (the code under test), `tests/resources/` (sample data)

### External
- pytest, pandas, polars, numpy

<!-- MANUAL: Any manually added notes below this line are preserved on regeneration -->
