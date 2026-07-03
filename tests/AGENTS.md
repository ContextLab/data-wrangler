<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-07-03 | Updated: 2026-07-03 -->

# tests

## Purpose
The pytest test suite for `data-wrangler`, plus sample data resources used by the tests. Tests exercise **real** functionality — real models, real files, real network calls (no mocks) — and are largely parameterized across both the pandas and Polars backends.

## Key Files
| File | Description |
|-|-|
| `__init__.py` | Marks the test package |

## Subdirectories
| Directory | Purpose |
|-|-|
| `wrangler/` | The actual test modules and shared fixtures (`conftest.py`) (see `wrangler/AGENTS.md`) |
| `resources/` | Sample data files (CSV, text, image) loaded by fixtures (see `resources/AGENTS.md`) |

## For AI Agents

### Working In This Directory
- Run the suite with `make test` or `pytest` from the repo root.
- Run a single test: `pytest tests/wrangler/test_zoo.py::test_function`.
- Never introduce mocks or simplify a failing test to make it pass (per repo `CLAUDE.md`). Fix the code instead; if blocked, take notes and commit before continuing.

### Testing Requirements
- Some tests download models (sentence-transformers, sklearn corpora) and fetch remote fixtures over HTTP — network access is required for a full run.
- New features must add tests to the matching `test_<subpackage>.py` and, where a DataFrame is produced, cover both backends.

### Common Patterns
- Shared fixtures live in `wrangler/conftest.py` (local/remote CSV, text, and image resources; a `backend` fixture parameterized over `['pandas', 'polars']`; and `assert_backend_type` / `assert_dataframes_equivalent` helpers).

## Dependencies

### Internal
- `datawrangler` (the package under test)

### External
- pytest, pandas, polars, numpy

<!-- MANUAL: Any manually added notes below this line are preserved on regeneration -->
