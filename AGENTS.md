<!-- Generated: 2026-07-03 | Updated: 2026-07-03 -->

# data-wrangler

## Purpose
`data-wrangler` (published on PyPI as `pydata-wrangler`, imported as `datawrangler` / `dw`) is a Python package that turns messy data into clean `DataFrame` objects (or lists of them). It auto-detects data types (arrays, text, DataFrames, files, URLs, null/empty) and converts them into a consistent DataFrame format, with a special emphasis on text/NLP embedding via scikit-learn and sentence-transformers. As of v0.4.0 it supports **dual backends**: pandas (default) and Polars (2-100x faster on large data) selectable with `backend='polars'`.

## Key Files
| File | Description |
|-|-|
| `setup.py` | Package metadata; name `pydata-wrangler`, version `0.4.0`, extras `[hf]` (HuggingFace) and `[dev]` |
| `requirements.txt` | Core deps: pandas, polars>=0.20, numpy, scipy, scikit-learn, matplotlib, requests, six, dill, Pillow, tqdm |
| `requirements_hf.txt` | Optional NLP deps: torch, transformers, sentence-transformers, datasets, tokenizers, sentencepiece |
| `requirements_dev.txt` | Dev/test deps |
| `Makefile` | Dev tasks: `make test`, `make lint`, `make coverage`, `make docs`, `make dist`, `make install` |
| `tox.ini` | Multi-version test matrix (Python 3.9-3.12) |
| `setup.cfg` | flake8 / tooling config |
| `dev.yaml` | Conda environment for full ML setup (`conda env create -f dev.yaml`) |
| `HISTORY.rst` | Changelog |
| `README.rst` | Overview and quick-start examples |
| `CLAUDE.md` | Repo-specific guidance for AI agents |
| `.readthedocs.yaml` | ReadTheDocs build config |

## Subdirectories
| Directory | Purpose |
|-|-|
| `datawrangler/` | The Python package source (see `datawrangler/AGENTS.md`) |
| `tests/` | pytest suite + sample resources (see `tests/AGENTS.md`) |
| `docs/` | Sphinx documentation and Jupyter tutorials (see `docs/AGENTS.md`) |
| `benchmarks/` | Standalone pandas-vs-Polars and import-time benchmarks (see `benchmarks/AGENTS.md`) |
| `.github/` | GitHub Actions CI configuration (see `.github/AGENTS.md`) |
| `notes/` | Session/handoff notes (working documents, not part of the package) |
| `build/`, `dist/`, `pydata_wrangler.egg-info/` | Build artifacts — do not edit by hand |

## For AI Agents

### Working In This Directory
- The public API surface is intentionally tiny: `dw.wrangle`, `dw.funnel`, `dw.stack`, `dw.unstack`, `dw.__version__`. Preserve these entry points.
- Version lives in `datawrangler/core/configurator.py` (`__version__`) **and** `setup.py`. Keep them in sync on release.
- Model/vectorizer/imputer defaults live in `datawrangler/core/config.ini`, not in code. Change behavior there when possible.
- Per repo `CLAUDE.md`: update docs when changing tests/examples, update `requirements*.txt` when changing deps, and re-run the full check suite (tests + lint + docs) after any fix.

### Testing Requirements
- Run `make test` (pytest). Tests use **real** models, files, and network calls — never mocks. Many tests are parameterized over both `pandas` and `polars` backends via the `backend` fixture.
- Run `make lint` (flake8) and `make docs` (Sphinx build) before pushing. If any check triggers a change, re-run all checks.

### Common Patterns
- **Plugin/priority dispatch**: `zoo/format.py` iterates `format_checkers = ['dataframe', 'text', 'array', 'null']` and calls the first matching `is_<type>` / `wrangle_<type>` pair.
- **Config-driven defaults**: `core/config.ini` + `apply_defaults` inject defaults by function/class name.
- **Lazy imports**: heavy deps (torch, transformers, sklearn submodules, polars) load on first use via `util/lazy_imports.py` to keep import time low.

## Dependencies

### External
- pandas, numpy, scipy, scikit-learn, matplotlib (core)
- polars>=0.20.0 (required; high-performance backend)
- requests, dill, Pillow, six, tqdm (I/O and utilities)
- torch, transformers, sentence-transformers, datasets (optional, via `[hf]` extra)

<!-- MANUAL: Any manually added notes below this line are preserved on regeneration -->
