# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview
**data-wrangler** turns messy data (arrays, text, DataFrames, files, URLs, null/empty, and nested lists of these) into clean `DataFrame` objects, with a special emphasis on text/NLP embedding. Published on PyPI as `pydata-wrangler`, imported as `datawrangler` / `dw`. As of v0.4.0 every operation supports **two backends**: pandas (default) and Polars (`backend='polars'`, 2-100x faster on large data).

Per-directory `AGENTS.md` files exist throughout the tree with deeper, localized notes — consult the one nearest the code you're editing.

## Development Commands
- `make test` — run the pytest suite (default Python)
- `pytest tests/wrangler/test_zoo.py::test_name` — run a single test
- `make test-all` — tox across Python 3.9-3.12
- `make lint` — flake8
- `make coverage` — coverage report
- `make docs` — build Sphinx docs (also runs the tutorial notebooks)
- `make dist` / `make install` / `make clean` — build / install / clean artifacts

Full dev install (text/NLP tests need the `[hf]` extra): `pip install -e ".[hf]"`, or `conda env create -f dev.yaml`.

**Tests use real resources, not mocks**: they download sklearn corpora and sentence-transformers models and fetch remote fixtures over HTTP, so a full run needs network access. Most DataFrame-producing tests are parameterized over both backends via the `backend` fixture in `tests/wrangler/conftest.py` (use its `assert_backend_type` / `assert_dataframes_equivalent` helpers).

## Architecture

The package is essentially **one priority-ordered dispatch loop** wrapped in decorators and fed by an I/O layer.

- **Dispatch (`zoo/format.py`)**: `wrangle(x, backend=None, **kwargs)` reads the ordered list `format_checkers = ['dataframe','text','array','null']` from `config.ini`, then calls the first matching `is_<type>(x)` and runs `wrangle_<type>(x, ...)`. **Order is priority** — earlier checkers win. Per-type options are passed as `<type>_kwargs` (e.g. `text_kwargs={...}`). For list inputs, a fitted model from the first element is reused across the rest.

- **The wrangler contract** (shared by every `wrangle_<type>` in `zoo/`, and relied on by `decorate/`): each accepts `return_model=False` and `backend=None`. When `return_model=True` it returns `(df, model)` where `model` is a reusable `{'model', 'args', 'kwargs'}` dict — pass it back later (e.g. `text_kwargs={'model': fitted}`) to apply the same fitted transform to new data.

- **Dual backend**: the `backend` argument threads from `wrangle` into every wrangler. pandas↔Polars conversion and Polars detection/handling live in `zoo/polars_dataframe.py` (`pandas_to_polars`, `polars_to_pandas`, `is_polars_dataframe`, `create_polars_dataframe`). A process-global default lives in `core/configurator.py` (`set_dataframe_backend` / `get_dataframe_backend` / `reset_dataframe_backend`). Predicate checks in `util/helpers.py` (`array_like`, `dataframe_like`) are **duck-typed by attribute**, not `isinstance`, so pandas- and Polars-like objects both pass.

- **Config-as-code (`core/config.ini` + `core/configurator.py`)**: model/vectorizer/imputer defaults, the format-checker order, and the data-cache path all live in `config.ini`. **Its values are Python expressions that get `eval()`-ed** (e.g. `np.nan`, list literals, `os.path.expanduser('~')`). `apply_defaults` injects a section's defaults into a function/class **by its `__name__`**; keys prefixed with `__` become positional args, others become keyword args.

- **Lazy imports (`util/lazy_imports.py`)**: heavy deps (torch, transformers, sentence-transformers, sklearn submodules, polars) load on first use via cached `get_*` importers, keeping `import datawrangler` fast (see `benchmarks/import_time.py`). Add new heavy dependencies as `get_*` lazy importers here — never as top-level imports.

- **I/O (`io/`)**: `load`/`save` auto-detect format by extension; remote URLs are content-hashed (`blake2b`) and cached under `~/.datawrangler/data/` (created on import). `panda_handler.load_dataframe` maps extensions to `pandas.read_*`.

- **Decorators (`decorate/decorate.py`)**: `@funnel` runs `zoo.wrangle` on a function's input so the function can assume clean DataFrame input; `interpolate` fills missing values using imputer/interpolation defaults from `config.ini`; `pandas_stack`/`pandas_unstack` (exported as `dw.stack`/`dw.unstack`) concat/split lists of DataFrames via a MultiIndex.

## Adding New Data Types
1. Implement `is_<type>(obj)` and `wrangle_<type>(obj, return_model=False, backend=None, **kwargs)` in a `zoo/` module (honor the wrangler contract above).
2. Export both in `zoo/__init__.py`.
3. Register `<type>` in `supported_formats.types` in `core/config.ini` at the correct **priority position**.

## Gotchas
- `__version__` lives in **both** `core/configurator.py` and `setup.py` — keep them in sync on release.
- HuggingFace corpora in `zoo/text.py`/`get_corpus` must use full `namespace/name` ids (e.g. `wikimedia/wikipedia`, `cam-cst/cbt`); bare legacy names (`wikipedia`, `cbt`) break on `datasets` >= 4.
- `array_like('some string')` will try to `load()` it as a file/URL (network/FS side-effect) unless `force_literal=True`.

## Dependencies
- **Core**: pandas, numpy, scipy, scikit-learn, matplotlib, polars>=0.20 (required), requests, dill, Pillow, six, tqdm
- **Optional NLP (`[hf]` extra / `requirements_hf.txt`)**: torch, transformers, sentence-transformers, datasets, tokenizers
- **Dev (`requirements_dev.txt`)**: pytest, Sphinx, flake8
