<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-07-03 | Updated: 2026-07-03 -->

# core

## Purpose
Configuration management and global backend state for the package. Parses `config.ini` into per-function default option dictionaries, injects those defaults into functions/classes, and holds the process-global DataFrame backend preference (pandas vs Polars). Also the source of truth for `__version__`.

## Key Files
| File | Description |
|-|-|
| `configurator.py` | Config + defaults engine. Defines `__version__` (`'0.4.0'`), `get_default_options()` (parse `config.ini`), `update_dict()` (merge template + overrides, optionally `eval`-ing config strings), `apply_defaults()` (decorator/wrapper injecting `config.ini` defaults by function name), and the backend accessors `set_dataframe_backend` / `get_dataframe_backend` / `reset_dataframe_backend`. On import it also creates the `~/.datawrangler/data/` cache dir. |
| `config.ini` | Declarative defaults: the format-checker priority list (`['dataframe','text','array','null']`), default backend, default text pipeline (`CountVectorizer` → `LatentDirichletAllocation`, corpus `minipedia`), sklearn vectorizer/decomposition params, sentence-transformers model list, imputer/interpolation defaults, and the data-cache path. |
| `__init__.py` | Re-exports `configurator` symbols (`__version__`, `get_default_options`, `update_dict`, etc.) for the rest of the package. |

## For AI Agents

### Working In This Directory
- **`config.ini` values are Python expressions.** Loaders `eval()` them (e.g. `np.nan`, `os.getenv('HOME')`, list literals). Keep expressions valid and side-effect-free.
- Section names map to function/class names; `apply_defaults` looks up defaults by `__name__`. Keys prefixed with `__` (e.g. `__model`) become **positional** args; other keys become keyword args.
- Bump `__version__` here in lockstep with `setup.py` on release.
- `set_dataframe_backend` only accepts `'pandas'` or `'polars'` (raises `ValueError` otherwise).

### Testing Requirements
- Covered by `tests/wrangler/test_core.py`. When adding a config section or changing defaults, add/adjust tests there.

### Common Patterns
- Config-as-code: behavior tuning lives in `config.ini`, read once into a module-level `defaults` dict, merged per-call via `update_dict`.

## Dependencies

### Internal
- Imported by nearly every other subpackage (`zoo`, `decorate`, `io`) for defaults and backend state.

### External
- `configparser` (stdlib), numpy (used inside `eval`-ed config expressions), sentence-transformers (lazily, optional)

<!-- MANUAL: Any manually added notes below this line are preserved on regeneration -->
