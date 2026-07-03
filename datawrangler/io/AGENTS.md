<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-07-03 | Updated: 2026-07-03 -->

# io

## Purpose
File and URL loading/saving with automatic format detection and transparent local caching of remote files. This is how `data-wrangler` ingests data from disk or the network before handing it to the zoo type handlers.

## Key Files
| File | Description |
|-|-|
| `io.py` | Core I/O. `load(x, dtype=None, **kwargs)` loads local paths or remote URLs, auto-detecting format by extension (text, pandas-supported tabular formats, numpy `.npy/.npz`, images via matplotlib, pickled objects via `dill`). `save(x, obj, dtype=None, **kwargs)` writes bytes/text/pickle/numpy. Remote files are hashed (`blake2b`) and cached under `~/.datawrangler/data/` (`get_local_fname`), with `load_remote` fetching over `requests`. |
| `panda_handler.py` | `load_dataframe(x, extension=None, ...)` — dispatches to the correct `pandas.read_*` function (csv, excel, json, html, xml, hdf, feather, parquet, orc, sas, spss, sql, gbq, stata, pkl) based on file extension; passes through DataFrame objects unchanged. |
| `extension_handler.py` | `get_extension(fname)` — returns the lowercase file extension, or `'dw'` when none can be determined. |
| `__init__.py` | Re-exports `load`, `save`, `load_dataframe`. |

## For AI Agents

### Working In This Directory
- The cache directory is `~/.datawrangler/data/`; it is created on package import (in `core/configurator.py`). Cached filenames are content-hashed from the source URL/path.
- Known `FIXME`s in `io.py`: the Google-Drive confirm-token path is unimplemented (raises), and the load-after-save path can create a duplicated local copy. Preserve or fix intentionally; don't paper over.
- To support a new tabular format, add a branch to `load_dataframe` **and** to the extension list in `io.load`'s helper. To support a new binary/image format, extend `img_types` or the numpy branch in `io.load`.

### Testing Requirements
- Covered by `tests/wrangler/test_io.py`, which loads real local files and remote URLs (from `tests/resources/`). Network access is required for the URL tests.

### Common Patterns
- Extension-driven dispatch to library readers; content-hash caching of remote resources; `dill` for arbitrary object (de)serialization.

## Dependencies

### Internal
- `core.configurator` (`get_default_options`, cache path), used by `zoo` and `util` for loading inputs

### External
- pandas (readers), requests (HTTP), dill (pickle), numpy, matplotlib/Pillow (images)

<!-- MANUAL: Any manually added notes below this line are preserved on regeneration -->
