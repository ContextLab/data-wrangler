<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-07-03 | Updated: 2026-07-03 -->

# docs

## Purpose
Sphinx documentation source for the project, published to ReadTheDocs (https://data-wrangler.readthedocs.io). Contains the reStructuredText pages, autodoc module stubs, a set of executable Jupyter tutorials, and static assets.

## Key Files
| File | Description |
|-|-|
| `conf.py` | Sphinx configuration (extensions, theme, autodoc settings) |
| `index.rst` | Documentation home page / table of contents |
| `api.rst`, `modules.rst`, `datawrangler*.rst` | Auto-generated API reference stubs (one per subpackage/module) |
| `installation.rst` | Install instructions |
| `migration_guide.rst` | Guidance for upgrading across versions (e.g. Polars backend) |
| `tutorials.rst` | Index page linking the notebook tutorials |
| `readme.rst`, `history.rst`, `authors.rst`, `contributing.rst` | Includes of the top-level RST docs |
| `requirements.txt` | Doc-build dependencies |
| `Makefile`, `make.bat` | Sphinx build entry points |
| `build.log` | Last build log (generated artifact) |

## Subdirectories
| Directory | Purpose |
|-|-|
| `tutorials/` | Executable Jupyter notebooks + `tutorial_helpers.py` (see `tutorials/AGENTS.md`) |
| `_static/` | Static assets for the HTML theme (currently empty) |
| `images/` | Logo/icon PNGs (`wrangler_logo.png`, `wrangler_icon.png`) |
| `_build/` | Generated HTML output — do not edit by hand |

## For AI Agents

### Working In This Directory
- Build docs with `make docs` from the repo root (runs Sphinx and opens the result). Per repo `CLAUDE.md`, a successful docs build is part of the pre-push checklist.
- The `datawrangler*.rst` files are API stubs consumed by autodoc; when you add/rename a public module or function, update the corresponding stub and the tutorial/prose that references it.
- Verify any web addresses you add — links must be manually checked (repo `CLAUDE.md`).

### Testing Requirements
- A clean Sphinx build (no warnings treated as errors where configured) is the bar. Notebooks under `tutorials/` should execute end-to-end.

### Common Patterns
- Prose docs are reStructuredText; runnable examples are Jupyter notebooks that import `datawrangler as dw`.

## Dependencies

### Internal
- Documents the `datawrangler` package API

### External
- Sphinx (+ theme/extensions listed in `requirements.txt`), Jupyter/nbsphinx for tutorials

<!-- MANUAL: Any manually added notes below this line are preserved on regeneration -->
