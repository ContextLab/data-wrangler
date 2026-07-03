<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-07-03 | Updated: 2026-07-03 -->

# tutorials

## Purpose
Executable Jupyter notebooks that teach `data-wrangler` feature-by-feature, rendered into the Sphinx docs site. They double as living examples and as an informal integration check (they must run end-to-end).

## Key Files
| File | Description |
|-|-|
| `wrangling_basics.ipynb` | Core `dw.wrangle` usage across data types |
| `core.ipynb` | Configuration and defaults (`config.ini`, backend selection) |
| `io.ipynb` | Loading/saving files and URLs |
| `util.ipynb` | Helper predicates and utilities |
| `decorators1.ipynb`, `decorators2.ipynb` | The `@funnel` decorator family, in two parts |
| `interpolation_and_imputation.ipynb` | Missing-value handling via the `interpolate` decorator |
| `polars_performance.ipynb`, `polars_advanced.ipynb`, `polars_benchmarks.ipynb` | The Polars backend: usage, advanced patterns, and speed comparisons |
| `real_world_examples.ipynb` | End-to-end applied walkthroughs |
| `tutorial_helpers.py` | Shared helper functions imported by the notebooks |

## For AI Agents

### Working In This Directory
- Notebooks `import datawrangler as dw` and must execute cleanly against the current source. When you change public API, update the affected notebooks (repo `CLAUDE.md`: update docs/examples alongside code).
- Keep shared logic in `tutorial_helpers.py` rather than duplicating it across notebooks.
- Notebooks may download models/corpora and hit the network; expect longer run times for the text and Polars-benchmark tutorials.

### Testing Requirements
- Correctness = every notebook runs top-to-bottom without errors and renders in the Sphinx build (`make docs`).

### Common Patterns
- One concept per notebook; prose + runnable cells; a shared `tutorial_helpers` module.

## Dependencies

### Internal
- `datawrangler`; linked from `docs/tutorials.rst`

### External
- Jupyter, pandas, polars, numpy; sklearn / sentence-transformers for the text examples

<!-- MANUAL: Any manually added notes below this line are preserved on regeneration -->
