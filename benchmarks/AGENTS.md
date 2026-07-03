<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-07-03 | Updated: 2026-07-03 -->

# benchmarks

## Purpose
Standalone performance scripts that quantify the two headline claims of v0.4.0: the pandas-vs-Polars speedup and the reduced import time from lazy loading. These are runnable scripts, not part of the installed package or the pytest suite.

## Key Files
| File | Description |
|-|-|
| `dataframe_performance.py` | Times `dw.wrangle` (and DataFrame ops) across the pandas and Polars backends over varying data sizes to demonstrate the 2-100x speedup |
| `import_time.py` | Measures `import datawrangler` startup time (via subprocess + `statistics`) to validate the lazy-import optimization |

## For AI Agents

### Working In This Directory
- Run directly, e.g. `python benchmarks/dataframe_performance.py`. Each script inserts the repo root on `sys.path`, so it runs against the working-tree source without installation.
- These scripts print human-readable timing tables; they are not asserted in CI. If you change import structure or backend dispatch, re-run them to confirm the performance claims still hold.

### Testing Requirements
- No automated assertions. Correctness here means the scripts run cleanly and report sensible numbers on the current machine.

### Common Patterns
- Timing via `time` / `statistics`; import-time measured in a fresh subprocess to avoid caching effects.

## Dependencies

### Internal
- `datawrangler` (imported for benchmarking)

### External
- pandas, polars, numpy

<!-- MANUAL: Any manually added notes below this line are preserved on regeneration -->
