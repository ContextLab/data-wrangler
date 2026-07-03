<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-07-03 | Updated: 2026-07-03 -->

# resources (test fixtures)

## Purpose
Static sample data files used by the test suite. Each is exposed both as a local path and as a raw-GitHub URL via fixtures in `tests/wrangler/conftest.py`, so the same content exercises both local-file and remote-download code paths.

## Key Files
| File | Description |
|-|-|
| `testdata.csv` | Tabular sample loaded as a pandas DataFrame (`data_file` / `data_url` fixtures; parsed with `index_col=0`). |
| `home_on_the_range.txt` | Plain-text sample for text-loading and text-embedding tests (`text_file` / `text_url`). |
| `wrangler.jpg` | Image sample for image-loading tests (`img_file` / `img_url`). |

## For AI Agents

### Working In This Directory
- These files are referenced by URL as `https://raw.githubusercontent.com/ContextLab/data-wrangler/main/tests/resources/<file>`. If you rename or move a file, update `conftest.py` **and** be aware remote tests hit the `main` branch on GitHub — the change only takes effect for URL tests after it is pushed to `main`.
- Keep fixtures small and free of sensitive data (they are public on GitHub).

### Testing Requirements
- No tests live here; this directory only supplies inputs consumed by `tests/wrangler/`.

## Dependencies

### Internal
- Consumed by `tests/wrangler/conftest.py` fixtures

<!-- MANUAL: Any manually added notes below this line are preserved on regeneration -->
