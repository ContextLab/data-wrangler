<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-07-03 | Updated: 2026-07-03 -->

# workflows

## Purpose
GitHub Actions CI workflow definitions that run the test suite on every push and pull request that touches package/test/build files.

## Key Files
| File | Description |
|-|-|
| `ci.yaml` | Workflow `wrangler-dev`. Triggers on push/PR affecting `datawrangler/**`, `tests/**`, `requirements*.txt`, `setup.*`, `tox.ini`, `Makefile`, `MANIFEST.in`, or the workflows themselves. Matrix-tests Python 3.9-3.12 on `ubuntu-latest`: installs `requirements.txt` + `requirements_hf.txt`, `pip install -e .`, then runs `pytest`. Includes a guard to avoid duplicate runs on fork PRs. |

## For AI Agents

### Working In This Directory
- CI installs the **`[hf]`** dependencies too, so text/NLP tests run in CI and can download models — keep them deterministic enough to pass on a clean runner.
- If you add a Python version, a new top-level config file, or a new dependency file, update both the `matrix` and the `paths` triggers here.
- Local pre-push parity: run `make test`, `make lint`, and `make docs` before pushing (repo `CLAUDE.md`).

### Testing Requirements
- The workflow's job is to run `pytest`. It does not currently run flake8 or the docs build — do those locally.

## Dependencies

### External
- GitHub Actions: `actions/checkout@v4`, `actions/setup-python@v4`

<!-- MANUAL: Any manually added notes below this line are preserved on regeneration -->
