# Supply-Chain Security

This project uses **uv** exclusively for Python dependency management.
`pip install` and `conda` are not used anywhere in the project.

---

## Why uv-only?

| Threat | pip | uv (this project) |
|--------|-----|-----|
| Typosquatting / dependency confusion | No built-in protection | Locked to exact packages via `uv.lock` |
| Malicious new release | Installs latest by default | `exclude-newer` quarantine prevents resolution of packages < 9 days old |
| Hash mismatch (tampered package) | Requires opt-in `--require-hashes` | `uv.lock` contains SHA-256 hashes for every wheel; verified automatically |
| Fallback to insecure installer | Common in Dockerfiles (`\|\| pip install …`) | No fallback — build fails loud if uv fails |

---

## Key mechanisms

### 1. Lockfile with hashes (`uv.lock`)

Every dependency (direct and transitive) is pinned to an exact version with
SHA-256 content hashes. `uv sync` verifies hashes on every install — a
tampered wheel is rejected immediately.

### 2. Nine-day quarantine (`exclude-newer`)

In `pyproject.toml`:

```toml
[tool.uv]
exclude-newer = "2026-03-19T00:00:00Z"
```

`uv` will not resolve any package version published after this date.
This gives the community time to detect and yank malicious releases before
they can enter the dependency tree.

**To update the quarantine window** (e.g. when upgrading dependencies):

```bash
# Set exclude-newer to 9 days ago
NEW_DATE=$(date -u -d '9 days ago' +%Y-%m-%dT00:00:00Z)
sed -i "s/^exclude-newer = .*/exclude-newer = \"$NEW_DATE\"/" pyproject.toml
uv lock
```

### 3. No pip anywhere

- **CI** (`ci.yml`): uses `astral-sh/setup-uv` action; all installs via `uv sync`.
- **Docker** (`Dockerfile`): uv installed via the official curl installer, not pip.
  No `pip install` fallback — the build fails if `uv sync` fails.
- **Docs**: all setup instructions reference `uv sync` only.

### 4. Explicit PyTorch index

The PyTorch CUDA wheels come from `https://download.pytorch.org/whl/cu128`,
configured as an `explicit = true` index in `pyproject.toml`. This prevents
accidental resolution from PyPI (where a similarly-named package could appear).

---

## Updating dependencies

```bash
# 1. Update the quarantine date (9 days ago)
NEW_DATE=$(date -u -d '9 days ago' +%Y-%m-%dT00:00:00Z)
sed -i "s/^exclude-newer = .*/exclude-newer = \"$NEW_DATE\"/" pyproject.toml

# 2. Upgrade a specific package
uv add 'structlog>=24.0.0'

# 3. Or refresh the entire lock
uv lock --upgrade

# 4. Commit the updated lock
git add pyproject.toml uv.lock
git commit -m "deps: update lockfile (quarantine $(date -u +%Y-%m-%d))"
```

---

## Auditing

```bash
# Show all resolved packages and their hashes
uv pip list --format json

# Verify lock integrity
uv lock --check
```
