# Publishing Hush Engine

Instructions for publishing the `hush-engine` package to GitHub and PyPI.

## Step 1: GitHub Repository

1. Create a new public repository at https://github.com/new
   - Name: `hush-engine`
   - Description: `Local-first PII detection engine using Presidio and Apple Vision OCR`
   - Do **not** initialize with README, license, or `.gitignore` — the repo already has them.
2. Push:

```bash
git remote add origin https://github.com/<owner>/hush-engine.git
git push -u origin main
```

### Repository settings

- Topics: `pii`, `privacy`, `detection`, `ocr`, `presidio`, `security`, `macos`
- Enable Issues and Discussions.
- Disable wiki unless you plan to use it.

## Step 2: PyPI

### Prerequisites

1. Create a PyPI account: https://pypi.org/account/register/
2. Verify email and enable 2FA.
3. Create a project-scoped API token (after first upload).

### Build and upload

```bash
# Install build tools
pip install build twine

# Build
python -m build
# Creates dist/hush_engine-<version>.tar.gz and .whl

# Upload
python -m twine upload dist/*
# Username: __token__
# Password: <paste API token>
```

### Verify

1. Visit `https://pypi.org/project/hush-engine/`.
2. Check version, README rendering, and installation:
   ```bash
   pip install hush-engine
   ```

## Step 3: Post-publication

- Add a GitHub release matching the version tag (`v1.9.0`, etc.).
- Announce on relevant communities if desired.
- Switch the PyPI token scope from account-wide to project-specific.
- Enable Dependabot for dependency updates.

## Troubleshooting

- **PyPI upload fails — package already exists**: Bump version in `pyproject.toml` and rebuild.
- **Invalid README on PyPI**: Ensure `README.md` is valid CommonMark; twine validates on upload.
- **GitHub auth fails**: Use a personal access token (`git remote set-url origin https://<token>@github.com/...`) or SSH.

---

**Repository**: https://github.com/NewMediaStudio/hush-engine
**Package**: https://pypi.org/project/hush-engine/
**License**: MIT
