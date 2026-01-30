# Publishing Hush Engine

This document contains instructions for publishing the hush-engine package to GitHub and PyPI.

## Current Status

✅ **Repository prepared locally** at `/Users/valentine/Documents/GitHub/hush-engine/`

### Files Included:
- `hush_engine/` - Complete Python package
- `pyproject.toml` - PyPI packaging configuration
- `README.md` - Comprehensive API documentation
- `CONTRIBUTING.md` - Contributor guidelines
- `LICENSE` - MIT License
- `tests/` - Unit tests
- `.gitignore` - Python-specific ignores
- Initial commit created (81e4533)

## Step 1: Create GitHub Repository

1. Go to https://github.com/new
2. Settings:
   - **Owner**: NewMediaStudio (or your organization)
   - **Repository name**: `hush-engine`
   - **Description**: "Local-first PII detection engine using Presidio and Apple Vision OCR"
   - **Visibility**: ✅ Public
   - **Initialize**: ❌ Do NOT add README, .gitignore, or license (we have them)
3. Click "Create repository"
4. Copy the repository URL (e.g., `https://github.com/NewMediaStudio/hush-engine.git`)

## Step 2: Push to GitHub

```bash
cd /Users/valentine/Documents/GitHub/hush-engine

# Add GitHub remote
git remote add origin https://github.com/NewMediaStudio/hush-engine.git

# Push to GitHub
git push -u origin main
```

## Step 3: Configure GitHub Repository

After pushing, configure these settings on GitHub:

### Repository Settings → General
- ✅ Enable "Issues"
- ✅ Enable "Discussions" (for Q&A)
- ❌ Disable "Projects" (unless you want to use them)
- ✅ Enable "Preserve this repository" (optional archival)

### Repository Settings → About
- Description: "Local-first PII detection engine using Presidio and Apple Vision OCR"
- Website: (leave blank or add docs site later)
- Topics: `pii`, `privacy`, `detection`, `ocr`, `presidio`, `security`, `macos`

### Repository Settings → Pages (optional)
- Set up GitHub Pages for documentation if desired

## Step 4: Publish to PyPI

### Prerequisites
1. **Create PyPI account**: https://pypi.org/account/register/
2. **Verify email**
3. **Enable 2FA** (recommended)
4. **Create API token**:
   - Go to Account Settings → API tokens
   - Click "Add API token"
   - Token name: "hush-engine-upload"
   - Scope: "Entire account" (for first upload, then change to project-specific)
   - **Save the token** - you won't see it again!

### Build and Upload

```bash
cd /Users/valentine/Documents/GitHub/hush-engine

# Install build tools (if not already installed)
pip install build twine

# Build distribution packages
python -m build
# Creates: dist/hush_engine-1.0.0.tar.gz and dist/hush_engine-1.0.0-py3-none-any.whl

# Upload to PyPI
python -m twine upload dist/*

# When prompted:
# Username: __token__
# Password: <paste your API token here>
```

### Verify Upload

1. Visit https://pypi.org/project/hush-engine/
2. Check that version 1.0.0 is visible
3. Verify README renders correctly
4. Test installation: `pip install hush-engine`

## Step 5: Update Main Wrapper Repository

After publishing to PyPI, update the wrapper repository to use the published package:

```bash
cd /Users/valentine/Documents/GitHub/hush

# Remove local hush_engine/ directory
git rm -rf hush_engine/
git commit -m "Remove hush_engine/ - now using PyPI package"

# Reinstall from PyPI
source venv/bin/activate
pip uninstall hush-engine  # Remove local build
pip install hush-engine    # Install from PyPI

# Test
./run.sh
# Upload a test image and verify detection works
```

## Step 6: Add GitHub Repository Badges (Optional)

Add badges to `README.md` for a professional look:

```markdown
# Hush Engine

[![PyPI version](https://badge.fury.io/py/hush-engine.svg)](https://badge.fury.io/py/hush-engine)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

Local-first PII detection engine using Presidio and Apple Vision OCR.
```

## Troubleshooting

### PyPI Upload Fails
- **Invalid token**: Regenerate API token on PyPI
- **Package already exists**: Bump version in `pyproject.toml` and rebuild
- **Invalid README**: Ensure `README.md` has valid markdown

### GitHub Push Fails
- **Authentication**: Use personal access token or SSH key
- **Remote exists**: Check if you already added remote with `git remote -v`

### Wrapper Can't Find Engine
- **PYTHONPATH issue**: Check that `hush-engine` is installed in venv
- **Import errors**: Verify all dependencies installed: `pip list | grep hush`

## Post-Publication Checklist

- [ ] GitHub repository created and pushed
- [ ] PyPI package published and installable
- [ ] Wrapper repo updated to use PyPI package
- [ ] Wrapper app tested and working
- [ ] README badges added (optional)
- [ ] GitHub repository description and topics set
- [ ] GitHub Discussions or Issues enabled

## Next Steps

After publishing:

1. **Announce**: Tweet, blog post, or share on relevant communities
2. **Documentation**: Consider setting up ReadTheDocs or GitHub Pages
3. **CI/CD**: Add GitHub Actions for automated testing
4. **Dependabot**: Enable for automatic dependency updates
5. **PyPI project settings**: Change token scope to project-specific

---

**Repository**: https://github.com/NewMediaStudio/hush-engine  
**Package**: https://pypi.org/project/hush-engine/  
**License**: MIT
