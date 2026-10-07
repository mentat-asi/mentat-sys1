# Mentat Sys1 README Rewrite Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rewrite the GitHub README around Mentat Sys1 v0.2 while retaining a compact, accurate v0.1 compatibility record.

**Architecture:** Keep `README.md` as the concise product and quick-start surface. Link detailed release, reproducibility, model-card, and historical benchmark material instead of duplicating it.

**Tech Stack:** GitHub-flavored Markdown, pytest content assertions, Ruff, mypy, uv

---

### Task 1: Lock The README Contract

**Files:**
- Modify: `tests/unit/test_identity.py`
- Test: `tests/unit/test_identity.py`

- [ ] **Step 1: Update the README content test**

Require the primary README to contain:

```python
required_markers = (
    "SemIF 187/231",
    "Mentat v0.1 200/231",
    "Mentat v0.2 206/231",
    "git clone https://github.com/mentat-asi/mentat-sys1.git",
    "hf download yunqu/mentat-sys1-v0.2",
    '"model": "mentat-sys1-v0.2"',
    "231/231 valid",
    "zero prediction differences",
)
```

Also reject the obsolete GitHub clone URL and the v0.1 model ID inside the
primary request/response example.

- [ ] **Step 2: Run the focused test and confirm failure**

Run:

```bash
uv run pytest tests/unit/test_identity.py::test_readme_exposes_current_release_and_reproducibility -q
```

Expected: FAIL because the current README still centers its detailed content
and API example on v0.1.

- [ ] **Step 3: Commit the failing contract**

```bash
git add tests/unit/test_identity.py
git commit -m "test: define v0.2 README contract"
```

### Task 2: Rewrite The README

**Files:**
- Modify: `README.md`
- Test: `tests/unit/test_identity.py`

- [ ] **Step 1: Replace the README narrative**

Use this section order:

```text
# mentat-sys1
one-sentence product definition
current release links
## Results
## What it does
## Run v0.2
## Call the API
## Reproducibility
## Releases
## Documentation
## License
```

Keep the main comparison compact and include only verified Public-231 values.
Use v0.2 in every primary command and API payload. Move detailed v0.1,
calibration, contamination, and benchmark-submission material behind links.

- [ ] **Step 2: Run the focused content test**

Run:

```bash
uv run pytest tests/unit/test_identity.py -q
```

Expected: PASS.

- [ ] **Step 3: Commit the README rewrite**

```bash
git add README.md
git commit -m "docs: rewrite README for Mentat Sys1 v0.2"
```

### Task 3: Validate The Documentation Release

**Files:**
- Verify: `README.md`
- Verify: `tests/unit/test_identity.py`

- [ ] **Step 1: Check stale primary references and Markdown whitespace**

Run:

```bash
rg -n "github.com/mentat-asi/mentat-sys1-v0.1|\"model\": \"mentat-sys1-v0.1\"" README.md
git diff --check HEAD~2..HEAD
```

Expected: no obsolete primary references and no whitespace errors.

- [ ] **Step 2: Run repository gates**

Run:

```bash
uv run pytest -q
uv run ruff check .
uv run mypy
uv build
```

Expected: 95 or more tests pass; Ruff and mypy report no issues; wheel and
source distribution build successfully.

- [ ] **Step 3: Confirm review state**

Run:

```bash
git status --short
git log -3 --oneline
```

Expected: clean worktree with the README rewrite committed locally for user
review. Do not move or republish the remote `v0.2.0` tag until the user accepts
the rewritten README.
