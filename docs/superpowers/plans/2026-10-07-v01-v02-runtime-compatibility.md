# Mentat Sys1 v0.1/v0.2 Runtime Compatibility Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Publish a version-neutral Mentat Sys1 runtime that validates and serves both frozen v0.1 and v0.2 model packages without weakening either release contract.

**Architecture:** A small registry in the configuration layer defines the two supported model IDs. The selected configuration and matching model manifest establish each backend instance's identity; a shared calibration parser normalizes scalar-v1 and per-type-v2 payloads into one runtime structure. Release validation remains hash-bound and fail-closed, while documentation and package metadata move to the version-neutral project name.

**Tech Stack:** Python 3.12, Pydantic 2, NumPy, PyTorch/Transformers/PEFT, FastAPI, pytest, Ruff, mypy, uv, Hugging Face Hub, GitHub CLI.

---

### Task 1: Register Both Frozen Model Configurations

**Files:**
- Modify: `src/mentat_sys1/__init__.py`
- Modify: `src/mentat_sys1/contracts.py`
- Create: `configs/mentat-sys1-v0.2.json`
- Modify: `tests/unit/test_contracts.py`
- Modify: `tests/unit/test_identity.py`

- [ ] **Step 1: Write failing configuration tests**

Add tests that load both checked-in configurations and reject any unregistered
model ID:

```python
@pytest.mark.parametrize(
    ("filename", "model_id", "correct"),
    [
        ("mentat-sys1-v0.1.json", "mentat-sys1-v0.1", 200),
        ("mentat-sys1-v0.2.json", "mentat-sys1-v0.2", 206),
    ],
)
def test_registered_release_configs_load(
    filename: str,
    model_id: str,
    correct: int,
) -> None:
    config = load_config(ROOT / "configs" / filename)
    assert config.project_id == model_id
    assert config.release.legacy_public_correct == correct
    assert config.release.legacy_public_total == 231
    assert config.release.legacy_public_easy_correct == 48


def test_project_config_rejects_unregistered_model_id() -> None:
    payload = _raw_config()
    payload["project_id"] = "mentat-sys1-v9"
    with pytest.raises(ValidationError):
        ProjectConfig.model_validate(payload)
```

Update the identity test to expect package version `0.2.0`,
`SUPPORTED_MODEL_IDS == {"mentat-sys1-v0.1", "mentat-sys1-v0.2"}`, and retain
`MODEL_ID == "mentat-sys1-v0.1"` as the documented legacy default.

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```bash
uv run pytest tests/unit/test_contracts.py tests/unit/test_identity.py -q
```

Expected: failures because v0.2 is not an accepted project ID, the v0.2 config
does not exist, and package metadata still reports `0.1.1`.

- [ ] **Step 3: Implement the two-ID registry and generic score fields**

Define:

```python
ModelId = Literal["mentat-sys1-v0.1", "mentat-sys1-v0.2"]
SUPPORTED_MODEL_IDS: frozenset[str] = frozenset(
    {"mentat-sys1-v0.1", "mentat-sys1-v0.2"}
)
MODEL_ID = "mentat-sys1-v0.1"
__version__ = "0.2.0"
```

Use `ModelId` for `ProjectConfig.project_id`. Change the three exact public
score fields from v0.1-only literals to constrained non-negative integers,
while preserving `231`, zero-invalid, and zero-severe-failure literals.

Create `configs/mentat-sys1-v0.2.json` with:

```json
{
  "schema_version": 1,
  "project_id": "mentat-sys1-v0.2",
  "base_model": {
    "repository": "Qwen/Qwen3.5-4B",
    "revision": "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
  },
  "paths": {"data": "data", "model": "model", "evidence": "evidence"},
  "runtime": {"readout_codes": 256, "max_length": 16384},
  "release": {
    "adapter_sha256": "cd27395b11dccbe4ed69c9d09ca7ceb64b24afafb2fb98db78510077d04ef228",
    "readout_sha256": "ae953826fea781c6b73741b14170bc550b0e3f62b7a127f7eb9af3b700e70eb9",
    "legacy_public_correct": 206,
    "legacy_public_total": 231,
    "legacy_public_easy_correct": 48,
    "maximum_invalid": 0,
    "maximum_severe_failures": 0
  }
}
```

- [ ] **Step 4: Run the focused tests and verify GREEN**

Run:

```bash
uv run pytest tests/unit/test_contracts.py tests/unit/test_identity.py -q
```

Expected: all selected tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/mentat_sys1/__init__.py src/mentat_sys1/contracts.py \
  configs/mentat-sys1-v0.2.json tests/unit/test_contracts.py \
  tests/unit/test_identity.py
git commit -m "feat: register mentat sys1 v0.2"
```

### Task 2: Parse Scalar And Per-Type Calibration

**Files:**
- Modify: `src/mentat_sys1/inference/calibration.py`
- Create: `tests/unit/test_calibration_compatibility.py`

- [ ] **Step 1: Write failing calibration parser tests**

Create tests for scalar normalization, v0.2 parsing, and fail-closed validation:

```python
def test_scalar_calibration_expands_to_all_runtime_types() -> None:
    loaded = load_temperature_calibration(
        {
            "schema_version": 1,
            "method": "scalar_temperature",
            "temperature": 1.0551417227266353,
        },
        digest="a" * 64,
    )
    assert loaded.pooled_temperature == pytest.approx(1.0551417227266353)
    assert loaded.temperature_by_type == {
        "choice": pytest.approx(1.0551417227266353),
        "noul": pytest.approx(1.0551417227266353),
        "score": pytest.approx(1.0551417227266353),
    }
    assert loaded.version == "scalar_temperature-v1:aaaaaaaaaaaa"


def test_v02_calibration_loads_exact_type_temperatures() -> None:
    loaded = load_temperature_calibration(
        {
            "schema_version": 2,
            "method": "temperature_by_type",
            "pooled_temperature": 1.5758800892767102,
            "temperature_by_type": {
                "choice": 1.609674650339169,
                "noul": 1.3462360767794004,
                "score": 1.6476480715733943,
            },
        },
        digest="b" * 64,
    )
    assert loaded.temperature_by_type["noul"] == pytest.approx(
        1.3462360767794004
    )
    assert loaded.version == "temperature_by_type-v2:bbbbbbbbbbbb"


@pytest.mark.parametrize(
    "temperatures",
    [
        {"choice": 1.0, "noul": 1.0},
        {"choice": 1.0, "noul": 1.0, "score": 0.0},
        {"choice": 1.0, "noul": 1.0, "score": float("inf")},
        {"choice": 1.0, "noul": 1.0, "score": 1.0, "extra": 1.0},
    ],
)
def test_v02_calibration_rejects_invalid_type_map(
    temperatures: dict[str, float],
) -> None:
    with pytest.raises(ValueError):
        load_temperature_calibration(
            {
                "schema_version": 2,
                "method": "temperature_by_type",
                "pooled_temperature": 1.0,
                "temperature_by_type": temperatures,
            },
            digest="c" * 64,
        )
```

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```bash
uv run pytest tests/unit/test_calibration_compatibility.py -q
```

Expected: import failure because `load_temperature_calibration` does not exist.

- [ ] **Step 3: Implement the normalized calibration contract**

Add:

```python
CalibrationType: TypeAlias = Literal["choice", "noul", "score"]
CALIBRATION_TYPES: tuple[CalibrationType, ...] = ("choice", "noul", "score")


@dataclass(frozen=True)
class CalibrationParameters:
    pooled_temperature: float
    temperature_by_type: dict[CalibrationType, float]
    version: str
```

Implement `_validated_temperature()` and
`load_temperature_calibration(payload, digest=...)`. Schema 1 expands the
single value to all three types. Schema 2 requires exactly the three registered
keys. Both paths reject booleans, non-numeric values, non-finite values, and
values less than or equal to zero.

- [ ] **Step 4: Run the focused tests and verify GREEN**

Run:

```bash
uv run pytest tests/unit/test_calibration_compatibility.py \
  tests/unit/test_calibration.py -q
```

Expected: all selected tests pass and the existing scalar fitting behavior is
unchanged.

- [ ] **Step 5: Commit**

```bash
git add src/mentat_sys1/inference/calibration.py \
  tests/unit/test_calibration_compatibility.py
git commit -m "feat: support versioned calibration schemas"
```

### Task 3: Make Runtime Identity Instance-Bound

**Files:**
- Modify: `src/mentat_sys1/inference/backend.py`
- Modify: `src/mentat_sys1/inference/api.py`
- Modify: `tests/unit/test_model_loading.py`
- Modify: `tests/integration/test_api.py`

- [ ] **Step 1: Write failing backend and API tests**

Add a v0.2 backend test with one choice, boolean, and ordinal field:

```python
def test_backend_uses_configured_identity_and_routes_typed_temperature() -> None:
    backend = PortableBackend(
        engine=TypedEngine(),
        identity={"name": "mentat-sys1-v0.2"},
        temperature_by_type={
            "choice": 0.5,
            "noul": 2.0,
            "score": 4.0,
        },
        calibration_version="temperature_by_type-v2:abc123",
    )
    results, _ = backend.score([], typed_request())
    confidences = [max(result.scores.values()) for result in results]
    assert backend.model_id == "mentat-sys1-v0.2"
    assert confidences[0] > confidences[1] > confidences[2]
```

Add assertions that a v0.2 API accepts `"model": "mentat-sys1-v0.2"`, rejects
v0.1 for that backend, and reports the version-neutral FastAPI title
`mentat-sys1`.

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```bash
uv run pytest \
  tests/unit/test_model_loading.py::test_backend_uses_configured_identity_and_routes_typed_temperature \
  tests/integration/test_api.py -q
```

Expected: v0.2 construction fails because `PortableBackend.model_id` is still
the global v0.1 constant, and type-specific temperatures are unsupported.

- [ ] **Step 3: Implement instance identity and temperature routing**

Remove runtime reliance on the legacy `MODEL_ID` constant. Validate model IDs
against `SUPPORTED_MODEL_IDS`, set `self.model_id` from `identity["name"]`, and
accept `temperature_by_type: Mapping[str, float] | None`.

Normalize scalar temperatures to all three runtime types, then route:

```python
runtime_types: dict[str, CalibrationType] = {
    "choice": "choice",
    "boolean": "noul",
    "ordinal": "score",
}
temperature = self.temperature_by_type[runtime_types[field.type]]
```

In `PortableBackend.load`, pass `config.project_id` through the identity and
forward the verified type map. In config-less directory verification, accept
only manifest IDs in `SUPPORTED_MODEL_IDS`. Set the FastAPI title to
`mentat-sys1`.

- [ ] **Step 4: Run focused and adjacent tests and verify GREEN**

Run:

```bash
uv run pytest tests/unit/test_model_loading.py tests/integration/test_api.py \
  tests/unit/test_inference_contracts.py tests/unit/test_scoring.py -q
```

Expected: all selected tests pass for both IDs and all decision types.

- [ ] **Step 5: Commit**

```bash
git add src/mentat_sys1/inference/backend.py src/mentat_sys1/inference/api.py \
  tests/unit/test_model_loading.py tests/integration/test_api.py
git commit -m "feat: bind runtime identity to model config"
```

### Task 4: Verify Both Release Schemas And Score Gates

**Files:**
- Modify: `src/mentat_sys1/release/verify.py`
- Modify: `src/mentat_sys1/release/package.py`
- Modify: `tests/release/test_verify.py`
- Modify: `tests/release/test_package.py`
- Modify: `tests/release/test_legacy_public_gate.py`

- [ ] **Step 1: Write failing v0.2 release tests**

Add a schema-2 calibration fixture matching the released structure:

```python
payload = {
    "schema_version": 2,
    "method": "temperature_by_type",
    "pooled_temperature": 1.5758800892767102,
    "temperature_by_type": {
        "choice": 1.609674650339169,
        "noul": 1.3462360767794004,
        "score": 1.6476480715733943,
    },
    "split": {"rows": 1250, "sha256": "a" * 64},
    "model": {
        "adapter_sha256": config.release.adapter_sha256,
        "readout_sha256": config.release.readout_sha256,
    },
    "before": {"nll": 1.0416192707769858},
    "after": {"nll": 0.9668147145225859},
    "type_fits": {
        "choice": {"rows": 375, "source": "shrinkage",
                   "temperature": 1.609674650339169},
        "noul": {"rows": 375, "source": "shrinkage",
                 "temperature": 1.3462360767794004},
        "score": {"rows": 500, "source": "shrinkage",
                  "temperature": 1.6476480715733943},
    },
    "prediction_digest": {"before": "b" * 64, "after": "b" * 64},
    "predictions_unchanged": True,
}
```

Assert that `validate_calibration()` returns the pooled and typed values,
schema-2 manifest metadata must match exactly, and the v0.2 public gate accepts
48/48 Easy, 71/72 Original, and 87/111 Hard for 206/231. Add a cross-version
test that verifies a v0.2 manifest with the v0.1 config raises
`model manifest identity differs`.

- [ ] **Step 2: Run release tests and verify RED**

Run:

```bash
uv run pytest tests/release/test_verify.py tests/release/test_package.py \
  tests/release/test_legacy_public_gate.py -q
```

Expected: schema 2 is rejected and the v0.2 score gate cannot be represented.

- [ ] **Step 3: Generalize validation without weakening v0.1**

Use `load_temperature_calibration()` in `validate_calibration()`. Keep the
exact v0.1 split identity for schema 1. For schema 2 require a positive row
count, a lowercase SHA-256 split digest, exact model hashes, unchanged
predictions, exact `choice/noul/score` fit coverage, row counts summing to the
split size, source in `{"type", "pooled", "shrinkage"}`, and fit temperatures
equal to the normalized map.

When verifying a calibrated model, require manifest calibration metadata:

```python
if calibration_payload["schema_version"] == 1:
    expected = {
        "schema_version": 1,
        "method": "scalar_temperature",
        "temperature": calibration["temperature"],
        "calibration_sha256": calibration["calibration_sha256"],
        "calibration_version": calibration["calibration_version"],
    }
else:
    expected = {
        "schema_version": 2,
        "method": "temperature_by_type",
        "pooled_temperature": calibration["temperature"],
        "temperature_by_type": calibration["temperature_by_type"],
        "calibration_sha256": calibration["calibration_sha256"],
        "calibration_version": calibration["calibration_version"],
    }
```

Update calibrated-model assembly to emit the same schema-specific manifest
record and receipt fields.

- [ ] **Step 4: Run all release tests and verify GREEN**

Run:

```bash
uv run pytest tests/release tests/integration/test_release_command.py \
  tests/integration/test_calibration_command.py -q
```

Expected: all existing v0.1 release tests and all new v0.2 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/mentat_sys1/release/verify.py \
  src/mentat_sys1/release/package.py tests/release
git commit -m "feat: verify v0.1 and v0.2 release artifacts"
```

### Task 5: Publish Version-Neutral Metadata And Documentation

**Files:**
- Modify: `pyproject.toml`
- Modify: `uv.lock`
- Modify: `README.md`
- Modify: `docs/MODEL_CARD.md`
- Modify: `docs/REPRODUCIBILITY.md`
- Modify: `docs/JEVBENCH_SUBMISSION.md`
- Create: `docs/releases/v0.1.md`
- Create: `docs/releases/v0.2.md`
- Modify: `tests/unit/test_identity.py`

- [ ] **Step 1: Write failing package and documentation assertions**

Require:

```python
assert metadata["project"]["name"] == "mentat-sys1"
assert metadata["project"]["version"] == "0.2.0"
assert "github.com/mentat-asi/mentat-sys1" in readme
assert "huggingface.co/yunqu/mentat-sys1-v0.1" in readme
assert "huggingface.co/yunqu/mentat-sys1-v0.2" in readme
assert "206/231" in readme
assert "temperature_by_type-v2" in reproducibility
```

Retain checks that public files contain no training recipe, private filesystem
path, credential, or unpublished model identity.

- [ ] **Step 2: Run identity tests and verify RED**

Run:

```bash
uv run pytest tests/unit/test_identity.py -q
```

Expected: package and documentation still identify the repository as v0.1.

- [ ] **Step 3: Update metadata and release documentation**

Set project name/version to `mentat-sys1`/`0.2.0`, then run:

```bash
uv lock
```

Make README lead with v0.2's 206/231 result, provide pinned v0.2 commands, and
include a compact compatibility table:

```markdown
| Release | Model package | Calibration | Public-231 |
| --- | --- | --- | ---: |
| v0.2 | `yunqu/mentat-sys1-v0.2@v0.2.0` | per-type v2 | 206/231 |
| v0.1 | `yunqu/mentat-sys1-v0.1@v0.1.0` | scalar v1 | 200/231 |
```

Use `configs/mentat-sys1-v0.2.json` in the primary serve example and retain a
complete v0.1 command in the compatibility section. Update current docs to
v0.2 facts and preserve the frozen historical links and scores in
`docs/releases/v0.1.md`; record the v0.2 model revision, hashes, calibration
method, and local Public-231 scope in `docs/releases/v0.2.md`.

- [ ] **Step 4: Run documentation and privacy tests and verify GREEN**

Run:

```bash
uv run pytest tests/unit/test_identity.py -q
uv run python -c \
  'from pathlib import Path; from mentat_sys1.release.verify import scan_public_tree; print(scan_public_tree(Path("."))["files_scanned"])'
```

Expected: tests pass and the public scan completes without forbidden content.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml uv.lock README.md docs tests/unit/test_identity.py
git commit -m "docs: publish version-neutral mentat sys1 runtime"
```

### Task 6: Verify Immutable v0.1 And v0.2 Packages

**Files:**
- No source changes expected
- Temporary downloads: `.artifacts/compat-v0.1`, `.artifacts/compat-v0.2`

- [ ] **Step 1: Download only release model files at immutable revisions**

Run:

```bash
hf download yunqu/mentat-sys1-v0.1 \
  --revision bc45d5d334879c4446188827263a22d57c15d1e4 \
  --include 'model/*' --local-dir .artifacts/compat-v0.1
hf download yunqu/mentat-sys1-v0.2 \
  --revision 9272fc4cafa49856118bf7c67af2803aafdd2872 \
  --include 'model/*' --local-dir .artifacts/compat-v0.2
```

Expected: both directories contain seven model files and no base-model weights.

- [ ] **Step 2: Verify each model against its registered configuration**

Run:

```bash
uv run python - <<'PY'
from pathlib import Path
from mentat_sys1.contracts import load_config
from mentat_sys1.inference.backend import verify_model_directory

for version in ("v0.1", "v0.2"):
    result = verify_model_directory(
        Path(f".artifacts/compat-{version}/model"),
        config=load_config(Path(f"configs/mentat-sys1-{version}.json")),
    )
    print(version, result["model_id"], result["calibration_version"])
PY
```

Expected:

```text
v0.1 mentat-sys1-v0.1 scalar_temperature-v1:01ac7306e831
v0.2 mentat-sys1-v0.2 temperature_by_type-v2:5023c1bb9c2a
```

- [ ] **Step 3: Verify cross-version rejection**

Run:

```bash
uv run python - <<'PY'
from pathlib import Path
from mentat_sys1.contracts import load_config
from mentat_sys1.inference.backend import verify_model_directory

try:
    verify_model_directory(
        Path(".artifacts/compat-v0.2/model"),
        config=load_config(Path("configs/mentat-sys1-v0.1.json")),
    )
except ValueError as error:
    assert str(error) == "model manifest identity differs"
else:
    raise AssertionError("v0.2 package unexpectedly passed the v0.1 contract")
PY
```

Expected: command exits zero after observing the required rejection.

- [ ] **Step 4: Run the complete local quality gate**

Run:

```bash
uv sync --frozen --all-extras --dev
uv run pytest -q
uv run ruff check .
uv run mypy
uv build
git diff --check
git status --short
```

Expected: all tests pass, Ruff and mypy report no issues, wheel and sdist build,
no whitespace errors, and only intended committed changes remain.

### Task 7: Rename And Publish GitHub Release v0.2.0

**Files:**
- Remote repository metadata and Git refs only

- [ ] **Step 1: Confirm authentication and remote preconditions**

Run:

```bash
gh auth status
git fetch origin --tags
test "$(git rev-parse origin/main)" = "dd9a0000979aef6829cd64e62731470afbf1b01d"
test -z "$(git status --porcelain)"
```

Expected: authenticated owner access, unchanged upstream baseline, and clean
local tree. If authentication is missing, complete GitHub's device flow before
continuing.

- [ ] **Step 2: Rename the repository**

Run:

```bash
gh api --method PATCH repos/mentat-asi/mentat-sys1-v0.1 \
  -f name=mentat-sys1 --jq '.full_name'
git remote set-url origin https://github.com/mentat-asi/mentat-sys1.git
```

Expected: `mentat-asi/mentat-sys1`.

- [ ] **Step 3: Push reviewed commits and immutable tag**

Run:

```bash
git push origin main
git tag -a v0.2.0 -m "Mentat Sys1 v0.2.0"
git push origin v0.2.0
gh release create v0.2.0 \
  --repo mentat-asi/mentat-sys1 \
  --title "Mentat Sys1 v0.2.0" \
  --notes "Version-neutral runtime with frozen v0.1 and v0.2 model compatibility."
```

Expected: main, tag, and GitHub Release all resolve to the same tested commit.

- [ ] **Step 4: Verify new URL, redirect, tag, and fresh clone**

Run:

```bash
git ls-remote https://github.com/mentat-asi/mentat-sys1.git HEAD refs/tags/v0.2.0
curl -sSIL https://github.com/mentat-asi/mentat-sys1-v0.1 | head
rm -rf .artifacts/fresh-clone
git clone --branch v0.2.0 --depth 1 \
  https://github.com/mentat-asi/mentat-sys1.git .artifacts/fresh-clone
cd .artifacts/fresh-clone
uv sync --frozen --dev
uv run pytest -q
```

Expected: the new repository resolves, the old URL redirects, the immutable
tag exists, and the fresh clone's complete test suite passes.
