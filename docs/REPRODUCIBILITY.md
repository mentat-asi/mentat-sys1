# Reproducibility

## Scope

This repository reproduces package installation, model integrity, and serving
for both frozen Mentat Sys1 releases. It does not claim
end-to-end reconstruction of the training dataset or training run. The data
source ledger remains fail-closed until its source and license inventory is
complete.

## Pinned Environment

- Python: `3.12`
- Dependency manager: `uv`
- Base model: `Qwen/Qwen3.5-4B`
- Base revision: `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`
- JevBench audit/evaluation revision:
  `2fa63fa3226cb369795525ed011800f57dcbd894`
- Verified inference device: one NVIDIA A800 80GB

## Current v0.2 Release

```bash
git clone https://github.com/mentat-asi/mentat-sys1.git
cd mentat-sys1
git checkout v0.2.0
uv sync --frozen --extra serve

hf download yunqu/mentat-sys1-v0.2 \
  --revision v0.2.0 \
  --local-dir MODEL_PACKAGE_DIR
hf download Qwen/Qwen3.5-4B \
  --revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a \
  --local-dir QWEN_BASE_DIR

uv run mentat-sys1 serve \
  --config configs/mentat-sys1-v0.2.json \
  --artifact-root RUNTIME_DIR \
  --base-model-path QWEN_BASE_DIR \
  --model-dir MODEL_PACKAGE_DIR/model \
  --device cuda \
  --host LOOPBACK_HOST \
  --port 8765
```

The v0.2 model is bound to
`temperature_by_type-v2:5023c1bb9c2a`; the runtime applies the registered
`choice`, `noul`, or `score` temperature for each field. The manifest is bound
to adapter SHA-256
`cd27395b11dccbe4ed69c9d09ca7ceb64b24afafb2fb98db78510077d04ef228`
and readout SHA-256
`ae953826fea781c6b73741b14170bc550b0e3f62b7a127f7eb9af3b700e70eb9`.

## Frozen v0.1 Release

```bash
git clone https://github.com/mentat-asi/mentat-sys1-v0.1.git
cd mentat-sys1-v0.1
git checkout v0.1.1
uv sync --frozen --extra serve
uv run pytest -q
uv run ruff check .
uv run mypy
```

Serving dependencies remain isolated in the `serve` extra.

## v0.1 Model Integrity

Download the publication bundle at its immutable revision, then verify both
checksum layers:

```bash
hf download yunqu/mentat-sys1-v0.1 \
  --revision v0.1.0 \
  --local-dir MODEL_PACKAGE_DIR
cd MODEL_PACKAGE_DIR
sha256sum --check SHA256SUMS
cd model
sha256sum --check SHA256SUMS
```

The publication release command also checks the model manifest, calibration
binding, four aggregate evidence receipts, exact output allowlist, and public
text scan before atomically creating the bundle.

## v0.1 Serve

Download the pinned Qwen base separately and launch the model:

```bash
uv run mentat-sys1 serve \
  --config configs/mentat-sys1-v0.1.json \
  --artifact-root RUNTIME_DIR \
  --base-model-path QWEN_BASE_DIR \
  --model-dir MODEL_PACKAGE_DIR/model \
  --device cuda \
  --host LOOPBACK_HOST \
  --port 8765
```

Readiness checks:

```bash
export ENDPOINT_URL="http://LOOPBACK_HOST:8765"
curl --fail "$ENDPOINT_URL/health"
curl --fail "$ENDPOINT_URL/v1/models"
```

`POST /v1/systemone` accepts `noul`, `choice`, and `score` fields. The server
returns native probabilities, reports processed input tokens, emits zero
generated output tokens, and uses one language-model pass per decision.

## v0.1 Publication Bundle

From a verified final artifact:

```bash
uv run mentat-sys1 release \
  --config configs/mentat-sys1-v0.1.json \
  --artifact-root PUBLICATION_DIR \
  --publication-source-root FINAL_ARTIFACT_DIR \
  --project-root .
```

The output contains only the calibrated model, aggregate calibration,
contamination and public-evaluation evidence, and release documentation. It
does not copy `evaluation/`, per-item results, raw responses, local paths, or
credentials.
