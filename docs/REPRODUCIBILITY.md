# Reproducibility

## Scope

This submission-first release reproduces package installation, model integrity,
serving, and the registered local evaluation gate. It does not yet claim
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

## Clean Install

```bash
git clone https://github.com/mentat-asi/mentat-sys1-v0.1.git
cd mentat-sys1-v0.1
git checkout v0.1.0
uv sync --frozen --extra serve
uv run pytest -q
uv run ruff check .
uv run mypy
```

Serving dependencies remain isolated in the `serve` extra.

## Model Integrity

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

## Serve

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

## Publication Bundle

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
