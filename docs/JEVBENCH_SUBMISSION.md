# JevBench Bench Request Draft

This request is bound to the `v0.1.0` code and model release tags.

## Issue Title

`[bench request]: Add mentat-sys1-v0.1 (Qwen3.5-4B + LoRA/readout, TypeSafe wire format, self-hosted)`

## Issue Body

Please evaluate **mentat-sys1-v0.1**, an independently trained open-weights
decision model based on Qwen3.5-4B. It answers `noul`, `choice`, and `score`
requests with native option probabilities through a TypeSafe-compatible
`/v1/systemone` endpoint.

This is a new model and release package, not an update to the model family in
issue #80. Please track it as a separate bench request.

### Pinned artifacts

- Model package: `https://huggingface.co/yunqu/mentat-sys1-v0.1`
- Model revision: `v0.1.0`
- Serving code: `https://github.com/mentat-asi/mentat-sys1-v0.1`
- Code revision: `v0.1.0`
- Base model: `Qwen/Qwen3.5-4B`
- Base revision: `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`
- Adapter SHA-256:
  `c72687243aaa8c38f984091247619c242f3f3443a1f3d35c9c3526c4dfde2d83`
- Readout SHA-256:
  `c2c5b2130fcb6604ab7bb7048464fb7c01281a929df767525525870b67ea7d7d`
- Calibration SHA-256:
  `01ac7306e8312beb749090ad5e3887dc23606d7a5d7dca5d2aa85ff99b612b33`
- License: Apache-2.0 for the project and model-specific files; the unchanged
  Qwen base retains its upstream Apache-2.0 license

The model package contains the LoRA adapter, decision readout, frozen scalar
calibration, manifests, checksums, and aggregate audit evidence. The Qwen base
weights are downloaded separately.

### Run it

```bash
git clone https://github.com/mentat-asi/mentat-sys1-v0.1.git
cd mentat-sys1-v0.1
git checkout v0.1.0
uv sync --frozen --extra serve

hf download yunqu/mentat-sys1-v0.1 \
  --revision v0.1.0 \
  --local-dir MODEL_PACKAGE_DIR
hf download Qwen/Qwen3.5-4B \
  --revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a \
  --local-dir QWEN_BASE_DIR

uv run mentat-sys1 serve \
  --config configs/mentat-sys1-v0.1.json \
  --artifact-root RUNTIME_DIR \
  --base-model-path QWEN_BASE_DIR \
  --model-dir MODEL_PACKAGE_DIR/model \
  --device cuda \
  --host LOOPBACK_HOST \
  --port 8765
```

Wait until both `GET /health` and `GET /v1/models` return 200, then use the
unchanged JevBench `typesafe` adapter against
`ENDPOINT_URL/v1/systemone`. Inference uses one model pass, a 16,384-token
input limit, and no generated answer tokens.

Verified hardware: one NVIDIA A800 80GB. Requested evaluator hardware: one H100
80GB or an equivalent CUDA GPU.

### Submitter-measured public results

Two fresh GPU server processes ran all 231 public items at JevBench revision
`2fa63fa3226cb369795525ed011800f57dcbd894`.

| Tier | Correct | Total |
| --- | ---: | ---: |
| Easy | 48 | 48 |
| Original | 66 | 72 |
| Hard | 86 | 111 |
| All | **200** | **231** |

Both runs had zero invalid responses and zero severe failures, and produced the
same prediction digest:
`b8eec370fbae0ae0e560cfd2480ebe25a6a10dae870fb2e6e1ea70c4dc066159`.
On one A800 80GB, the two serial public runs measured p50 latency
`0.090-0.091 s`, p95 latency `0.332-0.336 s`, and mean processed input length
`691.25` tokens per decision. The public-set ECE was `0.05835`.

These are local public-set measurements, not official leaderboard claims. We
request an evaluator-controlled offline run for the current open and sealed
suites. No hosted endpoint is offered.

### Integrity and training disclosure

- The registered audit covers 87,386 model-visible training rows and all 231
  public items.
- Exact IDs, derivative IDs, source groups, image hashes, and unresolved word
  8-grams are all zero.
- The initial scan found 24 distinct short 8-gram overlaps across 182
  occurrences. Each was reviewed as generic template language, standard legal
  or policy wording, or another non-instance-specific phrase. Hashes and review
  classes are in `evidence/contamination-report.json`.
- Calibration uses an independent 1,120-row reserve split. The frozen
  temperature is `1.0551417227266353`, and fitting did not change any predicted
  label.
- Serving makes no network calls and contains no item-specific IDs, answers,
  or response lookup.
- Complete public reconstruction of the training data is still pending its
  source and license ledger. This request claims reproducible model packaging
  and serving, not full training reproduction.

### Listing metadata

| Field | Value |
| --- | --- |
| Name | mentat-sys1-v0.1 |
| Class | Open-weights decision model, TypeSafe wire format |
| Base | Qwen/Qwen3.5-4B |
| Weights | LoRA adapter plus native decision readout |
| Open weights | Yes |
| License | Apache-2.0 |
| Cost basis | Evaluator methodology for self-hosted Qwen3.5-4B, using measured input tokens and zero generated output tokens |
