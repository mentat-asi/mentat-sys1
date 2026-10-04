# mentat-sys1-v0.1

`mentat-sys1-v0.1` is an independently trained decision model based on
`Qwen/Qwen3.5-4B` at revision
`851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.

It is not a revision, continuation, or release of another decision model.
Selected implementation techniques and data-conversion code are adapted from
the Apache-2.0 sources identified in `NOTICE`.

The frozen checkpoint scored `200/231` (86.58%) on the pinned 231-item
legacy-public JevBench set: Easy `48/48`, Original `66/72`, and Hard `86/111`.
Two fresh server processes produced the same prediction digest, with zero
invalid responses and zero severe failures. This is local evidence, not an
official current leaderboard score. Evaluation on the current open and sealed
suites remains evaluator-controlled.

The release package contains a LoRA adapter, decision readout, scalar
calibration, a TypeSafe-compatible `/v1/systemone` server, and hash-bound audit
evidence. The Qwen base weights are downloaded separately.

## Commands

```bash
mentat-sys1 calibrate
mentat-sys1 audit
mentat-sys1 release

mentat-sys1 serve \
  --config configs/mentat-sys1-v0.1.json \
  --artifact-root RUNTIME_DIR \
  --base-model-path QWEN_BASE_DIR \
  --model-dir MODEL_PACKAGE_DIR/model \
  --device cuda
```

Every command accepts an explicit configuration path and artifact root. The
repository does not contain model weights or the materialized training data.

Build the allowlisted publication tree from a verified final artifact:

```bash
mentat-sys1 release \
  --config configs/mentat-sys1-v0.1.json \
  --artifact-root PUBLICATION_DIR \
  --publication-source-root FINAL_ARTIFACT_DIR \
  --project-root .
```

The publication gate copies only the calibrated model, four aggregate evidence
files, and release documentation. Raw responses and evaluation directories are
excluded. Complete training-data reconstruction remains a separate, unfinished
release track and is not claimed by the model package.
