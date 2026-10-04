# mentat-sys1-v0.1 Model Card

## Identity

- Model: `mentat-sys1-v0.1`
- Base: `Qwen/Qwen3.5-4B`
- Base revision: `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`
- Architecture: LoRA adapter with a native decision readout
- Adapter SHA-256:
  `c72687243aaa8c38f984091247619c242f3f3443a1f3d35c9c3526c4dfde2d83`
- Readout SHA-256:
  `c2c5b2130fcb6604ab7bb7048464fb7c01281a929df767525525870b67ea7d7d`
- Calibration SHA-256:
  `01ac7306e8312beb749090ad5e3887dc23606d7a5d7dca5d2aa85ff99b612b33`

This is an independently trained model. It was not initialized from or resumed
from another decision-model adapter.

## Intended Use

The model answers bounded `noul`, `choice`, and `score` decisions through a
TypeSafe-compatible `/v1/systemone` endpoint. It returns a native probability
distribution over the supplied labels. Inference uses one model pass and a
frozen scalar temperature of `1.0551417227266353`.

The model is not a general chat assistant. It is not intended to replace human
review in high-impact decisions.

## Local Evaluation

The frozen checkpoint was evaluated twice from fresh GPU server processes on
the 231 public items at JevBench revision
`2fa63fa3226cb369795525ed011800f57dcbd894`.

| Tier | Correct | Total |
| --- | ---: | ---: |
| Easy | 48 | 48 |
| Original | 66 | 72 |
| Hard | 86 | 111 |
| All | 200 | 231 |

Both runs produced the same prediction digest
`b8eec370fbae0ae0e560cfd2480ebe25a6a10dae870fb2e6e1ea70c4dc066159`.
There were zero invalid responses and zero severe failures.

The two serial A800 runs measured p50 latency of `0.090-0.091 s` and p95
latency of `0.332-0.336 s`.

This `200/231` (86.58%) result is submitter-measured local evidence, not an
official current leaderboard score. Current open and sealed evaluation remains
evaluator-controlled.

## Calibration

Scalar temperature was fitted on an independent 1,120-row reserve split with
SHA-256
`a116f4d228e25bac5df61b9e52191adfe4181a0852f6050d9040cee7f35b0a02`.
The fit preserved every predicted label. Reserve metrics changed as follows:

| Metric | Before | After |
| --- | ---: | ---: |
| NLL | 0.372088 | 0.371613 |
| ECE | 0.020559 | 0.023916 |
| Brier | 0.210315 | 0.210195 |

Temperature fitting minimized NLL. ECE is reported as an observed diagnostic
and was not the optimization objective.

## Contamination Audit

The registered 87,386-row training projection was checked against all 231
public items. Exact IDs, derivative IDs, source groups, image hashes, and
unresolved word 8-grams were all zero.

The initial 8-gram scan found 24 distinct short overlaps across 182
occurrences. Every overlap was reviewed and classified as generic template
language, standard legal or policy wording, or another non-instance-specific
phrase. Their hashes and classifications are retained in
`evidence/contamination-report.json`; none remains on the unresolved list.

This is an exact-overlap audit of the registered projection, not proof against
all forms of semantic contamination.

## Reproducibility Scope

The publication bundle supports clean installation, checkpoint verification,
and serving from pinned model and code revisions. Full reconstruction of the
training dataset and training run is not part of this submission-first release:
the public source ledger remains incomplete and no data-reproduction claim is
made.

## Release State

- Code: `https://github.com/mentat-asi/mentat-sys1-v0.1`, tag `v0.1.0`
- Model: `https://huggingface.co/yunqu/mentat-sys1-v0.1`, tag `v0.1.0`

## License

Project code and released model-specific files use Apache-2.0. The unchanged
Qwen base model is downloaded separately and retains its upstream Apache-2.0
license. Adapted implementation sources and revisions are listed in `NOTICE`.
