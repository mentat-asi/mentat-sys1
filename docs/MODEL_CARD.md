# Mentat Sys1 Model Card

## Current Release: v0.2

[Serving code v0.2.0](https://github.com/mentat-asi/mentat-sys1/tree/v0.2.0)
| [Frozen model package v0.2.0](https://huggingface.co/yunqu/mentat-sys1-v0.2/tree/v0.2.0)

- Model: `mentat-sys1-v0.2`
- Base: `Qwen/Qwen3.5-4B`
- Base revision: `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`
- Adapter SHA-256:
  `cd27395b11dccbe4ed69c9d09ca7ceb64b24afafb2fb98db78510077d04ef228`
- Readout SHA-256:
  `ae953826fea781c6b73741b14170bc550b0e3f62b7a127f7eb9af3b700e70eb9`
- Calibration SHA-256:
  `5023c1bb9c2a3cc0d5607096c414257d455d47238e7d85033057b4acb9df6245`
- Local JevBench Public-231: `206/231` (89.18%)

v0.2 uses frozen per-type temperatures for `choice`, `noul`, and `score`.
Its public result is directly comparable with official Public-231 accuracy,
but it is not an official composite leaderboard score.

## Frozen v0.1 Record

[Serving code v0.1.1](https://github.com/mentat-asi/mentat-sys1-v0.1/tree/v0.1.1)
| [Frozen model package v0.1.0](https://huggingface.co/yunqu/mentat-sys1-v0.1/tree/v0.1.0)
| [JevBench request #186](https://github.com/fstandhartinger/jevbench/issues/186)

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

## Download and Package Boundaries

```bash
hf download yunqu/mentat-sys1-v0.1 \
  --revision v0.1.0 \
  --local-dir MODEL_PACKAGE_DIR
hf download Qwen/Qwen3.5-4B \
  --revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a \
  --local-dir QWEN_BASE_DIR
```

The first command downloads the frozen model-specific package into a local
directory you choose: the LoRA adapter, decision readout, calibration,
manifests, checksums, and aggregate audit evidence.
It does not download the Qwen base weights. The second command downloads the
separately licensed, pinned base revision needed to serve the model.

Installation and serving commands are in the
[GitHub README](https://github.com/mentat-asi/mentat-sys1-v0.1#run-the-frozen-release).

## Intended Use

The model answers bounded `noul`, `choice`, and `score` decisions through a
TypeSafe-compatible `/v1/systemone` endpoint. It returns a native probability
distribution over the supplied labels. Inference uses one model pass and a
frozen scalar temperature of `1.0551417227266353`.

The model is not a general chat assistant. It is not intended to replace human
review in high-impact decisions.

## Use the model

Mentat uses the same TypeSafe wire format as Jev for the core
[`POST /v1/systemone` contract](https://docs.typesafe.ai/api). Self-hosted
Mentat does not require an API key.

```bash
export ENDPOINT_URL="http://127.0.0.1:8765"
curl --fail --request POST "$ENDPOINT_URL/v1/systemone" \
  --header "Content-Type: application/json" \
  --data @- <<'JSON'
{
  "model": "mentat-sys1-v0.1",
  "state": {"listing": {"color": "blue", "kind": "shirt"}},
  "questions": {
    "category": {
      "type": "choice",
      "instructions": "Which category best describes the listing?",
      "criteria": {
        "shirt": "A garment worn on the upper body.",
        "shoe": "Footwear."
      }
    }
  }
}
JSON
```

The response follows the Jev-compatible answer shape. Values below are
illustrative:

```json
{
  "model": "mentat-sys1-v0.1",
  "answers": {
    "category": {
      "type": "choice",
      "choice": "shirt",
      "probabilities": {"shirt": 0.87, "shoe": 0.13},
      "confidence": 0.71,
      "unknown_probability": 0.03,
      "abstained": false
    }
  },
  "usage": {"input_tokens": 128, "rotations": 1, "images": 0}
}
```

Use `noul` for a true/false decision, `choice` for named alternatives, and
`score` for an ordered scale. Mentat adds diagnostic fields such as
`unknown_probability`, `abstained`, calibration identity, and local runtime
usage; Jev-compatible clients can ignore these additive fields.

## Local Evaluation

The frozen checkpoint was evaluated twice from fresh GPU server processes on
the Public-231 items at JevBench revision
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

The released Public-231 result is `200/231` (86.58%).

## JevBench Public-231 Leaderboard

Comparator values come from the `public_accuracy` field in JevBench's official
[v1.4.2.2 aggregate results](https://github.com/fstandhartinger/jevbench/blob/main/results/v1.4.2.2/jevbench-v1.4.2.2-results.json).
The table includes the official 4B and 4B-derived cohort, Jev 1.13.0 as the
closed reference, and this release's Mentat result. It is sorted by correct
answers; equal totals share a rank.

| Rank | Model | Public-231 | Accuracy |
| ---: | --- | ---: | ---: |
| 1 | Plumb-4B | 207 | 89.61% |
| **2** | **mentat-sys1-v0.1** | **200** | **86.58%** |
| 2 | Jev 1.13.0 | 200 | 86.58% |
| 4 | Imajev-4B | 199 | 86.15% |
| 5 | JevK5 v0.2.0 | 197 | 85.28% |
| 6 | decider-4b v2 | 193 | 83.55% |
| 7 | Hopper | 190 | 82.25% |
| 8 | SemIf (Qwen3.5-4B) | 187 | 80.95% |
| 8 | Jobe Qwen3.5-4B | 187 | 80.95% |
| 10 | local-jev Qwen3.5-4B | 186 | 80.52% |
| 11 | metask-jev-4b | 184 | 79.65% |
| 12 | reflex 4B | 183 | 79.22% |
| 12 | spark-s1-4b-v6 | 183 | 79.22% |
| 14 | OpenSourceJev (Qwen3.5-4B) | 181 | 78.35% |
| 15 | typecastlm (Qwen3.5-4B) | 180 | 77.92% |
| 16 | Malkuth-4B | 173 | 74.89% |
| 17 | open-alternative-jev (Qwen3.5-4B) | 171 | 74.03% |
| 18 | ZeroEntropy zerank-2 | 162 | 70.13% |
| 19 | Raw Qwen3 4B Instruct 2507 | 161 | 69.70% |
| 20 | Qwen3-Reranker-4B | 157 | 67.97% |
| 21 | kev 4B | 153 | 66.23% |

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

- Code: `https://github.com/mentat-asi/mentat-sys1-v0.1`, tag `v0.1.1`
- Model: `https://huggingface.co/yunqu/mentat-sys1-v0.1`, tag `v0.1.0`

## License

Project code and released model-specific files use Apache-2.0. The unchanged
Qwen base model is downloaded separately and retains its upstream Apache-2.0
license. Adapted implementation sources and revisions are listed in `NOTICE`.
