# mentat-sys1-v0.1

**200/231 (86.58%) on the JevBench Public-231 set.** A one-pass,
open-weights decision model built on Qwen3.5-4B, with native option
probabilities and zero generated answer tokens.

[Code v0.1.1](https://github.com/mentat-asi/mentat-sys1-v0.1/tree/v0.1.1)
| [Model package v0.1.0](https://huggingface.co/yunqu/mentat-sys1-v0.1/tree/v0.1.0)
| [JevBench request #186](https://github.com/fstandhartinger/jevbench/issues/186)

| Tier | Correct | Total |
| --- | ---: | ---: |
| Easy | 48 | 48 |
| Original | 66 | 72 |
| Hard | 86 | 111 |
| **All** | **200** | **231** |

Two fresh A800 server processes produced the same prediction digest, with zero
invalid responses and zero severe failures. Public-set ECE was `0.05835`;
serial p50 latency was `0.090-0.091 s` and p95 was `0.332-0.336 s`.

## JevBench Public-231 leaderboard

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

## Run the frozen release

```bash
git clone https://github.com/mentat-asi/mentat-sys1-v0.1.git
cd mentat-sys1-v0.1
git checkout v0.1.1
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
  --host 127.0.0.1 \
  --port 8765
```

`MODEL_PACKAGE_DIR`, `QWEN_BASE_DIR`, and `RUNTIME_DIR` are paths you choose.
The first `hf download` fetches the Mentat LoRA adapter, decision readout,
calibration, manifests, and audit evidence. It does not fetch the Qwen base
weights; the second command downloads that exact base revision separately.

The server exposes a TypeSafe-compatible `POST /v1/systemone` endpoint. It
answers bounded `noul`, `choice`, and `score` decisions in one model pass with
a 16,384-token input limit.

## Evidence and scope

- Model package: LoRA adapter, native decision readout, scalar calibration,
  checksums, and aggregate audit evidence.
- Repeated-run prediction digest:
  `b8eec370fbae0ae0e560cfd2480ebe25a6a10dae870fb2e6e1ea70c4dc066159`.
- Registered training projection: 87,386 rows; zero exact IDs, derivative IDs,
  source groups, image hashes, or unresolved word 8-gram overlaps with the 231
  public items.
- Full training-data reconstruction is not claimed because the public source
  and license ledger is incomplete.

See the [model card](docs/MODEL_CARD.md),
[reproducibility guide](docs/REPRODUCIBILITY.md), and
[bench request record](docs/JEVBENCH_SUBMISSION.md) for pinned hashes and
method details.
