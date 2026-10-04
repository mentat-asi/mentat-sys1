# mentat-sys1-v0.1

**200/231 (86.58%) on the legacy JevBench Public-231 set.** A one-pass,
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

## Official JevBench context

All comparator data below comes from JevBench-published result artifacts, not
submission reports.

### Current composite leaderboard

The current
[v1.4.2.2 aggregate results](https://github.com/fstandhartinger/jevbench/blob/main/results/v1.4.2.2/jevbench-v1.4.2.2-results.json)
cover 95 systems and rank 91. The JevBench Score combines Intelligence,
Calibration, Speed, and Cost, including evaluator-controlled sealed tasks.

| Rank | System | JevBench Score |
| ---: | --- | ---: |
| 1 | Imajev-4B | 67.37 |
| 2 | Plumb-4B | 65.84 |
| 3 | decider-4b v2 | 64.13 |
| 4 | Jev 1.13.0 | 63.29 |
| 5 | JevK5 v0.2.0 | 62.04 |

### Public-231

JevBench's official
[v1.2 per-task results](https://github.com/fstandhartinger/jevbench/blob/main/results/v1.2/jevbench-v1.2-per-task.json)
provide evaluator-produced counts on the same 231 public items: 48 Easy, 72
Original (`standard` in the artifact), and 111 Hard.

| System | All | Easy | Original | Hard |
| --- | ---: | ---: | ---: | ---: |
| **mentat-sys1-v0.1** | **200** | **48** | **66** | **86** |
| Jev 1.13.0 | 200 | 48 | 71 | 81 |
| SemIf (Qwen3.5-4B) | 187 | 48 | 71 | 68 |
| reflex 4B | 183 | 48 | 68 | 67 |
| open-alternative-jev (Qwen3.5-4B) | 171 | 48 | 60 | 63 |
| Qwen3-Reranker-4B | 157 | 48 | 54 | 55 |
| kev 4B | 153 | 48 | 64 | 41 |

Mentat matches Jev 1.13.0's Public-231 total with a different tier profile:
`86/111` versus `81/111` on Hard and `66/72` versus `71/72` on Original.
Public-231 counts and the current v1.4.2.2 composite score are separate
protocols.

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
