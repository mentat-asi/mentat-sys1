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

## Public-231 context

The table below uses the same 231 public items: 48 Easy, 72 Original, and 111
Hard. It covers source-linked, open-weight, approximately 4B text-capable
submissions whose public issue reports exact tier counts as of 2026-10-04.
Where a family has several releases, the latest release with a complete tier
split is shown; materially different deployments remain separate.

These are submitter-reported public-set results, not an official leaderboard
or a substitute for JevBench's current evaluator-controlled open and sealed
suites. The current 534/904-item results are intentionally not mixed into this
legacy Public-231 comparison.

| Model | All | Easy | Original | Hard | Source |
| --- | ---: | ---: | ---: | ---: | --- |
| Loupe 1.1 | 207 | 48 | 72 | 87 | [#161](https://github.com/fstandhartinger/jevbench/issues/161) |
| Plumb-4B | 207 | 48 | 70 | 89 | [#84](https://github.com/fstandhartinger/jevbench/issues/84) |
| H2O-Lightning-4B v1.0 | 205 | 48 | 71 | 86 | [#181](https://github.com/fstandhartinger/jevbench/issues/181) |
| OpenJev-4B | 204 | 48 | 72 | 84 | [#159](https://github.com/fstandhartinger/jevbench/issues/159) |
| Decision 4B v1.2 | 203 | 48 | 69 | 86 | [#86](https://github.com/fstandhartinger/jevbench/issues/86) |
| deck-4B v1.0 | 203 | 48 | 68 | 87 | [#100](https://github.com/fstandhartinger/jevbench/issues/100) |
| JevK5 v0.3 | 203 | 48 | 68 | 87 | [#31](https://github.com/fstandhartinger/jevbench/issues/31#issuecomment-5827758155) |
| JPT-4B | 203 | 48 | 68 | 87 | [#78](https://github.com/fstandhartinger/jevbench/issues/78) |
| Intern-Decision-4B | 201 | 48 | 71 | 82 | [#158](https://github.com/fstandhartinger/jevbench/issues/158) |
| mentat-sys1-v0.1 | **200** | 48 | 66 | 86 | [#186](https://github.com/fstandhartinger/jevbench/issues/186) |
| blink-4b | 199 | 48 | 71 | 80 | [#81](https://github.com/fstandhartinger/jevbench/issues/81) |
| Imajev-4B | 197 | 48 | 71 | 78 | [#80](https://github.com/fstandhartinger/jevbench/issues/80) |
| Tura-S1-4B | 197 | 48 | 70 | 79 | [#139](https://github.com/fstandhartinger/jevbench/issues/139) |
| decider-4b v2 | 193 | 48 | 71 | 74 | [#79](https://github.com/fstandhartinger/jevbench/issues/79) |
| DeskMind Brain 4B | 193 | 48 | 69 | 76 | [#173](https://github.com/fstandhartinger/jevbench/issues/173) |
| Hopper | 192 | 48 | 68 | 76 | [#14](https://github.com/fstandhartinger/jevbench/issues/14) |
| Mica v0.1 4B | 192 | 48 | 72 | 72 | [#91](https://github.com/fstandhartinger/jevbench/issues/91) |
| reflex 4B, two orders | 190 | 48 | 66 | 76 | [#5](https://github.com/fstandhartinger/jevbench/issues/5) |
| typecastlm 1.3.0 | 187 | 47 | 67 | 73 | [#168](https://github.com/fstandhartinger/jevbench/issues/168) |
| Jobe | 186 | 48 | 71 | 67 | [#28](https://github.com/fstandhartinger/jevbench/issues/28) |
| local-jev | 186 | 48 | 69 | 69 | [#15](https://github.com/fstandhartinger/jevbench/issues/15) |
| Quire | 186 | 48 | 65 | 73 | [#45](https://github.com/fstandhartinger/jevbench/issues/45) |
| RYOTIDE-Qwen | 184 | 48 | 63 | 73 | [#82](https://github.com/fstandhartinger/jevbench/issues/82) |
| tde-qwen3.5-4b-v0.1 | 184 | 48 | 70 | 66 | [#99](https://github.com/fstandhartinger/jevbench/issues/99) |
| Rev Qwen3.5-4B | 181 | 48 | 67 | 66 | [#52](https://github.com/fstandhartinger/jevbench/issues/52) |
| Compass 0.2.0 | 178 | 48 | 63 | 67 | [#53](https://github.com/fstandhartinger/jevbench/issues/53) |
| kapteeni-v1.1c | 177 | 48 | 67 | 62 | [#178](https://github.com/fstandhartinger/jevbench/issues/178) |
| Noma | 176 | 48 | 71 | 57 | [#172](https://github.com/fstandhartinger/jevbench/issues/172) |
| Malkuth-4B | 173 | 48 | 69 | 56 | [#77](https://github.com/fstandhartinger/jevbench/issues/77) |
| kapteeni-v1-intuit | 165 | 48 | 65 | 52 | [#102](https://github.com/fstandhartinger/jevbench/issues/102) |
| kapteeni-v1-meticulous | 164 | 48 | 64 | 52 | [#102](https://github.com/fstandhartinger/jevbench/issues/102) |

Mentat's `86/111` Hard result is competitive with the strongest systems in
this cohort. Its main gap is the Original tier at `66/72`, which is why the
full total trails the leading 4B releases despite the strong Hard score.
Plumb's source reports `89/111` in its primary table and `90/111` in a second
run; the table uses the conservative primary result.

Loupe 1.0 reported the same `207/231` split as 1.1; its archived source is
[issue #119](https://github.com/fstandhartinger/jevbench/issues/119).

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
