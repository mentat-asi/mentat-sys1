# mentat-sys1

An open-weights System One model for fast, calibrated decisions over bounded
options.

**Current release: Mentat v0.2, with 206/231 (89.18%) on JevBench
Public-231.**

[Code v0.2.0](https://github.com/mentat-asi/mentat-sys1/tree/v0.2.0)
| [Model v0.2.0](https://huggingface.co/yunqu/mentat-sys1-v0.2/tree/v0.2.0)
| [Release notes](docs/releases/v0.2.md)

## Results

`SemIF 187/231` -> `Mentat v0.1 200/231` -> `Mentat v0.2 206/231`

Mentat v0.2 answers 19 more public questions correctly than the
Qwen3.5-4B-based SemIF baseline and six more than Mentat v0.1.

| Release | Public-231 | Accuracy | Change from SemIF |
| --- | ---: | ---: | ---: |
| SemIF | 187/231 | 80.95% | - |
| Mentat v0.1 | 200/231 | 86.58% | +13 |
| **Mentat v0.2** | **206/231** | **89.18%** | **+19** |

Selected 4B and 4B-derived results:

| Rank | Model | Public-231 | Accuracy |
| ---: | --- | ---: | ---: |
| 1 | Plumb-4B | 207 | 89.61% |
| **2** | **Mentat v0.2** | **206** | **89.18%** |
| 3 | Mentat v0.1 | 200 | 86.58% |
| 3 | Jev 1.13.0 | 200 | 86.58% |
| 4 | Imajev-4B | 199 | 86.15% |
| 5 | JevK5 v0.2.0 | 197 | 85.28% |
| 6 | SemIF | 187 | 80.95% |

Comparator values are the `public_accuracy` results in JevBench's official
[v1.4.2.2 aggregate](https://github.com/fstandhartinger/jevbench/blob/main/results/v1.4.2.2/jevbench-v1.4.2.2-results.json).
Mentat values are local evaluations on the exact Public-231 set. The rank
shown is the dense order within this selected public-results table; it is not
an official composite JevBench rank. Sealed accuracy, official speed, and
official cost were not measured for v0.2.

## What it does

Mentat answers bounded decisions through a TypeSafe-compatible
`POST /v1/systemone` endpoint:

- `noul` for true/false decisions
- `choice` for named alternatives
- `score` for ordered scales

It returns native option probabilities in one model pass with zero generated
answer tokens. v0.2 uses frozen per-type calibration and accepts up to 16,384
input tokens. It is a decision model, not a general chat assistant.

## Run v0.2

Python 3.12, `uv`, and a CUDA GPU are required.

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
  --host 127.0.0.1 \
  --port 8765
```

`MODEL_PACKAGE_DIR`, `QWEN_BASE_DIR`, and `RUNTIME_DIR` are local paths you
choose. The Mentat package contains the LoRA adapter, native decision readout,
calibration, manifests, and checksums. The pinned Qwen base is downloaded
separately.

Before loading weights, the runtime verifies the v0.2 identity, adapter and
readout hashes, and `temperature_by_type-v2` calibration binding.

## Call the API

Mentat uses the same TypeSafe wire format as Jev for the core
[`POST /v1/systemone` contract](https://docs.typesafe.ai/api). Self-hosted
Mentat does not require an API key.

```bash
export ENDPOINT_URL="http://127.0.0.1:8765"
curl --fail --request POST "$ENDPOINT_URL/v1/systemone" \
  --header "Content-Type: application/json" \
  --data @- <<'JSON'
{
  "model": "mentat-sys1-v0.2",
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
  "model": "mentat-sys1-v0.2",
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

Mentat adds diagnostic fields such as `unknown_probability`, `abstained`,
calibration identity, and local runtime usage; Jev-compatible clients can
ignore these additive fields.

## Reproducibility

Both releases were rerun from repository commit `7c74c92` against JevBench
revision `2fa63fa3226cb369795525ed011800f57dcbd894` on an NVIDIA A800 80GB:

| Release | Easy | Original | Hard | Total |
| --- | ---: | ---: | ---: | ---: |
| v0.1 | 48/48 | 66/72 | 86/111 | 200/231 |
| **v0.2** | **48/48** | **71/72** | **87/111** | **206/231** |

Each run returned 231/231 valid responses, zero errors, and zero renormalized
responses. Comparing every rerun with its historical release output found
zero prediction differences.

See the [reproducibility guide](docs/REPRODUCIBILITY.md) for pinned
environments and artifact hashes.

## Releases

| Release | Code | Model package | Calibration |
| --- | --- | --- | --- |
| v0.2 | [`v0.2.0`](https://github.com/mentat-asi/mentat-sys1/tree/v0.2.0) | [`yunqu/mentat-sys1-v0.2@v0.2.0`](https://huggingface.co/yunqu/mentat-sys1-v0.2/tree/v0.2.0) | per-type v2 |
| v0.1 | [`v0.1.1`](https://github.com/mentat-asi/mentat-sys1/tree/v0.1.1) | [`yunqu/mentat-sys1-v0.1@v0.1.0`](https://huggingface.co/yunqu/mentat-sys1-v0.1/tree/v0.1.0) | scalar v1 |

The current runtime supports both model IDs. Use the matching config and model
package for each frozen release.

## Documentation

- [Model card](docs/MODEL_CARD.md)
- [Reproducibility](docs/REPRODUCIBILITY.md)
- [v0.2 release](docs/releases/v0.2.md)
- [v0.1 release](docs/releases/v0.1.md)
- [Data card](docs/DATA_CARD.md)

## License

The project and model-specific files are licensed under
[Apache License 2.0](LICENSE). The unchanged Qwen base retains its upstream
license.
