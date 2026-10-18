# v0.3 inference

Install `pip install -e ".[accelerate]"` for all acceleration features, or `pip install -e ".[serve]"` for the native runtime. Python 3.12, the pinned CUDA PyTorch/Transformers versions and the separately downloaded Qwen3.5-4B base are required. The accelerated dependencies are pinned to the tested FLA 0.5.2 and Triton 3.4.0; first use compiles kernels.

## Python API

```python
import json
from mentat_sys1.v03 import load_model

model = load_model(model_package="model-v0.3", base_model="qwen-base",
                   runtime="accelerated", device="cuda:0")
payload = json.load(open("model-v0.3/example.json"))
answer = model.predict(payload)
```

`model_package` is the full directory downloaded from `yunqu/mentat-sys1-v0.3@v0.3`; its checksummed inference code and files are required. The loader verifies file checksums and the frozen adapter/readout identities. No absolute machine paths, private artifacts or training code are needed.

## Runtime options

| Option | Default | Behavior |
|---|---|---|
| `--runtime` | `accelerated` | `native` preserves the original unmerged release computation |
| `--device` | `cuda:0` | CUDA device inside the selected process |
| `--threads` | `4` | PyTorch CPU threads |
| `--no-cuda-graphs` | off | Run batched prefix/fused inference eagerly |
| `--max-graphs` | `8` | Maximum number of cached graph shapes; zero disables capture |
| `--max-graph-tokens` | `4096` | Maximum padded batch tokens eligible for graph capture |
| `--warmup-request` | none | `serve` only; repeat for representative request JSON files |

Larger or uncaptured shapes run eagerly. First use includes kernel compilation and graph capture; these startup costs are excluded from the reported warm latency. The CUDA graph cache stores computation graphs, not answers. Every request computes fresh model activations and recurrent state, including any shared prefix. Prediction calls are serialized because CUDA graph buffers are reused. This is a single-worker server; use separate processes for native and accelerated modes because the accelerated kernels install process-local hooks.

The base and adapter files are never modified. BF16 adapter merging and fused kernels can change rounding: the accelerated seven-suite accuracy is 89.52%, versus 89.59% native; the public JevBench total is 204/231 in both. The main quality table uses native results. Accelerated DI is unmeasured.

## HTTP

`POST /v1/systemone` accepts the model package’s `example.json` with `model` set to `mentat-sys1-v0.3`. `GET /health` returns model ID and runtime; `GET /v1/models` lists the active model. No API key is required for this local server.

Input types are `choice`, `noul` and `score`. Images are unsupported in this text/JSON release. Input and output use the [model package’s documented contract](https://huggingface.co/yunqu/mentat-sys1-v0.3/blob/v0.3/docs/inference.md).

## Verify downloads

```bash
mentat-sys1-v03 verify --model-package model-v0.3
```

The legacy `mentat-sys1` commands and v0.1/v0.2 configurations remain available for those model packages.
