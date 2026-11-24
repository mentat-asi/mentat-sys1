# mentat-Spice-4B inference

Python 3.12 · Linux · NVIDIA CUDA GPU. Weights are currently private: sign in with an HF account that has access to [yunqu/mentat-Spice-4B](https://huggingface.co/yunqu/mentat-Spice-4B).

## Install and run

```bash
pip install "mentat-sys1[serve] @ https://github.com/mentat-asi/mentat-sys1/releases/download/v0.4.0/mentat_sys1-0.4.0-py3-none-any.whl"
hf auth login
mentat-spice predict --demo
```

The installer downloads an approximately 80 KB code-only wheel; demos and evaluation records stay in GitHub. The first run downloads only four adapter files and the Qwen3.5-4B backbone at revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`. Subsequent runs reuse the HF cache. The demo checks whether an address missing its street number is complete and prints a System One JSON response. Inference runs locally; the Hub is used to download files.

For your own request, replace `--demo` with `--input request.json`. See [example request](../configs/spice-example.json).

## Python and images

```python
from mentat_spice import DecisionModel

model = DecisionModel.from_pretrained("yunqu/mentat-Spice-4B")
response = model.predict({
    "state": "A delivery address is missing its street number.",
    "questions": {"complete": {
        "type": "noul", "instructions": "The delivery address is complete."
    }},
})
print(response["answers"])
```

For image decisions, add `images=["./image.png"]` to `predict`. Inputs can be local image paths, PIL images or encoded bytes. Up to four images are accepted, each at most 1,638,400 pixels and 2,048 visual tokens. Native context is 32,768 processed tokens; input limits do not guarantee accuracy at every length.

`choice`, `noul` and `score` return candidate distributions, Boolean probabilities and ordered scores. Choice/score probabilities are conditional on the supplied candidates; `unknown_probability` and `abstained` are separate. Set action thresholds on your own validation data. Each question is evaluated separately without generating an answer explanation. Generic text-generation pipelines do not reproduce the decision attention mask and LM readout.

## Runtime selection

Native is the reference default. For acceleration, install the optional kernels and select a runtime:

```bash
pip install "mentat-sys1[accelerate] @ https://github.com/mentat-asi/mentat-sys1/releases/download/v0.4.0/mentat_sys1-0.4.0-py3-none-any.whl"
mentat-spice predict --runtime turbo --demo
```

Python: `DecisionModel.from_pretrained("yunqu/mentat-Spice-4B", runtime="turbo")`.

| Runtime | Behavior |
| --- | --- |
| native | Reference text/image scoring with the original LM projection |
| accelerated | Faster text preparation and qualified CUDA Graphs; unmerged adapter |
| turbo | Two resident cores: merged/fused text path with an independent native guard and fallback |

Turbo combines FLA Triton convolution, compiled pointwise operations and CUDA Graphs for eligible text inputs up to 2,048 tokens. Graph capture is limited to eight shapes; other shapes fall back. Images use native eager inference. First use includes compilation and capture; ambiguous probabilities may trigger additional native work. About 18 GiB was allocated in the measured setup, plus allocator/driver headroom. Keep Triton's default floating-point fusion enabled.

A warm A800 probe (251 input tokens, one four-choice question, 100 sequential Python calls) measured **21.9 ms median / 22.3 ms P95** with Turbo. Loading, compilation and HTTP transport are excluded. See [raw timing evidence](spice/latency-results.json). This is workload-specific; fresh matched Jev API timing is unavailable. Turbo can change probabilities: first-shape checks and confidence guards do not guarantee accuracy for future inputs. Full accelerated ID/DI scores have not been measured. Native Transformers may already dispatch installed FLA kernels.

## HTTP service

```bash
mentat-spice serve --host 127.0.0.1 --port 8765
curl --fail http://127.0.0.1:8765/v1/systemone \
  -H 'Content-Type: application/json' \
  -d '{"state":"Street number missing.","questions":{"complete":{"type":"noul","instructions":"The delivery address is complete."}}}'
```

Add `--runtime turbo` for acceleration. Use one worker per GPU; calls are serialized within an instance. The server accepts text/JSON states and questions and rejects local image paths; use Python for images. It has no built-in authentication: keep loopback binding or place it behind an authenticated gateway. Request bodies are limited to 1 MiB; `/health` reports readiness.

## Offline and local loading

The downloader honors the normal HF cache. Set `HF_HOME` before starting, or pass `--cache-dir /path/to/cache` / Python `cache_dir=...`. Use `--offline` or `local_files_only=True` after all files have been cached.

Existing local downloads continue to work:

```bash
mentat-spice predict --base-model ./qwen-base --model-package ./model-spice --demo
```

Python: `DecisionModel("./qwen-base", "./model-spice/model")`. The model package root contains `model/` with all four files. To reuse a local backbone while downloading the adapter, use `DecisionModel.from_pretrained(base_model="./qwen-base")`. To pin a specific adapter version, pass Python `revision="<commit>"` or CLI `--revision <commit>`; the backbone revision remains fixed.

If downloading fails with 401/403, verify that the account shown by `hf auth whoami` has model access. An offline cache miss needs one online download before offline use. No API key should be inserted into example source files.

## Run the workflows

Clone this repository for executable examples and archived records:

```bash
git clone https://github.com/mentat-asi/mentat-sys1.git
cd mentat-sys1
python docs/spice/examples/run.py --case invoice
python docs/spice/examples/run.py --case receipt
python docs/spice/workflows/run.py --output ./workflow-run-001
```

The runners download/reuse the same model automatically. `--base-model` optionally reuses a local backbone; `--adapter-directory` with `--base-model` loads entirely local files. Use a new workflow output directory. For saved evidence without inference: `python docs/spice/examples/run.py --case babyai --recorded`.

The three workflows create a support ticket, check an invoice and verify a retrieved-source draft through local SQLite transactions. Controllers execute writes and validate postconditions; no money moves and no external site is operated. Publishing repair extracts source sentences, which the model verifies. The example runner scores saved states; `--recorded` only reads archived responses. [Example sources and media terms](spice/assets/ATTRIBUTION.md).

## Measure your workload

```bash
python docs/spice/benchmark_latency.py --turbo \
  --warmup 10 --samples 100 --output ./latency-run-001
```

The harness downloads/reuses model files and retains raw samples, environment and scope. Omit `--turbo` for native; use `--accelerated` for the unmerged fast runtime. Existing local paths can be supplied with `--base-model` and `--adapter-directory`. Compare identical requests, runtime, device, warmup and concurrency: Python latency, localhost HTTP latency and a remote API's latency measure different scopes.

## Decision Index author engine

The [DI inference handoff](../decision-index/README.md) provides a pinned installer,
Turbo-enabled `spice_di:SpiceEngine`, native/accelerated comparison modes and a
portable smoke check for the official text and vision runners. It handles opaque
question IDs, long instructions, all supplied options and multi-question inputs
without truncation. It does not require the author's server or training code.

Turbo's short-text latency does not establish full DI latency or score equivalence.
Use the same pinned engine and runtime for public and private runs; retain complete
official public run artifacts before submitting. The handoff does not submit to
the leaderboard or upload results automatically.

## Repository layout

- **HF:** four model files, model card, licenses and displayed media.
- **GitHub:** runtime, installation, HTTP API, examples, workflow code and evaluation records.
- **Qwen Hub:** separately licensed pinned backbone, downloaded automatically.

The earlier v0.3 model still uses `mentat-sys1-v03`; [its guide](inference-v0.3.md) remains available.
