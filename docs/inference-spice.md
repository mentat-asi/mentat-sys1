# mentat-Spice-4B inference

Python 3.12, Linux and an NVIDIA CUDA GPU. The model repository is currently private; request access from its owner before downloading. This runtime is separate from the v0.3 runtime; use `mentat-spice` for Spice adapters.

## Install and download

```bash
git clone https://github.com/mentat-asi/mentat-sys1.git
cd mentat-sys1
pip install -e ".[accelerate]"
# Authenticate with an account that can read the private model.
hf auth login
hf download yunqu/mentat-Spice-4B --local-dir model-spice
hf download Qwen/Qwen3.5-4B \
  --revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a \
  --local-dir qwen-base
mentat-spice predict --base-model qwen-base --model-package model-spice \
  --input configs/spice-example.json
```

The backbone must be downloaded locally, including its processor and tokenizer. The adapter package's `model/` directory supplies the adapter, attention configuration and original LM readout bindings. A generic text-generation pipeline does not reproduce this decision interface.

## Runtime selection

Use `--runtime native` (default), `accelerated`, or `turbo` with either CLI command. Native installation only requires `pip install -e ".[serve]"`.

| Runtime | Behavior |
| --- | --- |
| native | Reference text/image scoring with the original LM projection |
| accelerated | Faster text preparation and first-use-qualified CUDA Graphs; unmerged adapter |
| turbo | Two independent resident cores: merged/fused text path with a native probability guard and fallback |

Turbo uses FLA Triton convolution, compiled pointwise operations and CUDA Graphs for eligible text inputs up to 2,048 tokens. Graphs are limited to eight shapes; other shapes run through fallback. Images use native eager inference. First use includes compilation and graph capture. Ambiguous probabilities can trigger extra native work, so acceleration is workload-dependent. Turbo requires about 18 GiB allocated GPU memory in the measured setup, plus allocator and driver headroom. Do not disable Triton's default floating-point fusion.

Turbo probabilities can differ from native probabilities. First-shape checks and confidence guards are not an accuracy guarantee; validate your deployment workload. Full accelerated ID/DI scores have not been measured. Native Transformers may already use installed FLA kernels.

A warm A800 text probe (251 input tokens, one four-choice question, 100 sequential Python calls) measured **21.9 ms median / 22.3 ms P95** with Turbo. This excludes loading, compilation and HTTP transport. It is not a latency promise for images, long contexts or queued requests. No fresh matched Jev API result is available yet.

```bash
mentat-spice predict --runtime turbo --base-model qwen-base \
  --model-package model-spice --input configs/spice-example.json
```

## Python and images

```python
from mentat_spice import DecisionModel
# For acceleration, import AcceleratedDecisionModel from mentat_spice.accelerated
# or TurboDecisionModel from mentat_spice.turbo instead.
model = DecisionModel("./qwen-base", "./model-spice/model")
result = model.predict({
    "state": "A delivery address is missing its street number.",
    "questions": {"complete": {
        "type": "noul", "instructions": "The delivery address is complete."
    }},
})
print(result["answers"])
```

For image decisions pass `images=["./image.png"]` to `predict`. Local image paths, PIL images and encoded bytes are accepted. The native context limit is 32,768 processed tokens; up to four images, each at most 1,638,400 pixels and 2,048 visual tokens. These limits are input guards, not measured accuracy guarantees at every length.

## HTTP service

```bash
mentat-spice serve --runtime turbo --base-model qwen-base \
  --model-package model-spice --host 127.0.0.1 --port 8765
curl --fail http://127.0.0.1:8765/v1/systemone \
  -H 'Content-Type: application/json' --data-binary @configs/spice-example.json
```

Use one worker per GPU; prediction calls are serialized within a model instance. The server accepts text/JSON states and named questions, and rejects image paths in HTTP requests. Use the Python interface for images. There is no built-in authentication; keep the default loopback binding or put an authenticated gateway in front of it. The request body limit is 1 MiB.

`choice`, `noul` and `score` produce candidate distributions, Boolean probabilities and ordered scores. Choice/score probabilities are conditional on provided candidates; `unknown_probability` and `abstained` are separate. Set application thresholds on your own validation data. Each question is scored separately, with no generated answer tokens.

## Run the workflows

The downloaded model package contains local executable workflows and recorded evidence:

```bash
python model-spice/workflows/run.py --base-model ./qwen-base --output ./workflow-run-001
python model-spice/examples/run.py --case invoice --base-model ./qwen-base
python model-spice/examples/run.py --case receipt --base-model ./qwen-base
python model-spice/examples/run.py --case babyai --recorded
```

The three workflows create a support ticket, check an invoice and verify a retrieved-source draft using local SQLite transactions. Controllers execute the transactions and check postconditions. No funds move and no external website is operated. Publishing repair extracts source sentences; the decision model does not generate the replacement text. Example scripts score saved decision states; `--recorded` only reads archived responses.

## Measure your workload

The model package includes a latency harness with text and image scenarios:

```bash
python model-spice/benchmark_latency.py --base-model ./qwen-base \
  --adapter-directory ./model-spice/model --turbo \
  --warmup 10 --samples 100 --output ./latency-run-001
```

Use a new output directory each time. The report retains raw timings and environment information. Compare the same request, runtime, device, warmup and concurrency; Python latency, localhost HTTP latency and a remote API's end-to-end latency are different measurements.
