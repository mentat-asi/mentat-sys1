# mentat-Spice-4B: Decision Index author engine

This is an inference-only handoff for Decision Index maintainers. It loads the
released adapter and pinned Qwen backbone locally. No connection to the author's
server, training code or training data is needed. Public and private runs must
use the same engine revision and options.

## Install and check

Linux, Python 3.12, NVIDIA CUDA GPU and a compiler toolchain are required. Use a
dedicated GPU worker. The tested software stack is PyTorch 2.8.0 with CUDA 12.8,
Transformers 5.17.0, PEFT 0.21.0, FLA 0.5.2 and Triton 3.4.0. The maintainer's
RTX PRO 6000 hardware still needs its own qualification; A800 timing does not
establish RTX performance.

```bash
bash install.sh
bash run.sh smoke
```

`install.sh` creates an isolated `.venv`, installs the hash-pinned public runtime
wheel and commit-pinned official DI kit, applies the v0.4.0 input compatibility
fix, and checks dependencies and CUDA. Downloaded model files use the normal HF
cache. If access is restricted, authenticate an account granted model access.
If the exact dependency stack cannot be installed, report the error instead of
silently changing versions or falling back to another runtime.

The compatibility fix accepts opaque question IDs, long instructions and long
option descriptions. Nothing is truncated. Context and image token capacity
checks still apply. The original adapter weights, prompts, candidate mapping,
calibration and scoring code are retained.

The first Turbo call compiles kernels and captures shapes. The smoke check reports
first-call time separately from warm timings, validates probability distributions
and prints acceleration counters. Compilation and first-use shape capture are
not silently subtracted from subsequent official request timings.

## Run the public suites

Build and strictly verify the official suites first. Do not use `--no-verify`.
For text, run from the working directory containing the verified `suite-0.3`:

```bash
/absolute/path/to/inference/run.sh text --out runs/mentat-spice-4b-turbo-text
/absolute/path/to/inference/run.sh vision --dir /absolute/path/to/vision-suite \
  --out runs/mentat-spice-4b-turbo-vision
```

If executing without the executable bit, use `bash /absolute/path/to/inference/run.sh`.
Only the text and vision `run.sh` commands above are full-run commands. The
`smoke` command and bundled test evidence are integration checks, not DI scores.

## Run the private suites

Use the unmodified official runner with this author engine on `PYTHONPATH`:

```bash
export PYTHONPATH=/absolute/path/to/inference
export HF_ENDPOINT=https://huggingface.co
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
```

Engine class: `spice_di:SpiceEngine`. Fixed options:

```json
{
  "model": "yunqu/mentat-Spice-4B",
  "revision": "90d4f76971ebd07c11054ada274767458b615ad5",
  "device": "cuda:0",
  "runtime": "turbo"
}
```

It implements `__call__(state, questions, images=(), prompt=None) -> (response, raw)`.
The official vision runner supplies `image_root`; do not supply it a second time.
Use one process and one worker. `synchronize()` synchronizes CUDA for timing.
Unknown outcomes raise `NativeAbstention`; declared capacity limits raise
`Unsupported`. Every original question and option is retained; multi-question
requests use fixed groups of eight regardless of task. There is no benchmark
name, evaluation label, answer cache or task-specific prompt branch.

For offline loading, replace `model` with a local package containing `model/`
and add `base_model` pointing to the downloaded pinned backbone. The Hub loader
downloads only the four adapter/config files plus the backbone.

## Acceleration and limits

- Turbo uses fused pointwise operations, FLA Triton kernels and exact-shape CUDA
  Graphs. It keeps separate merged fast and unmerged fallback cores; plan for
  more VRAM than native mode. Tested peak memory is in the validation evidence.
- The released runtime accepts graph candidates only for ordinary unpadded text
  up to 2,048 tokens, with at most eight fast captured shapes. First-use numerical
  checks and per-call confidence guards may choose the native fallback. Different
  contents of one shape are not guaranteed numerically identical to native.
- Images, long contexts and ineligible inputs retain the native core; images are
  not processed by the fused text path. Adapter compute weights are BF16 in
  Turbo, so probability equality is measured rather than assumed.
- Native limits remain: 32,768 processed tokens, 255 choices and four images.
  Official image decoding caps each image at 1,638,400 pixels. The model's native
  processor also enforces its 2,048 visual tokens per image.
- `raw.acceleration_delta` records whether fast outputs or fallback graphs were
  actually used for each request. `environment.json` records the runtime settings
  and adapter source hash. Cold compilation, warm short-text microbenchmarks and
  diverse public DI request timings are distinct measurements.

`runtime=native` and `runtime=accelerated` are available for diagnostic comparisons.
Changing runtime requires a separate public run directory and corresponding
public results. Native quality scores must not be paired with Turbo latency as
if they were measured in the same configuration.

## Submission status

Local A800 integration checks passed for 42 text requests (217 judgments) and
12 vision requests covering all 11 public vision tasks. Compared with the native
engine, there were no changed choices or Boolean decisions; the largest text
probability difference was 0.001876 and vision probabilities matched exactly.
This sample does not establish full-suite equivalence. See [validation.json](validation.json)
for timing scopes, source hashes and the limits of the installation check.

No leaderboard PR or result upload is made by these scripts. A formal submission
still requires complete, untouched public runs (`complete: true`) produced using
this exact inference configuration. The author's existing local text corpus
lacks 501 gated HLE requests; obtain access and strictly rebuild/verify that
corpus before a formal text submission. No such source access issue was found
in the verified 12,210-row public vision corpus.

Official interfaces and submission process:
[engines](https://github.com/apolinario/decision-index/blob/main/docs/engines.md) ·
[submission](https://github.com/apolinario/decision-index#submitting-a-model-to-the-leaderboard).
