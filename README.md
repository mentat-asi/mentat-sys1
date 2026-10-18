<h1 align="center">mentat-sys1-v0.3</h1>
<p align="center"><strong>Decisions with probabilities.</strong><br>THU-IDM · Tsinghua University</p>

A 4B model for choosing actions, using tools and making structured decisions. Give it a state and typed questions; it returns answer probabilities in one forward pass per question, without generating answer text.

Qwen3.5-4B · Text / JSON · 16K context · LoRA adapter + decision readout

[Team · THU-IDM](https://www.thuidm.com) · [Code](https://github.com/mentat-asi/mentat-sys1) · [Demos](#see-it-in-action) · [Model](https://huggingface.co/yunqu/mentat-sys1-v0.3) · [Quick start](#quick-start) · [Results](#results) · [Inference guide](docs/inference-v0.3.md)

## See it in action

Recorded games and completed workflows. Click a demo to watch the enlarged replay.

<table class="demo-grid">
<tr>
<td width="33%" valign="top"><strong>Mario</strong><br><a href="https://huggingface.co/yunqu/mentat-sys1-v0.3/resolve/v0.3/demos/recorded-games/tiles/mario.gif"><img src="https://huggingface.co/yunqu/mentat-sys1-v0.3/resolve/v0.3/demos/recorded-games/tiles/mario.gif" alt="Mario: actual recorded environment frames" width="100%"></a><br>Jump over obstacles and reach the goal.</td>
<td width="33%" valign="top"><strong>Football</strong><br><a href="https://huggingface.co/yunqu/mentat-sys1-v0.3/resolve/v0.3/demos/recorded-games/tiles/football.gif"><img src="https://huggingface.co/yunqu/mentat-sys1-v0.3/resolve/v0.3/demos/recorded-games/tiles/football.gif" alt="Football: actual recorded environment frames" width="100%"></a><br>Play against JevBall’s built-in AI team.</td>
<td width="33%" valign="top"><strong>Doom</strong><br><a href="https://huggingface.co/yunqu/mentat-sys1-v0.3/resolve/v0.3/demos/recorded-games/tiles/doom.gif"><img src="https://huggingface.co/yunqu/mentat-sys1-v0.3/resolve/v0.3/demos/recorded-games/tiles/doom.gif" alt="Doom: actual recorded environment frames" width="100%"></a><br>Battle seven built-in Doom bots.</td>
</tr>
<tr>
<td width="33%" valign="top"><strong>Shopping</strong><br><a href="https://huggingface.co/yunqu/mentat-sys1-v0.3/resolve/v0.3/demos/recorded-games/tiles/store.gif"><img src="https://huggingface.co/yunqu/mentat-sys1-v0.3/resolve/v0.3/demos/recorded-games/tiles/store.gif" alt="Shopping: actual recorded environment frames" width="100%"></a><br>Compare prices and delivery options, then place an order.</td>
<td width="33%" valign="top"><strong>Workspace</strong><br><a href="https://huggingface.co/yunqu/mentat-sys1-v0.3/resolve/v0.3/demos/recorded-games/tiles/workspace.gif"><img src="https://huggingface.co/yunqu/mentat-sys1-v0.3/resolve/v0.3/demos/recorded-games/tiles/workspace.gif" alt="Workspace: actual recorded environment frames" width="100%"></a><br>Invite a teammate with the requested permissions.</td>
<td width="33%" valign="top"><strong>Chess</strong><br><a href="https://huggingface.co/yunqu/mentat-sys1-v0.3/resolve/v0.3/demos/recorded-games/tiles/chess.gif"><img src="https://huggingface.co/yunqu/mentat-sys1-v0.3/resolve/v0.3/demos/recorded-games/tiles/chess.gif" alt="Chess: actual recorded environment frames" width="100%"></a><br>Play against a greedy chess bot.</td>
</tr>
</table>

The model receives structured **text/JSON** states and selects actions.

<sub>Controllers provide Mario lookahead, Doom aiming/navigation, football physics/off-ball AI, and chess legal/tactical facts. Shopping and Workspace use local mock sites. Game clips show selected highlights.</sub>

## Results

| Measurement | mentat-sys1-v0.3 | Jev, reported |
|---|---:|---:|
| Intern-Decision — average accuracy (%) ↑ | **89.59%** | 88.74% |
| JevBench — correct answers (public set) ↑ | **204/231** | 199/231 |
| Typed Decisions — Brier score ↓ | **0.05403** | 0.148 |
| Decision Index 0.2.1 ↑ | 52.07* | **57.91** |
| Latency (ms) ↓ | **21.47 ms** | 150 ms† |

<sub>Jev benchmark results: [StartLux results](https://github.com/StartLuxLabs/StartLux-Decision/blob/a9101e918e097b26fea5ead9ed893b823e875e46/docs/results.md).</sub>

<sub>\* DI: 149,816 requests, 37 tasks; HLE unmeasured (zero contribution, original weights); abstentions count as wrong.</sub>

<sub>† mentat: one question, accelerated runtime, warm A800, 600 localhost HTTP requests. Jev: [TypeSafe’s official 150 ms reference](https://docs.typesafe.ai/concepts/use-case-map#real-time-applications), question count unspecified.</sub>

## Quick start

Python 3.12 · Linux · NVIDIA CUDA GPU. Download the adapter package and Qwen base separately:

```bash
git clone https://github.com/mentat-asi/mentat-sys1.git
cd mentat-sys1
pip install -e ".[accelerate]"

hf download yunqu/mentat-sys1-v0.3 --revision v0.3 --local-dir model-v0.3
hf download Qwen/Qwen3.5-4B \
  --revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a \
  --local-dir qwen-base

mentat-sys1-v03 predict --model-package model-v0.3 --base-model qwen-base \
  --input model-v0.3/example.json
```

The accelerated runtime is enabled by default. For the original probability computation, select `--runtime native` and install with `pip install -e ".[serve]"`. Run the two modes in separate processes. [Inference guide](docs/inference-v0.3.md).

## Serve the model

```bash
mentat-sys1-v03 serve --model-package model-v0.3 --base-model qwen-base \
  --warmup-request model-v0.3/example.json --host 127.0.0.1 --port 8765

curl --fail --request POST http://127.0.0.1:8765/v1/systemone \
  --header 'Content-Type: application/json' \
  --data-binary @model-v0.3/example.json
```

Supports named choices (`choice`), yes/no (`noul`) and ordered scales (`score`). The API returns probabilities, confidence and abstention diagnostics, with zero generated answer tokens.

## Acceleration included

All six optimizations are available through the CLI and Python API: in-memory LoRA merging, batched questions, request-local shared prefixes, FLA Triton kernels, pointwise fusion and CUDA Graphs.

| Runtime | One question · HTTP mean | Three questions · HTTP mean | Intern-Decision |
|---|---:|---:|---:|
| Native v0.3 | 83.60 ms | 249.86 ms | 89.59% |
| Accelerated v0.3 | **21.47 ms** | **40.66 ms** | 89.52% |

Warm A800 measurements, 600 requests per scenario; loading, compilation and graph capture excluded. Accelerated JevBench is also **204/231**; accelerated DI is unmeasured. Shapes are captured on first use within configurable memory limits; other shapes run eagerly. [Runtime options](docs/inference-v0.3.md#runtime-options).

## Earlier releases

The original `mentat-sys1` CLI continues to support v0.1 and v0.2. Their frozen configurations and documentation remain available.

| Model | Weights | Code |
|---|---|---|
| v0.2 | [mentat-sys1-v0.2](https://huggingface.co/yunqu/mentat-sys1-v0.2/tree/v0.2.0) | [v0.2.0](https://github.com/mentat-asi/mentat-sys1/tree/v0.2.0) |
| v0.1 | [mentat-sys1-v0.1](https://huggingface.co/yunqu/mentat-sys1-v0.1/tree/v0.1.0) | [v0.1.1](https://github.com/mentat-asi/mentat-sys1/tree/v0.1.1) |

## License

Adapter and bundled code: Apache-2.0. The separately downloaded Qwen base retains its upstream license. [NOTICE](NOTICE).
