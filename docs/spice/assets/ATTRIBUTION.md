# Example media and benchmark attribution

The repository-level Apache-2.0 license covers the mentat adapter and inference code.
Third-party examples retain the separate terms below. No affiliation or endorsement is implied.
Graphics add mentat framing, captions and recorded model probabilities to the cited material.

| Files | Source and attribution | Terms |
| --- | --- | --- |
| `example-chart.gif`, `example-chart.png`, `examples/chart.json` | CharXiv, Wang et al. (2024), validation row 444 / figure `images/1028.jpg`. Original chart: Robert J. Taggart, *Point forecasting and forecast evaluation with generalized Huber loss*, [arXiv:2108.12426](https://arxiv.org/abs/2108.12426), `robust_fig4.jpg`. | Original chart: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), as linked by the paper. CharXiv question and this annotated question/chart composition: [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). |
| `example-web.gif`, `example-web.png`, `examples/web.json` | Mind2Web / Multimodal-Mind2Web, OSU NLP Group; Deng et al. (2023), Zheng et al. (2024). Six reference screenshots with target highlights and recorded model choices. | Original [Mind2Web dataset](https://huggingface.co/datasets/osunlp/Mind2Web): CC BY 4.0. The [multimodal dataset card](https://huggingface.co/datasets/osunlp/Multimodal-Mind2Web) separately lists OpenRAIL and a research-purpose statement. Website graphics and trademarks remain with their owners. |
| `example-visual.gif`, `example-visual.png`, `examples/visual.json` | [BLINK](https://huggingface.co/datasets/BLINK-Benchmark/BLINK), Fu et al. (2024), `val_Semantic_Correspondence_3`. Paired images with recorded correspondence judgment. | Dataset card: Apache-2.0. |
| `example-customer.gif`, `example-customer.png`, `example-security.gif`, `example-security.png`, `examples/customer.json`, `examples/security.json` | [Typed Decisions](https://huggingface.co/datasets/LocalLLaMA/typed-decisions), customer_service_000017 and security_incidents_000084. | Dataset card: Apache-2.0. |
| `example-babyai.gif`, `example-babyai.png`, `examples/babyai.json` | [MiniGrid / BabyAI](https://github.com/Farama-Foundation/Minigrid), Farama Foundation and contributors. MiniGrid 3.2.0 rendering of recorded partial observations from one complete UnlockToUnlock episode. | Apache-2.0. |

The browser is a replay of decisions on supplied reference states. The game is an actual completed episode using structured partial observations. These are selected successful demonstrations, not success-rate measurements or evidence of unseen-task generalization.

Reference licenses: [Apache-2.0](../../../LICENSE), [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).

## Provider benchmark references

The Typed Decisions panel adapts published numerical results from
[System One Bench](https://systemonebench.dev/benchmarks/typed-decisions/),
by [alilibx and contributors](https://github.com/alilibx/system-one-bench),
whose results are published under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
Selected reference rows have been redrawn, and mentat's local predictions have
been rescored using the same modal-label rule. The graphic is not an official
System One Bench leaderboard entry. Original model names and scores are retained.

The marked Gemini and OpenAI JevBench numbers are numerical results reported in
[Google's Gemini Decisions API announcement](https://blog.google/innovation-and-ai/technology/developers-tools/gemini-decisions-api/).
Google's original artwork is not reproduced. See
[provider score definitions](../provider-reference-results.json) for source links,
reported precision and the limits of protocol matching.

The animated demo cards use a custom mentat interface around the same recorded inputs and outputs. Reveal timing is presentation timing, not measured inference latency. Browser screenshots and scientific figures are preserved; the game redraws only its recorded partial observations.

## Additional recorded examples

| Files | Source | Terms and presentation |
| --- | --- | --- |
| `example-receipt.gif`, `example-receipt.png`, `examples/receipt.json` | [CORD](https://huggingface.co/datasets/naver-clova-ix/cord-v2), Naver Clova, test-10. | CC BY 4.0. Original receipt photo is displayed without alterations. |
| `example-count.gif`, `example-count.png`, `examples/count.json` | [CV-Bench](https://huggingface.co/datasets/nyu-visionx/CV-Bench), NYU VisionX / Cambrian, item 72; repurposes ADE20K, COCO and Omni3D. | Dataset card: Apache-2.0; underlying photograph retains its source rights. The photo is displayed without alterations. |
| `example-traffic.gif`, `example-traffic.png`, `examples/traffic.json` | [RealWorldQA](https://huggingface.co/datasets/xai-org/RealworldQA), xAI, item 14. | CC BY-ND 4.0. The unaltered full photograph is included in the demo collection; no crop, overlay or alteration of image contents. |
| `example-invoice.gif`, `example-invoice.png`, `examples/invoice.json` | [Typed Decisions](https://huggingface.co/datasets/LocalLLaMA/typed-decisions), invoice_processing_000004. | Apache-2.0. UI displays recorded supplied quantities and the model recommendation. |
| `example-tool.gif`, `example-tool.png`, `examples/tool.json` | [API-Bank](https://github.com/AlibabaResearch/DAMO-ConvAI/tree/main/api-bank), AddAgenda-AddAlarm-GetUserToken-level-2-1, via [Decision Index](https://github.com/apolinario/decision-index). | Source dialogue and candidate APIs are credited to API-Bank. This shows two tool-name selections after authentication on successive reference histories; no live API execution. |
| `example-phishing.gif`, `example-phishing.png`, `examples/phishing.json` | [Decision Index](https://github.com/apolinario/decision-index), PhishNChips, phish_0856, meeting_notes. | Displays a short email excerpt and a recorded click recommendation. The URL is rendered as inert, defanged text; no URL scan or request occurred. |

## Application-focused gallery revision

The README gallery replaces the counting, traffic-light, point-matching and email examples. Older assets remain archived, outside the gallery.

| Files | Source and scope |
| --- | --- |
| `example-contract.*`, `examples/contract.json` | [ContractNLI](https://github.com/stanfordnlp/contract-nli), Koreeda and Manning, test document 2. Three selected hypothesis classifications from a supplied agreement. Short excerpts retain the contract wording; original source-document rights remain with their owners. |
| `example-rag.*`, `examples/rag.json` | [RAGTruth](https://github.com/CodingLL/RAGTruth), case 359. A short news excerpt and generated-summary excerpt illustrate a recorded response-level hallucination decision; the date highlight is an editorial comparison. |
| `example-shopping.*`, `examples/shopping.json` | [Amazon ESCI](https://github.com/amazon-science/esci-data), test pair 392938. Query and product title reproduced as a short relevance example. Product and brand names remain with their owners. |
| `example-agent.*`, `examples/agent.json` | [Typed Decisions](https://huggingface.co/datasets/LocalLLaMA/typed-decisions), agent_trace_observability_000006; dataset Apache-2.0. Supplied benchmark trace summary and recorded recommendation, not a production agent trace. |

## Runnable example inputs

`examples/media/receipt.png`, `examples/media/chart.jpg` and `examples/media/web-01.jpg` through `web-06.jpg` are byte-identical copies of the CORD, CharXiv and Mind2Web inputs credited above, under the same respective terms. They allow local decision reruns; the screenshots retain their original reference markings. `examples/run.py` is mentat inference helper code under the repository Apache-2.0 license.

## Executed local workflows

The customer and invoice workflow fixtures retain the Typed Decisions cases credited above, adding matching invoice `invoice_processing_000002` (Apache-2.0). The publishing workflow uses RAGTruth case 359 and eleven other source news documents from its public summarization task. Source news rights remain with their owners. The model executes fresh verification; the replacement is a deterministic extract of the supplied source, not generated reasoning. `workflows/run.py` is mentat controller code under Apache-2.0. The database and traces record local test execution, not a production deployment or new benchmark evaluation.

## Case-by-case audit

The security gallery uses `security_incidents_000084`, with internally consistent evidence. Contract claims show survival obligations, scope of confidential information and the absence of a non-solicitation clause. API-Bank displays only AddAgenda and AddAlarm after reference authentication. BabyAI grids and inventory show the observation before the selected action; success is the resulting environment outcome. The example helper normalizes historical field identifiers for the native schema without changing question content.
