"""Batched FP32 decision readout with optional FLA kernels and CUDA Graphs."""

import math


class Optimized:
    def __init__(
        self,
        native,
        mode="batch",
        lora_bf16=False,
        fast_kernels=False,
        fast_prepare=False,
        graph_bucket=4,
    ):
        from importlib import import_module

        self.scoring = import_module("mentat_jev_v1.inference.scoring")
        self.native = native
        self.engine = native.engine
        self.torch = native.engine._torch
        self.mode = mode
        self.graphs = {}
        self.failures = []
        self.body = self.engine.model.get_base_model().model.language_model
        self.dtype_change = lora_bf16
        self.fast_kernels = fast_kernels
        self.fast_prepare = fast_prepare
        self.graph_bucket = graph_bucket
        if fast_prepare:
            tokenizer = self.engine.processor.tokenizer
            boundary = tokenizer.encode("</think>", add_special_tokens=False)
            added = (
                tokenizer.added_tokens_decoder.get(boundary[0])
                if len(boundary) == 1
                else None
            )
            if (
                added is None
                or added.content != "</think>"
                or added.normalized
                or added.lstrip
                or added.rstrip
                or str(tokenizer.backend_tokenizer.normalizer) not in ("None", "NFC()")
            ):
                raise ValueError(
                    "Fast preparation needs the audited "
                    "non-normalized special-token boundary"
                )
            self.suffix = tokenizer.encode("</think>\n\n", add_special_tokens=False)
        if lora_bf16:
            for name, module in self.engine.model.named_modules():
                if (".lora_A." in name or ".lora_B." in name) and isinstance(
                    module, self.torch.nn.Linear
                ):
                    module.to(self.torch.bfloat16)
        if fast_kernels:
            import transformers.models.qwen3_5.modeling_qwen3_5 as modeling
            from fla.modules.conv import causal_conv1d
            from fla.ops.gated_delta_rule import chunk_gated_delta_rule

            def conv(hidden_states, weight, bias=None, activation=None, **kw):
                out, _ = causal_conv1d(
                    hidden_states.transpose(1, 2),
                    weight=weight,
                    bias=bias,
                    activation=activation,
                    backend="triton",
                    output_final_state=False,
                )
                return out.transpose(1, 2)

            modeling.causal_conv1d_fn = conv
            modeling.torch_chunk_gated_delta_rule = chunk_gated_delta_rule

    def prepare(self, payload):
        body = dict(payload, model="mentat-jev-v1")
        if body.get("images"):
            raise ValueError("This benchmark tests the released text-only interface")
        req = self.scoring.to_decision_request(
            body, expected_model="mentat-jev-v1", request_id="v03-latency"
        )
        items = []
        for field in req.fields:
            header, choices, texts = self.scoring.compile_question(field, req.state)
            if len(choices) > len(self.engine.codebook):
                raise ValueError("Candidate capacity exceeded")
            labels = (
                [code for code, _ in self.engine.codebook[: len(choices)]]
                if self.fast_prepare
                else self.engine.labels(len(choices), 0)
            )
            prompt = header + "\n".join(
                f"{code}: {text}" for code, text in zip(labels, texts, strict=True)
            )
            if self.fast_prepare:
                rendered = self.engine._render(prompt, 0)
                ids = self.engine.processor.tokenizer.encode(
                    rendered, add_special_tokens=False
                )
                if (
                    not rendered.endswith("</think>\n\n")
                    or ids[-len(self.suffix) :] != self.suffix
                    or len(ids) > self.engine.max_length
                ):
                    raise ValueError("Fast preparation boundary/context mismatch")
                tids = [tid for _, tid in self.engine.codebook[: len(labels)]]
                inputs = {"input_ids": self.torch.tensor([ids], dtype=self.torch.long)}
            else:
                rendered, inputs, tids = self.engine._prepare([], prompt, labels)
            items.append(
                {
                    "field": field,
                    "choices": choices,
                    "labels": labels,
                    "input_ids": inputs["input_ids"][0],
                    "tokens": inputs["input_ids"].shape[-1],
                    "tids": tids,
                }
            )
        return req, items

    def tensors(self, items, padded=None):
        t = self.torch
        n = padded or max(x["tokens"] for x in items)
        ids = t.full(
            (len(items), n), self.engine.processor.tokenizer.pad_token_id, dtype=t.long
        )
        last = t.tensor([x["tokens"] - 1 for x in items], dtype=t.long)
        for i, x in enumerate(items):
            ids[i, : x["tokens"]] = x["input_ids"]
        return ids.to(self.engine.device), last.to(self.engine.device)

    def forward(self, ids, last):
        t = self.torch
        hidden = self.body(
            input_ids=ids, use_cache=False, return_dict=True
        ).last_hidden_state
        # Retain the original per-field FP32 GEMV readout instead of changing
        # its matrix shape, calibration, candidate order, or unknown outcome.
        selected = hidden[t.arange(ids.shape[0], device=ids.device), last].float()
        return t.stack([self.engine.readout(selected[i]) for i in range(ids.shape[0])])

    def capture(self, shape):
        t = self.torch
        b, n = shape
        ids = t.full(
            shape,
            self.engine.processor.tokenizer.pad_token_id,
            dtype=t.long,
            device=self.engine.device,
        )
        last = t.full((b,), n - 1, dtype=t.long, device=self.engine.device)
        stream = t.cuda.Stream()
        stream.wait_stream(t.cuda.current_stream())
        with t.cuda.stream(stream), t.inference_mode():
            for _ in range(3):
                self.forward(ids, last)
        t.cuda.current_stream().wait_stream(stream)
        t.cuda.synchronize()
        graph = t.cuda.CUDAGraph()
        with t.inference_mode(), t.cuda.graph(graph):
            try:
                output = self.forward(ids, last)
            except Exception:
                import traceback

                traceback.print_exc()
                raise
        self.graphs[shape] = (graph, ids, last, output)

    def warm(self, payloads):
        if self.mode != "graph":
            return
        for payload in payloads:
            _, items = self.prepare(payload)
            n = self.graph_bucket * math.ceil(
                max(x["tokens"] for x in items) / self.graph_bucket
            )
            shape = (len(items), n)
            if shape in self.graphs:
                continue
            try:
                self.capture(shape)
            except Exception as exc:
                self.failures.append({"shape": shape, "error": str(exc)})

    def predict(self, payload):
        t = self.torch
        req, items = self.prepare(payload)
        shape = (
            len(items),
            self.graph_bucket
            * math.ceil(max(x["tokens"] for x in items) / self.graph_bucket),
        )
        with t.inference_mode():
            if self.mode == "graph" and shape in self.graphs:
                graph, static_ids, static_last, out = self.graphs[shape]
                ids, last = self.tensors(items, shape[1])
                static_ids.copy_(ids)
                static_last.copy_(last)
                graph.replay()
                values = out.float().cpu().tolist()
            else:
                ids, last = self.tensors(items)
                values = self.forward(ids, last).float().cpu().tolist()
        results = []
        for x, z in zip(items, values, strict=True):
            kind = {"choice": "choice", "boolean": "noul", "ordinal": "score"}[
                x["field"].type
            ]
            results.append(
                self.scoring.result_from_logits(
                    x["choices"],
                    z[: len(x["labels"])],
                    token_ids=x["tids"],
                    temperature=self.native.backend.temperature_by_type[kind],
                    calibration_version=self.native.backend.calibration_version,
                )
            )
        result = self.scoring.to_systemone_response(
            req, results, model="mentat-sys1-v0.3"
        )
        result["usage"] = {
            "input_tokens": sum(x["tokens"] for x in items),
            "maximum_field_tokens": max(x["tokens"] for x in items),
            "language_model_passes": 1,
            "generated_answer_tokens": 0,
        }
        return result
