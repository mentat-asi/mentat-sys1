"""Request-local prefix sharing; fresh attention and recurrent state per call."""

import math

from transformers import DynamicCache

from .batched import Optimized


class PrefixOptimized(Optimized):
    def __init__(self, *args, share_prefix=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.share_prefix = share_prefix
        self.active_cut = 0
        self.max_graphs = 8
        self.max_graph_tokens = 4096

    def cut(self, items):
        if not self.share_prefix or len(items) < 2:
            return 0
        common = 0
        for col in zip(*(x["input_ids"].tolist() for x in items), strict=False):
            if len(set(col)) != 1:
                break
            common += 1
        cut = common // 64 * 64
        return cut if cut >= 128 and min(x["tokens"] for x in items) - cut >= 2 else 0

    def forward(self, ids, last):
        if self.active_cut == 0:
            return super().forward(ids, last)
        t = self.torch
        cut = self.active_cut
        b = ids.shape[0]
        cache = DynamicCache(config=self.body.config)
        self.body(
            input_ids=ids[:1, :cut].contiguous(),
            use_cache=True,
            past_key_values=cache,
            return_dict=True,
        )
        cache.reorder_cache(t.zeros(b, dtype=t.long, device=ids.device))
        hidden = self.body(
            input_ids=ids[:, cut:].contiguous(),
            use_cache=True,
            past_key_values=cache,
            return_dict=True,
        ).last_hidden_state
        selected = hidden[t.arange(b, device=ids.device), last - cut].float()
        return t.stack([self.engine.readout(selected[i]) for i in range(b)])

    def capture(self, key):
        b, n, cut = key
        self.active_cut = cut
        super().capture((b, n))
        self.graphs[key] = self.graphs.pop((b, n))

    def warm(self, payloads):
        if self.mode != "graph":
            return
        for payload in payloads:
            _, items = self.prepare(payload)
            n = self.graph_bucket * math.ceil(
                max(x["tokens"] for x in items) / self.graph_bucket
            )
            key = (len(items), n, self.cut(items))
            if (
                key not in self.graphs
                and len(self.graphs) < self.max_graphs
                and key[0] * key[1] <= self.max_graph_tokens
            ):
                self.capture(key)

    def predict(self, payload):
        t = self.torch
        req, items = self.prepare(payload)
        cut = self.cut(items)
        shape = (
            len(items),
            self.graph_bucket
            * math.ceil(max(x["tokens"] for x in items) / self.graph_bucket),
            cut,
        )
        with t.inference_mode():
            if (
                self.mode == "graph"
                and shape not in self.graphs
                and len(self.graphs) < self.max_graphs
                and shape[0] * shape[1] <= self.max_graph_tokens
            ):
                self.capture(shape)
            if self.mode == "graph" and shape in self.graphs:
                graph, static_ids, static_last, out = self.graphs[shape]
                ids, last = self.tensors(items, shape[1])
                static_ids.copy_(ids)
                static_last.copy_(last)
                graph.replay()
                values = out.float().cpu().tolist()
            else:
                self.active_cut = cut
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
            "language_model_passes": 2 if cut else 1,
            "generated_answer_tokens": 0,
            "request_local_shared_prefix_tokens": cut,
            "computed_tokens_before_padding": sum(x["tokens"] for x in items)
            - (len(items) - 1) * cut,
        }
        return result
