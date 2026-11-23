"""Inference-only native text/image preparation and frozen LM projection."""
from __future__ import annotations
from pathlib import Path
from typing import Any
from time import perf_counter
import json, math
from .scoring import verified_label_ids

class NativeEngine:
    def __init__(self, base_model, adapter_directory, device="cuda:0"):
        import torch
        from transformers import AutoProcessor
        from .core import PortableDecisionCore
        started = perf_counter()
        root = Path(adapter_directory)
        cfg = json.loads((root / "inference_config.json").read_text())
        expected = {"bidirectional_full_attention_layers": 8, "linear_attention_layers_causal": 24,
                    "original_lm_projection": True, "additional_output_head": False,
                    "temperature": 1.0, "temperature_fitted": False,
                    "include_unknown_candidate": True, "max_options_excluding_unknown": 255}
        if any(cfg.get(k) != v for k, v in expected.items()):
            raise ValueError("Unsupported architecture/readout configuration")
        self.processor = AutoProcessor.from_pretrained(base_model, local_files_only=True)
        self.processor.image_processor.size = {"shortest_edge":65536,"longest_edge":1638400}
        self.codebook = tuple((x["code"], x["token_id"]) for x in json.loads((root / "decision_readout.json").read_text())["codes"])
        if len(self.codebook) != 256 or len({x[0] for x in self.codebook}) != 256 or len({x[1] for x in self.codebook}) != 256:
            raise ValueError("Expected 256 unique decision code/token bindings")
        self.device = device
        self.training_max_length = int(cfg["maximum_training_context"])
        if self.training_max_length != 32768:
            raise ValueError("Unqualified training context configuration")
        self.max_length = cfg.get("maximum_inference_context", self.training_max_length)
        if type(self.max_length) is not int or self.max_length not in {32768, 40960}:
            raise ValueError("Unqualified inference context configuration")
        self._verify_codebook(0)
        self.network = PortableDecisionCore(base_model, adapter_directory, device=device)
        self.probe_meta = {}
        self.load_seconds = perf_counter() - started

    def _render(self, prompt: str, n_images: int) -> str:
            messages = [
                {
                    "role": "user",
                    "content": [
                        *({"type": "image"} for _ in range(n_images)),
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            try:
                rendered = self.processor.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=False,
                    enable_thinking=False,
                )
            except ValueError as error:
                if "does not have a chat template" not in str(error):
                    raise
                rendered = self.processor.tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=False,
                    enable_thinking=False,
                )
            if not isinstance(rendered, str) or not rendered.endswith(
                "<think>\n\n</think>\n\n"
            ):
                raise ValueError("unexpected Qwen non-thinking template boundary")
            return rendered

    def _verify_codebook(self, n_images: int) -> None:
            rendered = self._render("", n_images)
            labels = [code for code, _ in self.codebook]
            observed = verified_label_ids(self.processor.tokenizer, rendered, labels)
            expected = [token_id for _, token_id in self.codebook]
            if observed != expected:
                raise ValueError(
                    "decision readout code/token binding does not match this tokenizer"
                )

    def labels(self, count: int, n_images: int) -> list[str]:
            if not 1 <= count <= len(self.codebook):
                raise ValueError(
                    f"{count} candidates exceed the {len(self.codebook)}-code readout"
                )
            labels = [code for code, _ in self.codebook[:count]]
            rendered = self._render("", n_images)
            observed = verified_label_ids(self.processor.tokenizer, rendered, labels)
            expected = [token_id for _, token_id in self.codebook[:count]]
            if observed != expected:
                raise ValueError(
                    "decision readout code/token binding does not match this tokenizer"
                )
            return labels

    def _prepare(engine,images,prompt,labels):
        # Same released compiler/code binding and context guards, with capacity raised
        # from two to four images only in this private experimental instance.
        if len(images)>4:raise ValueError('Experimental DI adapter supports at most four images')
        rendered=engine._render(prompt,len(images))
        token_ids=verified_label_ids(engine.processor.tokenizer,rendered,labels)
        expected=[token_id for _,token_id in engine.codebook[:len(labels)]]
        if token_ids!=expected:raise ValueError('decision readout code/token binding mismatch')
        inputs=engine.processor(text=[rendered],images=images or None,return_tensors='pt')
        input_ids=inputs['input_ids']
        if int(input_ids.shape[-1])>engine.max_length:raise ValueError('processed request exceeds context limit')
        suffix=engine.processor.tokenizer.encode('</think>\n\n',add_special_tokens=False)
        if input_ids[0,-len(suffix):].tolist()!=suffix:raise ValueError('decision-position suffix mismatch')
        image_tokens=[]
        if images:
            grids=inputs.get('image_grid_thw')
            if grids is None or len(grids)!=len(images):raise ValueError('invalid image grid')
            image_tokens=[int(g[0]*g[1]*g[2]//4) for g in grids]
            if any(n>2048 for n in image_tokens):raise ValueError('image exceeds released 2048-token guard')
        engine.probe_meta={'image_pixels':[im.width*im.height for im in images],'image_tokens':sum(image_tokens),'image_tokens_each':image_tokens,'input_tokens':int(input_ids.shape[-1])}
        return rendered,dict(inputs),token_ids

    def score_prompt(self, images, prompt, labels):
        import torch
        started = perf_counter()
        rendered, inputs, token_ids = self._prepare(images, prompt, labels)
        preprocess_seconds = perf_counter() - started
        moved = {k:v.to(self.device) for k,v in inputs.items()}
        forward_started = perf_counter()
        with torch.inference_mode():
            logits = self.network(moved)[0, :len(labels)]
        values = [float(x) for x in logits.detach().cpu().tolist()]
        if not all(math.isfinite(x) for x in values):
            raise ValueError("Nonfinite candidate logits")
        return values, {"preprocess_seconds":preprocess_seconds,
                        "forward_seconds":perf_counter()-forward_started,
                        "input_tokens":int(inputs["input_ids"].shape[-1]),
                        "choice_token_ids":token_ids,
                        "readout_indices":list(range(len(labels))),
                        "logit_precision":"float32_original_lm_projection"}
