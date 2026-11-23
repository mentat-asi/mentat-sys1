"""Opt-in fast text preparation and exact-shape CUDA Graphs for mentat-Spice.

Use one dedicated worker; model.predict serializes calls. Resident LoRA weights
can be precast to their existing BF16 autocast compute dtype. Adapter files stay
unchanged. No merging, padding, prefix reuse or output caching is performed.
"""
from pathlib import Path
import math
import torch
from mentat_spice import DecisionModel
from mentat_spice.scoring import calibrated_probabilities


class FastPreparation:
    def __init__(self, engine):
        self.engine = engine
        self.original_prepare = engine._prepare
        self.original_labels = engine.labels
        tokenizer = engine.processor.tokenizer
        ids = tokenizer.encode('</think>', add_special_tokens=False)
        special = tokenizer.added_tokens_decoder.get(ids[0]) if len(ids) == 1 else None
        if (special is None or special.content != '</think>' or special.normalized
                or special.lstrip or special.rstrip
                or str(tokenizer.backend_tokenizer.normalizer) not in ('None','NFC()')):
            raise ValueError('Unqualified tokenizer boundary')
        engine._verify_codebook(0)
        self.suffix = tokenizer.encode('</think>\n\n', add_special_tokens=False)
        _, inputs, _ = self.original_prepare([], '', [engine.codebook[0][0]])
        if not {'input_ids','attention_mask'} <= inputs.keys() <= {'input_ids','attention_mask','mm_token_type_ids'}:
            raise ValueError('Unqualified processor tensor contract')
        self.modality = 'mm_token_type_ids' in inputs

    def labels(self, count, n_images):
        if n_images:
            return self.original_labels(count, n_images)
        if not 1 <= count <= len(self.engine.codebook):
            raise ValueError('Candidate count exceeds codebook')
        return [code for code, _ in self.engine.codebook[:count]]

    def prepare(self, images, prompt, labels):
        if images:
            return self.original_prepare(images, prompt, labels)
        if list(labels) != [code for code, _ in self.engine.codebook[:len(labels)]]:
            raise ValueError('Candidate codebook mismatch')
        rendered = self.engine._render(prompt, 0)
        tokenizer = self.engine.processor.tokenizer
        ids = tokenizer.encode(rendered, add_special_tokens=False)
        if not rendered.endswith('</think>\n\n') or ids[-len(self.suffix):] != self.suffix:
            raise ValueError('Decision-position boundary mismatch')
        if len(ids) > self.engine.max_length:
            raise ValueError('processed request exceeds context limit')
        inputs = {'input_ids':torch.tensor([ids],dtype=torch.long),
                  'attention_mask':torch.ones((1,len(ids)),dtype=torch.long)}
        if self.modality:
            inputs['mm_token_type_ids'] = torch.tensor(
                self.engine.processor.create_mm_token_type_ids([ids]), dtype=torch.long)
        self.engine.probe_meta = {'image_pixels':[], 'image_tokens':0,'image_tokens_each':[], 'input_tokens':len(ids)}
        return rendered, inputs, [tid for _, tid in self.engine.codebook[:len(labels)]]

    def install(self):
        self.engine.labels = self.labels
        self.engine._prepare = self.prepare


class ExactShapeGraphs:
    def __init__(self, network, max_tokens=2048, max_graphs=8):
        self.network = network
        self.max_tokens, self.max_graphs = max_tokens, max_graphs
        self.graphs = {}
        self.failures = []
        self.replays = 0
        self.candidate_count = None
        self.rejected = {}
        self.qualifications = {}

    @staticmethod
    def compare(reference, candidate, count):
        """Qualify the first actual input for a shape, at native temperature.

        This is a first-use check, not a numerical bound on future contents.
        Include UNKNOWN and confidence, since matching choices alone is weaker.
        """
        probabilities = [calibrated_probabilities(x[0, :count].float().cpu().tolist(), temperature=1.)
                         for x in (reference, candidate)]
        known = []
        confidences = []
        for values in probabilities:
            mass = math.fsum(values[:-1])
            n = len(values) - 1
            normalized = [v / mass for v in values[:-1]] if mass > 0 else [1. / n] * n
            known.append(normalized)
            certainty = max(0., (n * max(normalized) - 1) / (n - 1)) if n > 1 else 1.
            confidences.append(certainty * (1 - values[-1]))
        a, b = probabilities
        same_raw = max(range(count), key=a.__getitem__) == max(range(count), key=b.__getitem__)
        same_known = (max(range(count - 1), key=known[0].__getitem__)
                      == max(range(count - 1), key=known[1].__getitem__))
        same_boolean = count != 3 or ((a[0] + .5 * a[-1] >= .5) == (b[0] + .5 * b[-1] >= .5))
        delta = max([abs(x-y) for x,y in zip(known[0], known[1])]
                    + [abs(a[-1]-b[-1]), abs(confidences[0]-confidences[1])])
        return {'passed': same_raw and same_known and same_boolean and delta <= .005,
                'max_probability_or_confidence_delta': delta,
                'same_raw_mode': same_raw, 'same_known_mode': same_known,
                'same_boolean_threshold': same_boolean}

    def __call__(self, inputs):
        tokens = inputs['input_ids'].shape[-1]
        # The graph is for ordinary, unpadded text. Images and literal media
        # markers continue through the native eager network.
        ordinary = (set(inputs) <= {'input_ids','attention_mask','mm_token_type_ids'}
                    and tokens <= self.max_tokens
                    and bool((inputs['attention_mask'] == 1).all())
                    and not bool(inputs.get('mm_token_type_ids', torch.zeros((),device=inputs['input_ids'].device)).any()))
        if not ordinary or self.candidate_count is None:
            return self.network(inputs)
        key = (self.candidate_count, tuple((k,tuple(v.shape),str(v.dtype)) for k,v in sorted(inputs.items())))
        if key in self.rejected:
            return self.network(inputs)
        if key not in self.graphs:
            if len(self.graphs) >= self.max_graphs:
                return self.network(inputs)
            with torch.inference_mode():
                reference = self.network(inputs).clone()
            try:
                static = {k:v.clone() for k,v in inputs.items()}
                stream = torch.cuda.Stream(); stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream),torch.inference_mode():
                    for _ in range(3): self.network(static)
                torch.cuda.current_stream().wait_stream(stream); torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.inference_mode(), torch.cuda.graph(graph): output = self.network(static)
                graph.replay()
                proof = self.compare(reference, output, self.candidate_count)
                self.qualifications[str(key)] = proof
                if not proof['passed']:
                    self.rejected[key] = proof
                    return reference
                self.graphs[key] = (graph,static,output)
            except Exception as error:
                self.failures.append({'shape':str(key),'error':str(error)})
                self.rejected[key] = {'error':str(error)}
                return reference
        graph,static,output = self.graphs[key]
        for k,v in inputs.items():static[k].copy_(v)
        graph.replay(); self.replays += 1
        return output


class AcceleratedDecisionModel(DecisionModel):
    def __init__(self, base_model, adapter_directory, *, device='cuda:0',
                 fast_prepare=True, fast_kernels=False, cuda_graphs=True,
                 cache_adapter_casts=True):
        super().__init__(base_model,adapter_directory,device=device)
        self.fast_preparation = FastPreparation(self.engine) if fast_prepare else None
        if self.fast_preparation:self.fast_preparation.install()
        if fast_kernels:
            raise ValueError('FLA replacement is not qualified for this release; retain native operators')
        count = 0
        if cache_adapter_casts:
            for name, module in self.engine.network.named_modules():
                if ('.lora_A.' in name or '.lora_B.' in name) and isinstance(module, torch.nn.Linear):
                    module.to(torch.bfloat16)
                    count += 1
            if count != 400:
                raise ValueError('Unqualified adapter module layout')
        self.acceleration = {'fast_prepare':fast_prepare,'fast_kernels':fast_kernels,'cuda_graphs':cuda_graphs,
                             'cache_adapter_casts':cache_adapter_casts,
                             'resident_adapter_dtype':'bfloat16' if cache_adapter_casts else 'float32',
                             'converted_adapter_linear_modules':count,
                             'weight_dtype_changed':bool(count),'weights_merged':False,'prefix_reuse':False}
        self.graph_network = ExactShapeGraphs(self.engine.network) if cuda_graphs else None
        if self.graph_network:
            self.engine.network = self.graph_network
            original_prepare = self.engine._prepare
            def prepare(images, prompt, labels):
                result = original_prepare(images, prompt, labels)
                self.graph_network.candidate_count = len(labels)
                return result
            self.engine._prepare = prepare
