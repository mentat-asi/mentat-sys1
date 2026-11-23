"""Optional merged/fused CUDA inference with an independent native fallback.

Two resident cores require more VRAM than AcceleratedDecisionModel. No weights
are changed on disk. Qualification tests the first content for each shape;
it is not a guarantee of exact equivalence on later contents of that shape.
"""
import copy
import types
import torch
from mentat_spice.accelerated import AcceleratedDecisionModel, ExactShapeGraphs
from mentat_spice.scoring import calibrated_probabilities


def install_fusion(core):
    import transformers.models.qwen3_5.modeling_qwen3_5 as modeling
    options = {'emulate_precision_casts': True}
    compiled = {cls: torch.compile(cls.forward, fullgraph=True, dynamic=True, options=options)
                for cls in (modeling.Qwen3_5RMSNorm, modeling.Qwen3_5RMSNormGated)}

    @torch.compile(fullgraph=True, dynamic=True, options=options)
    def gate_up(gate, up):
        return torch.nn.functional.silu(gate) * up

    def mlp(self, x):
        return self.down_proj(gate_up(self.gate_proj(x), self.up_proj(x)))

    counts = {}
    for module in core.modules():
        cls = type(module)
        if cls in compiled:
            module.forward = types.MethodType(compiled[cls], module)
        elif cls is modeling.Qwen3_5MLP:
            module.forward = types.MethodType(mlp, module)
        else:
            continue
        counts[cls.__name__] = counts.get(cls.__name__, 0) + 1
    return counts


def install_fast(core):
    install_fusion(core)
    import transformers.models.qwen3_5.modeling_qwen3_5 as modeling
    from fla.modules.conv import causal_conv1d
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule
    def conv(hidden_states,weight,bias=None,activation=None,**kw):
        result,_=causal_conv1d(hidden_states.transpose(1,2),weight=weight,bias=bias,
            activation=activation,backend='triton',output_final_state=False)
        return result.transpose(1,2)
    cls=modeling.Qwen3_5GatedDeltaNet
    namespace=dict(cls.forward.__globals__)
    namespace.update(causal_conv1d_fn=conv,torch_chunk_gated_delta_rule=chunk_gated_delta_rule)
    forward=types.FunctionType(cls.forward.__code__,namespace,cls.forward.__name__,cls.forward.__defaults__,cls.forward.__closure__)
    forward.__kwdefaults__=cls.forward.__kwdefaults__
    for module in core.modules():
        if type(module) is cls:module.forward=types.MethodType(forward,module)


class GuardedGraphs(ExactShapeGraphs):
    def __init__(self,native,fast):
        super().__init__(fast)
        self.native=native
        self.safe_graphs=ExactShapeGraphs(native)
        self.confidence_fallbacks=0
        self.ambiguous_eager_fallbacks=0
        self.fast_outputs=0

    def fallback(self, inputs):
        self.safe_graphs.candidate_count = self.candidate_count
        replays=self.safe_graphs.replays
        output=self.safe_graphs(inputs)
        if self.safe_graphs.replays>replays:
            probabilities=calibrated_probabilities(output[0,:self.candidate_count].float().cpu().tolist(),temperature=1.)
            known=sorted(probabilities[:-1],reverse=True)
            margin=known[0]-known[1] if len(known)>1 else known[0]
            if margin < .20 or probabilities[-1] > .01:
                self.ambiguous_eager_fallbacks+=1
                return self.native(inputs)
        return output

    def __call__(self,inputs):
        tokens=inputs['input_ids'].shape[-1]
        ordinary=(set(inputs)<={'input_ids','attention_mask','mm_token_type_ids'} and tokens<=self.max_tokens
                  and bool((inputs['attention_mask']==1).all())
                  and ('mm_token_type_ids' not in inputs or not bool(inputs['mm_token_type_ids'].any())))
        if not ordinary or self.candidate_count is None:return self.native(inputs)
        key=(self.candidate_count,tuple((k,tuple(v.shape),str(v.dtype)) for k,v in sorted(inputs.items())))
        if key in self.rejected:return self.fallback(inputs)
        if key not in self.graphs:
            if len(self.graphs)>=self.max_graphs:return self.fallback(inputs)
            reference=self.native(inputs).clone()
            try:
                static={k:v.clone() for k,v in inputs.items()}
                stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream),torch.inference_mode():
                    for _ in range(3):self.network(static)
                torch.cuda.current_stream().wait_stream(stream);torch.cuda.synchronize()
                graph=torch.cuda.CUDAGraph()
                with torch.inference_mode(),torch.cuda.graph(graph):output=self.network(static)
                graph.replay();proof=self.compare(reference,output,self.candidate_count)
                self.qualifications[str(key)]=proof
                if not proof['passed']:
                    self.rejected[key]=proof
                    return self.fallback(inputs)
                self.graphs[key]=(graph,static,output)
            except Exception as error:
                self.failures.append({'shape':str(key),'error':str(error)});self.rejected[key]={'error':str(error)};return reference
        graph,static,output=self.graphs[key]
        for k,v in inputs.items():static[k].copy_(v)
        graph.replay();self.replays+=1
        probabilities=calibrated_probabilities(output[0,:self.candidate_count].float().cpu().tolist(),temperature=1.)
        # Preserve the unmerged path for ambiguous distributions on every call,
        # rather than relying only on the first content seen for this shape.
        if max(probabilities[:-1]) < .90 or probabilities[-1] > .01:
            self.confidence_fallbacks+=1
            return self.fallback(inputs)
        self.fast_outputs+=1
        return output


class TurboDecisionModel(AcceleratedDecisionModel):
    def __init__(self, base_model, adapter_directory, *, device='cuda:0', fast_prepare=True):
        try:
            import fla
            import triton.knobs
        except ImportError as error:
            raise RuntimeError('Install this package with its accelerate extra for TurboDecisionModel') from error
        if not triton.knobs.language.default_fp_fusion:
            raise ValueError('TurboDecisionModel is qualified with default Triton FP fusion enabled')
        super().__init__(base_model, adapter_directory, device=device,
                         fast_prepare=fast_prepare, cuda_graphs=False)
        native=self.engine.network
        fast=copy.deepcopy(native)
        if {p.data_ptr() for p in native.parameters()} & {p.data_ptr() for p in fast.parameters()}:
            raise RuntimeError('Independent native and fast cores are required')
        fast.peft=fast.peft.merge_and_unload(safe_merge=True)
        install_fast(fast)
        self.acceleration.update(resident_core_count=2, fast_core_weights_merged=True,
                                 native_fallback_unmerged=True, pointwise_fusion=True,
                                 convolution_override='FLA Triton', image_runtime='native eager',
                                 minimum_fast_raw_known_probability=.90,
                                 maximum_fast_unknown_probability=.01,
                                 minimum_unmerged_graph_probability_margin=.20)
        self.graph_network=GuardedGraphs(native,fast);self.engine.network=self.graph_network
        prepare_original=self.engine._prepare
        def prepare(images,prompt,labels):
            result=prepare_original(images,prompt,labels);self.graph_network.candidate_count=len(labels);return result
        self.engine._prepare=prepare
