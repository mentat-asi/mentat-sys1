"""Process-local pointwise fusion; preserve explicit precision casts."""


def install():
    import torch
    import torch._inductor.config as config
    import transformers.models.qwen3_5.modeling_qwen3_5 as modeling

    config.emulate_precision_casts = True
    modeling.Qwen3_5RMSNorm.forward = torch.compile(
        modeling.Qwen3_5RMSNorm.forward, fullgraph=True, dynamic=True
    )
    modeling.Qwen3_5RMSNormGated.forward = torch.compile(
        modeling.Qwen3_5RMSNormGated.forward, fullgraph=True, dynamic=True
    )

    @torch.compile(fullgraph=True, dynamic=True)
    def gate_up(gate, up):
        return torch.nn.functional.silu(gate) * up

    def mlp(self, x):
        return self.down_proj(gate_up(self.gate_proj(x), self.up_proj(x)))

    modeling.Qwen3_5MLP.forward = mlp
