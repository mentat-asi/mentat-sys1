"""Load the exported PEFT decision model without training code or recipes.

This is a tensor-input inference core. Request/prompt compilation remains the
native decision contract; this file does not introduce a different prompt API.
Only eight full-attention layers are bidirectional. DeltaNet stays unchanged.
"""
from pathlib import Path
import json,os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import torch
from peft import PeftModel
from transformers import Qwen3_5ForConditionalGeneration


class PortableDecisionCore(torch.nn.Module):
    def __init__(self, base_model, adapter_directory, device='cuda:0'):
        super().__init__()
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
        torch.backends.cudnn.benchmark=False
        torch.backends.cudnn.deterministic=True
        torch.use_deterministic_algorithms(True)
        root=Path(adapter_directory)
        base=Qwen3_5ForConditionalGeneration.from_pretrained(
            base_model,local_files_only=True,dtype=torch.bfloat16,
            attn_implementation='sdpa').to(device)
        self.peft=PeftModel.from_pretrained(base,str(root),is_trainable=False,
            autocast_adapter_dtype=True)
        generator=self.peft.get_base_model()
        self.backbone=generator.model
        self.original_lm_head=generator.lm_head
        self.backbone.language_model.config.use_cache=False
        layers=self.backbone.language_model.layers
        full=[layer.self_attn for layer in layers if layer.block_type=='full_attention']
        assert len(layers)==32 and len(full)==8
        self.backbone.language_model.config.is_causal=False
        for layer in full:layer.is_causal=False
        self.backbone.rope_deltas=None
        codes=json.loads((root/'decision_readout.json').read_text())['codes']
        assert len(codes)==256
        self.register_buffer('decision_token_ids',torch.tensor(
            [r['token_id'] for r in codes],dtype=torch.long,device=device))
        assert sum('.lora_A.' in name or '.lora_B.' in name
                   for name,_ in self.peft.named_parameters())==400
        assert not any(p.requires_grad for p in self.parameters())
        self.eval()

    @torch.inference_mode()
    def forward(self, inputs):
        assert inputs['attention_mask'].ndim==2
        assert inputs.get('past_key_values') is None and not inputs.get('use_cache',False)
        with torch.autocast(device_type='cuda',dtype=torch.bfloat16):
            hidden=self.backbone(**inputs,use_cache=False).last_hidden_state
        last=inputs['attention_mask'].sum(-1)-1
        selected=hidden[torch.arange(len(last),device=last.device),last].float()
        return torch.nn.functional.linear(selected,
            self.original_lm_head.weight.index_select(0,self.decision_token_ids).float())
