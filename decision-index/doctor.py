"""Check dependencies and source compatibility without loading model weights."""
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path
import sys

assert sys.version_info[:2] == (3, 12), 'Use Python 3.12'
expected = {'mentat-sys1':'0.4.0', 'torch':'2.8.0', 'transformers':'5.17.0',
            'peft':'0.21.0', 'triton':'3.4.0', 'flash-linear-attention':'0.5.2'}
versions = {name:metadata.version(name) for name in expected}
assert {name:version.split('+')[0] for name,version in versions.items()} == expected, (versions, expected)
import mentat_spice.contracts as contracts
digest = hashlib.sha256(Path(contracts.__file__).read_bytes()).hexdigest()
assert digest == 'a13d1f877c6c7c1913df2f07fd241e3183d12c39732fec80e1c29932ca16bdc8', 'Run apply_compatibility.py first'
import torch
assert torch.cuda.is_available(), 'CUDA GPU required for Turbo'
import fla
import triton.knobs
assert triton.knobs.language.default_fp_fusion, 'Triton FP fusion must be enabled'
from spice_di import SpiceEngine
print(json.dumps({'versions':versions, 'cuda':torch.version.cuda,
                  'gpu':torch.cuda.get_device_name(0), 'contracts_sha256':digest,
                  'ready_for_model_smoke':True}, indent=2))
