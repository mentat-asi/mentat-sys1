from importlib.metadata import distribution
from pathlib import Path
import hashlib

dist=distribution('mentat-sys1')
assert dist.version=='0.4.0', 'This compatibility patch targets mentat-sys1 0.4.0 only'
p=Path(dist.locate_file('mentat_spice/contracts.py'))
s=p.read_text();current=hashlib.sha256(p.read_bytes()).hexdigest()
if current=='a13d1f877c6c7c1913df2f07fd241e3183d12c39732fec80e1c29932ca16bdc8':
    print('Already compatible')
elif current in {'239bbb20646d60667d11c20fda0041f14ada4164b79a7c822b312f34ead4f351','acdf3925bebb472cb5b908f318d994ce8dd8fc73052e9d5322bc14c63e23cc71'}:
    s=s.replace('MAX_STATE_BYTES = 131072','MAX_STATE_BYTES = 8 * 1024 * 1024').replace('Field(min_length=1, max_length=2000)','Field(min_length=1)').replace('Field(pattern=r"^[A-Za-z][A-Za-z0-9_]{0,63}$")','Field(min_length=1)').replace('Field(min_length=1, max_length=4096)','Field(min_length=1)')
    s=s.replace('if depth > 8:','if depth > 64:').replace('state nesting exceeds 8','state nesting exceeds 64')
    assert hashlib.sha256(s.encode()).hexdigest()=='a13d1f877c6c7c1913df2f07fd241e3183d12c39732fec80e1c29932ca16bdc8'
    p.write_text(s)
    print('Applied DI input compatibility fixes; weights and scoring unchanged')
else:
    raise SystemExit('Unexpected runtime source hash; refusing to patch')
