"""Verify bundled measurements and aggregates without a GPU or model download."""
from pathlib import Path
import hashlib
import json
import math
import statistics

ROOT = Path(__file__).resolve().parents[1] / 'docs/benchmarks/v0.3-latency'


def read(name):
    return json.loads((ROOT / name).read_text())


def near(got, expected):
    assert math.isclose(got, expected, rel_tol=1e-12, abs_tol=1e-10), (got, expected)


def verify():
    manifest = read('manifest.json')
    for name, digest in manifest['files'].items():
        assert Path(name).name == name
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest, name
    summary = read('summary.json')
    assert summary['status'] == 'complete'
    variants = {'native': 'native', 'batched': 'previous_accelerated', 'prefix-fused': 'prefix_fused'}
    identity = summary['model_identity']
    for filename, label in variants.items():
        raw = read(f'raw-{filename}.json')
        assert raw['status'] == 'complete'
        assert raw['adapter_sha256'] == identity['adapter_sha256']
        assert raw['readout_sha256'] == identity['readout_sha256']
        assert len(raw['trials']) == 6 and raw['warmup_per_trial'] == 20
        for scenario in ['one_field', 'three_fields']:
            trials = [t for t in raw['trials'] if t['scenario'] == scenario]
            assert sorted(t['trial'] for t in trials) == [0, 1, 2]
            assert all(t['n'] == 200 for t in trials)
            for source_key, kind in [('wall_ms', 'HTTP'), ('server_ms', 'server')]:
                values = sorted(v for trial in trials for v in trial[source_key])
                assert len(values) == 600 and all(math.isfinite(v) and v > 0 for v in values)
                expected = summary['latency'][scenario][label][kind]
                near(statistics.mean(values), expected['mean_ms'])
                near(values[300], expected['p50_ms'])
                near(values[570], expected['p95_ms'])
            if label == 'prefix_fused':
                before = summary['latency'][scenario]['previous_accelerated']['HTTP']['mean_ms']
                after = summary['latency'][scenario][label]['HTTP']['mean_ms']
                near(1 - after / before, summary['latency'][scenario]['HTTP_latency_reduction_fraction'])
    quality = summary['quality']
    assert len(quality['suite_scores']) == 7 and quality['fields'] == 12351
    assert sum(x['fields'] for x in quality['comparisons'].values()) == 12351
    assert sum(x['winner_flips'] for x in quality['comparisons'].values()) == quality['winner_flips'] == 30
    near(1 - 30 / 12351, quality['winner_agreement'])
    near(100 * statistics.mean(x['accuracy'] for x in quality['suite_scores'].values()), quality['ID_average'])
    for name, score in quality['suite_scores'].items():
        assert score['missing_rows'] == score['invalid'] == 0
        assert score['total'] == quality['native_suite_scores'][name]['total']
    prefix = read('prefix-validation.json')
    assert prefix['eligible_cases'] == len(prefix['cases']) == 78
    assert sum(x['fields'] for x in prefix['cases']) == prefix['fields'] == 234
    assert prefix['winner_flips'] == 0 and prefix['request_isolation_A_B_A_exact']
    print('Verified: 3 runtime variants;600 timed requests per scenario each;12,351 ID fields;234 prefix-check fields.')


if __name__ == '__main__':
    verify()
