"""Measure warm, sequential latency of this package's native decision runtime.

This measures Python calls, including input preparation and image loading. It
excludes model loading, HTTP/network transport and concurrent request queues.
"""
from pathlib import Path
import argparse
import copy
import hashlib
import importlib.metadata
import json
import math
import platform
import statistics
import time

HERE = Path(__file__).resolve().parent


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def percentile(values, fraction):
    values = sorted(values)
    index = (len(values) - 1) * fraction
    lo, hi = math.floor(index), math.ceil(index)
    return values[lo] + (values[hi] - values[lo]) * (index - lo)


def summarize(values):
    return {'mean': statistics.mean(values), 'p50': percentile(values, .5),
            'p95': percentile(values, .95), 'min': min(values), 'max': max(values)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-model', help='Optional local pinned backbone')
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--accelerated', action='store_true', help='Use fast preparation and qualified exact-shape CUDA Graphs')
    modes.add_argument('--turbo', action='store_true', help='Use two resident cores, merged/fused Graphs and native fallback')
    parser.add_argument('--adapter-directory', help='Optional local adapter directory; requires --base-model')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--samples', type=int, default=100)
    parser.add_argument('--cpu-threads', type=int, default=2)
    args = parser.parse_args()
    if args.samples < 20 or args.warmup < 1 or args.cpu_threads < 1:
        parser.error('Use at least 20 samples, one warmup, and one CPU thread')
    if args.adapter_directory and not args.base_model:
        parser.error('--adapter-directory requires --base-model')
    if not args.adapter_directory:
        from mentat_spice.hub import resolve_pretrained
        args.base_model, args.adapter_directory = resolve_pretrained(base_model=args.base_model)
    args.output.mkdir(parents=True, exist_ok=False)
    import torch
    from mentat_spice import DecisionModel
    if args.accelerated:
        from mentat_spice.accelerated import AcceleratedDecisionModel as DecisionModel
    elif args.turbo:
        from mentat_spice.turbo import TurboDecisionModel as DecisionModel
    optimized = args.accelerated or args.turbo
    from mentat_spice.scoring import to_decision_request, compile_question
    torch.set_num_threads(args.cpu_threads)
    torch.set_num_interop_threads(1)
    device = torch.device(args.device)
    if device.type != 'cuda':
        parser.error('This native release benchmark requires a CUDA device')
    torch.cuda.set_device(device)
    started = time.perf_counter()
    model = DecisionModel(args.base_model, args.adapter_directory, device=args.device)
    torch.cuda.synchronize(device)
    load_seconds = time.perf_counter() - started

    def question(count=4, kind='choice'):
        if kind == 'noul':
            return {'type': 'noul', 'instructions': 'The supplied delivery address is complete.'}
        return {'type': 'choice', 'instructions': 'Choose the best next handling action for this delivery.',
                'criteria': {f'option_{i}': text for i, text in enumerate(
                    ['Request address details', 'Dispatch immediately', 'Cancel the order', 'Manual review']
                    + [f'Alternative handling route {i}' for i in range(4, count)])}}

    def token_count(request):
        native = to_decision_request({'model': 'mentat-jev-v1', **request}, expected_model='mentat-jev-v1')
        field = native.fields[0]
        header, choices, texts = compile_question(field, native.state)
        labels = model.engine.labels(len(choices), 0)
        prompt = header + '\n'.join(f'{label}: {text}' for label, text in zip(labels, texts, strict=True))
        rendered = model.engine._render(prompt, 0)
        return len(model.engine.processor.tokenizer.encode(rendered, add_special_tokens=False))

    def text_case(name, target, options=4, fields=1, kind='choice'):
        line = 'Delivery note: confirm the missing street number with the customer before dispatch. '
        def make(repeats):
            return {'state': 'The delivery address has no street number.\n' + line * repeats,
                    'questions': {f'q{i}': question(options, kind) for i in range(fields)}}
        lo, hi = 0, target
        while lo < hi:
            mid = (lo + hi + 1)//2
            if token_count(make(mid)) <= target: lo = mid
            else: hi = mid - 1
        return {'name': name, 'request': make(lo), 'target_tokens_per_question': target,
                'description': 'Synthetic fixed-length load probe; not an accuracy evaluation.'}

    cases = [text_case('text_choice_256', 256), text_case('text_choice_512', 512),
             text_case('text_choice_2048', 2048), text_case('text_choice_32_options', 1024, options=32),
             text_case('text_noul_512', 512, kind='noul'),
             text_case('text_five_questions', 512, fields=5)]
    receipt = json.loads((HERE/'examples/receipt.json').read_text())
    request = copy.deepcopy(receipt['request']); request['images'] = [str(HERE/'examples/media/receipt.png')]
    cases.append({'name': 'receipt_image', 'request': request, 'description': 'CORD receipt example; one question.'})
    web = json.loads((HERE/'examples/web.json').read_text())
    request = copy.deepcopy(web['decisions'][0]['request']); request['images'] = [str(HERE/'examples/media/web-01.jpg')]
    cases.append({'name': 'web_image', 'request': request, 'description': 'Mind2Web reference screenshot; one question.'})
    save(args.output/'inputs.json', cases)
    rows = []
    for case in cases:
        for _ in range(args.warmup): model.predict(case['request'])
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        samples = []
        reference = None
        for _ in range(args.samples):
            torch.cuda.synchronize(device)
            tick = time.perf_counter()
            response = model.predict(case['request'])
            torch.cuda.synchronize(device)
            elapsed = (time.perf_counter() - tick) * 1000
            semantic = {k: {f: v for f, v in answer.items() if f != 'calibration_version'}
                        for k, answer in response['answers'].items()}
            if reference is None: reference = semantic
            if semantic != reference:
                raise RuntimeError(f'Non-deterministic output for {case["name"]}; retain actual samples')
            samples.append({'wall_ms': elapsed, 'usage': response['usage'], 'internal_timing': response['timing']})
        usage = samples[-1]['usage']
        wall = summarize([s['wall_ms'] for s in samples])
        row = {'scenario': case['name'], 'samples': args.samples, 'warmup': args.warmup,
               'questions_per_request': len(response['answers']), 'images': usage['images'],
               'input_tokens_per_question_max': usage['input_tokens'],
               'forward_passes_per_request': usage['language_model_passes'],
               'latency_ms': wall,
               'forward_ms': summarize([s['usage']['questions_ms'] for s in samples]),
               'preprocess_ms': summarize([s['usage']['preprocess_ms'] for s in samples]),
               'sequential_requests_per_second': 1000/wall['mean'],
               'sequential_fields_per_second': 1000*len(response['answers'])/wall['mean'],
               'peak_GPU_allocated_bytes': torch.cuda.max_memory_allocated(device),
               'peak_GPU_reserved_bytes': torch.cuda.max_memory_reserved(device),
               'output_identical_across_repeats': True}
        save(args.output/f'{case["name"]}.json', {'summary': row, 'samples': samples, 'response': response})
        rows.append(row)
        print(json.dumps(row), flush=True)
    versions = {name: importlib.metadata.version(name) for name in ['torch','transformers','peft','safetensors']}
    for name in ['flash-linear-attention', 'triton']:
        try: versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: pass
    prop = torch.cuda.get_device_properties(device)
    receipt = {'passed': True, 'model': 'mentat-Spice-4B', 'GPU': torch.cuda.get_device_name(device),
               'GPU_total_memory_bytes': prop.total_memory,
               'runtime': ('Two resident cores: merged/fused BF16 Graphs with independent native guard and fallback' if args.turbo else 'Fast text preparation, BF16 resident LoRA compute weights, first-use-qualified exact-shape CUDA Graphs, native operators' if args.accelerated else 'Bundled native eager PyTorch, BF16 base + FP32 LoRA, SDPA; no quantization or CUDA graphs'),
               'linear_attention_kernel': ('Instance-local FLA convolution override; native fallback keeps Transformers dispatch' if args.turbo else 'Native convolution fallback; Transformers automatically uses installed FLA for gated delta rule'),
               'batch_size': 1, 'concurrency': 1, 'model_load_seconds': load_seconds,
               'model_loading_in_latency': False, 'http_and_network_in_latency': False,
               'CUDA_synchronized_wall_time': True, 'CPU_threads': args.cpu_threads,
               'KV_or_output_cache': False, 'generated_answer_tokens': 0,
               'versions': versions, 'Python': platform.python_version(),
               'adapter_sha256': sha(Path(args.adapter_directory)/'adapter_model.safetensors'),
               'script_sha256': sha(Path(__file__)),
               'scope': 'Repeated fixed workloads on one shared host; not saturated throughput, API latency, or an accuracy benchmark.',
               'scenarios': rows, 'accelerated': optimized, 'turbo': args.turbo,
               'graph_count': len(model.graph_network.graphs) if optimized else 0,
               'graph_replays': model.graph_network.replays if optimized else 0,
               'graph_failures': model.graph_network.failures if optimized else [],
               'graph_qualifications': model.graph_network.qualifications if optimized else {},
               'graph_rejected': {str(k):v for k,v in model.graph_network.rejected.items()} if optimized else {},
               'safe_graph_count': len(model.graph_network.safe_graphs.graphs) if args.turbo else 0,
               'acceleration_config': model.acceleration if optimized else {}}
    save(args.output/'receipt.json', receipt)
    print(json.dumps({'passed': True, 'scenarios': len(rows), 'requests_measured': args.samples*len(rows)}), flush=True)


if __name__ == '__main__': main()
