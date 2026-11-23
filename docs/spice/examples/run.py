"""Inspect recorded decisions or rerun one example with the native release runtime."""
from __future__ import annotations

import argparse
import copy
import json
import re
from pathlib import Path


HERE = Path(__file__).resolve().parent
CASES = ('customer', 'invoice', 'security', 'contract', 'rag', 'shopping',
         'agent', 'tool', 'web', 'receipt', 'chart', 'babyai')


def prepare(case: str, decision: int = 1):
    record = json.loads((HERE / f'{case}.json').read_text())
    if case == 'babyai':
        raise ValueError('The complete BabyAI episode is an archived trajectory. Use --recorded to inspect it; this runner does not launch a game environment.')
    image_names = []
    if case in ('web', 'tool'):
        steps = record['decisions']
        if not 1 <= decision <= len(steps):
            raise ValueError(f'{case} decision must be between 1 and {len(steps)}')
        selected = steps[decision - 1]
        request = copy.deepcopy(selected['request'])
        if case == 'web':
            image_names = [f'web-{decision:02d}.jpg']
    elif case == 'receipt':
        if decision not in (1, 2):
            raise ValueError('receipt decision 1 reads total; decision 2 reads change')
        selected = record if decision == 1 else record['second_recorded_question']
        request = copy.deepcopy(record['request'])
        if decision == 2:
            request['questions'] = copy.deepcopy(selected['request'])
        image_names = ['receipt.png']
    else:
        if decision != 1:
            raise ValueError(f'{case} has one recorded request')
        selected = record
        request = copy.deepcopy(record['request'])
        if case == 'contract':
            # These are the three displayed fields. The release evaluates each
            # question separately; the complete archived request is retained.
            request['questions'] = {f: request['questions'][f] for f in record['displayed_fields']}
        if case == 'chart':
            image_names = ['chart.jpg']
    if not 1 <= len(request['questions']) <= 8:
        raise ValueError('Prepared example must fit the native eight-question limit')
    # Historical benchmark field IDs can contain hyphens. Native field IDs
    # are identifiers; normalize those names without changing question content.
    field_ids = {name: re.sub(r'[^A-Za-z0-9_]', '_', name)
                 for name in request['questions']}
    if len(set(field_ids.values())) != len(field_ids):
        raise ValueError('Historical field IDs collide after normalization')
    request['questions'] = {field_ids[name]: question
                            for name, question in request['questions'].items()}
    selected = copy.deepcopy(selected)
    selected['response']['answers'] = {
        field_ids.get(name, name): answer
        for name, answer in selected['response']['answers'].items()}
    if image_names:
        request['images'] = [str(HERE / 'media' / name) for name in image_names]
        for image in request['images']:
            if not Path(image).is_file():
                raise FileNotFoundError(image)
    return request, selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', required=True, choices=CASES)
    parser.add_argument('--decision', type=int, default=1, help='1-based reference step; receipt: 1=total, 2=change')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--recorded', action='store_true', help='Print the archived record; no model is loaded')
    mode.add_argument('--prepare-only', action='store_true', help='Print a native request, including local image paths; no model is loaded')
    parser.add_argument('--base-model', help='Optional local pinned backbone')
    parser.add_argument('--adapter-directory', help='Optional local adapter directory; requires --base-model')
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    if args.recorded:
        print((HERE / f'{args.case}.json').read_text(), end='')
        return
    try:
        request, _ = prepare(args.case, args.decision)
    except (ValueError, FileNotFoundError) as exc:
        parser.error(str(exc))
    if args.prepare_only:
        print(json.dumps(request, indent=2, ensure_ascii=False))
        return
    from mentat_spice import DecisionModel
    if args.adapter_directory:
        if not args.base_model:
            parser.error('--adapter-directory requires --base-model')
        model = DecisionModel(args.base_model, args.adapter_directory, device=args.device)
    else:
        model = DecisionModel.from_pretrained(base_model=args.base_model, device=args.device)
    response = model.predict(request)
    print(json.dumps({'case': args.case, 'reference_decision': args.decision,
                      'mode': 'new_native_inference', 'external_actions_executed': False,
                      'response': response}, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
