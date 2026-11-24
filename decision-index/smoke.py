"""Portable integration check; this is not a benchmark score."""
import argparse
import json
import statistics
import time
from decision_index.engines import NativeAbstention, validate
from spice_di import SpiceEngine

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='yunqu/mentat-Spice-4B')
parser.add_argument('--revision', default='90d4f76971ebd07c11054ada274767458b615ad5')
parser.add_argument('--base-model')
parser.add_argument('--device', default='cuda:0')
parser.add_argument('--runtime', default='turbo', choices=['native','accelerated','turbo'])
args = parser.parse_args()
import torch
torch.set_num_threads(4)
engine = SpiceEngine(**vars(args))
questions = {'color':{'type':'choice','instructions':'Which color is named?',
                     'criteria':{'r':'red','b':'blue','g':'green','y':'yellow'}}}
def call():
    try: response, raw = engine('The color is red.', questions)
    except NativeAbstention as error:
        raw=error.raw;response=raw['response']
    validate(questions,response)
    return response
engine.synchronize();start=time.perf_counter();response=call();engine.synchronize()
cold_ms=1000*(time.perf_counter()-start)
times=[]
for _ in range(20):
    engine.synchronize();start=time.perf_counter();call();engine.synchronize()
    times.append(1000*(time.perf_counter()-start))
print(json.dumps({'smoke_only':True,'response':response,'first_request_ms':cold_ms,
                  'warm_median_ms':statistics.median(times),'warm_mean_ms':statistics.mean(times),
                  'acceleration':engine.acceleration_state(),'runtime':engine.runtime()}, indent=2))
