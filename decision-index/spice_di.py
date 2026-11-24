"""Author inference adapter for the unmodified official Decision Index runners."""
from functools import lru_cache
from pathlib import Path
from time import perf_counter
import hashlib,math
from decision_index.engines import Engine,NativeAbstention,Unsupported

@lru_cache(maxsize=1)
def _model(model,revision,base_model,device,runtime):
    from mentat_spice import DecisionModel
    return DecisionModel.from_pretrained(model,revision=revision,base_model=base_model,device=device,runtime=runtime)

class SpiceEngine(Engine):
    name='mentat-spice'
    latency='Synchronized in-process request wall time, including image decoding and fixed prompt construction; excludes model loading.'
    def __init__(self,model='yunqu/mentat-Spice-4B',revision='90d4f76971ebd07c11054ada274767458b615ad5',base_model=None,device='cuda:0',image_root=None,max_pixels=1638400,runtime='turbo',**options):
        super().__init__(**options)
        if runtime not in {'native','accelerated','turbo'}:raise ValueError('runtime must be native, accelerated or turbo')
        self.runtime_name=runtime
        import torch
        torch.set_num_threads(4)
        if int(max_pixels)!=1638400:raise ValueError('DI submission image budget must be 1,638,400 pixels')
        self.model=_model(str(model),revision,str(base_model) if base_model else None,device,runtime)
        self.image_root=image_root;self.max_pixels=1638400
        self.provenance={'model':str(model),'revision':revision,'runtime':'mentat-sys1 0.4.0 '+runtime,'base_model':'Qwen/Qwen3.5-4B','base_revision':'851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a','device':device,'max_pixels':1638400,'max_images':4,'max_options':255,'max_context':self.model.engine.max_length,'image_resize':'Official decision_index.vision.images.pil, aspect-ratio preserving cap at 1,638,400 pixels','question_batch_size':8,'question_policy':'All questions retained in order and evaluated with identical state and unmodified criteria. Fixed groups of at most eight, independent of benchmark.','abstention_policy':'Preserved as official NativeAbstention; no forced fallback answer.'}
    def acceleration_state(self):
        graph=getattr(self.model,'graph_network',None)
        safe=getattr(graph,'safe_graphs',None)
        return {'mode':getattr(self,'runtime_name','native'),
                'fast_outputs':getattr(graph,'fast_outputs',0),
                'graph_replays':getattr(graph,'replays',0),
                'captured_shapes':len(getattr(graph,'graphs',{})),
                'rejected_shapes':len(getattr(graph,'rejected',{})),
                'graph_failures':len(getattr(graph,'failures',[])),
                'confidence_fallbacks':getattr(graph,'confidence_fallbacks',0),
                'native_graph_replays':getattr(safe,'replays',0),
                'native_graph_shapes':len(getattr(safe,'graphs',{})),
                'ambiguous_eager_fallbacks':getattr(graph,'ambiguous_eager_fallbacks',0)}
    def __call__(self,state,questions,images=(),prompt=None):
        from decision_index.vision.images import pil
        images=list(images or ())
        if len(images)>4:raise Unsupported('Declared maximum of four images per request')
        if not isinstance(questions,dict) or not questions:raise ValueError('questions must be a non-empty mapping')
        for question in questions.values():
            if question.get('type')=='choice' and len(question.get('criteria',{}))>255:raise Unsupported('Declared maximum of 255 options per choice')
        before=self.acceleration_state()
        decoded=[pil(str(x) if isinstance(x,Path) else x,self.image_root,self.max_pixels).convert('RGB') for x in images]
        answers={};usage={};parts=[];items=list(questions.items());tokens=[]
        for start in range(0,len(items),8):
            batch=dict(items[start:start+8]);payload={'state':state,'questions':batch}
            try:response=self.model.predict(payload,images=decoded)
            except ValueError as exc:
                message=str(exc)
                if any(k in message for k in ['processed request exceeds context limit','image exceeds released 2048-token guard']):raise Unsupported(message) from exc
                raise
            answers.update(response['answers']);parts.append(len(batch))
            for key,value in response.get('usage',{}).items():
                if isinstance(value,(int,float)):usage[key]=usage.get(key,0)+value
            probe=dict(self.model.engine.probe_meta)
            if 'image_tokens' in probe:tokens.append(probe['image_tokens'])
        assert list(answers)==list(questions),'question mapping changed'
        output={'model':'mentat-Spice-4B','answers':answers,'usage':usage}
        raw={'question_batches':parts,'image_pixels':[im.width*im.height for im in decoded],'image_tokens':max(tokens) if tokens else 0,'prompt_metadata_received':prompt is not None,'readout':'Native learned 256-code decision readout; supplied letters retain their original mapping','response':output,'acceleration':self.acceleration_state()}
        raw['acceleration_delta']={k:raw['acceleration'][k]-v for k,v in before.items() if type(v) is int}
        raw['input_tokens_last_field']=probe.get('input_tokens')
        if any(a.get('abstained') for a in answers.values()):raise NativeAbstention(raw)
        return output,raw
    def runtime(self):
        import torch,transformers,peft
        return {'torch':torch.__version__,'transformers':transformers.__version__,'peft':peft.__version__,'cuda':torch.version.cuda,'device':torch.cuda.get_device_name(self.model.engine.device) if torch.cuda.is_available() else 'cpu', 'acceleration_config':getattr(self.model,'acceleration',{}), 'adapter_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    def synchronize(self):
        import torch
        if torch.cuda.is_available():torch.cuda.synchronize()
    def close(self):pass
