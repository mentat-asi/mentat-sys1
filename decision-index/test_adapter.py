from types import SimpleNamespace
import pytest
from decision_index.engines import NativeAbstention,Unsupported,validate
from spice_di import SpiceEngine

def fake_model():
 class Model:
  engine=SimpleNamespace(probe_meta={'image_tokens':0})
  def __init__(self):self.calls=[];self.abstain=False;self.error=None
  def predict(self,payload,images):
   if self.error:raise ValueError(self.error)
   self.calls.append(payload)
   answers={k:{'type':'choice','choice':next(iter(q['criteria'])),'probabilities':{x:1/len(q['criteria']) for x in q['criteria']},'abstained':self.abstain} for k,q in payload['questions'].items()}
   return {'answers':answers}
 return Model()
def engine():
 e=object.__new__(SpiceEngine);e.model=fake_model();e.image_root=None;e.max_pixels=1638400;return e
def questions(n=20,options=4):return {f'q{i}':{'type':'choice','instructions':'Select the matching option','criteria':{str(k):str(k) for k in range(options)}} for i in range(n)}
def test_full_question_mapping_and_generic_chunking():
 e=engine();q=questions();state={'unchanged':['state','data']};response,raw=e(state,q)
 validate(q,response);assert [len(x['questions']) for x in e.model.calls]==[8,8,4]
 assert all(x['state'] is state for x in e.model.calls);assert list(response['answers'])==list(q)
 assert {k:v for c in e.model.calls for k,v in c['questions'].items()}==q
def test_255_options_and_complete_distribution():
 e=engine();q=questions(1,255);response,_=e('',q);validate(q,response)
 assert len(response['answers']['q0']['probabilities'])==255
@pytest.mark.parametrize('images,options',[([b'x']*5,4),([],256)])
def test_declared_limits(images,options):
 with pytest.raises(Unsupported):engine()('',questions(1,options),images=images)
def test_native_abstention_is_not_a_valid_answer():
 e=engine();e.model.abstain=True
 with pytest.raises(NativeAbstention) as ex:e('',questions(1))
 assert ex.value.raw['response']['answers']['q0']['abstained']
def test_only_capacity_errors_are_unsupported():
 e=engine();e.model.error='processed request exceeds context limit'
 with pytest.raises(Unsupported):e('',questions(1))
 e.model.error='unexpected tokenizer binding mismatch'
 with pytest.raises(ValueError,match='tokenizer'):e('',questions(1))

def test_opaque_ids_and_long_content_are_preserved():
 from mentat_spice.scoring import to_decision_request
 question='Evidence sentence. '*600
 description='Detailed criterion. '*200
 payload={'model':'mentat-jev-v1','state':'x','questions':{'nda-11#arbitrary':{'type':'choice','instructions':question,'criteria':{'a':description,'b':'Other'}}}}
 request=to_decision_request(payload,expected_model='mentat-jev-v1')
 assert request.fields[0].id=='nda-11#arbitrary'
 assert request.fields[0].question==question
 assert request.fields[0].options[0].description==description

def test_runtime_selection_and_cache_are_separate(monkeypatch):
 from mentat_spice import DecisionModel
 from spice_di import _model
 calls=[]
 def load(model,**options):
  calls.append(options);return object()
 monkeypatch.setattr(DecisionModel,'from_pretrained',load)
 _model.cache_clear()
 a=_model('model','pinned',None,'cuda:0','native')
 b=_model('model','pinned',None,'cuda:0','turbo')
 assert a is not b
 assert [x['runtime'] for x in calls]==['native','turbo']
 _model.cache_clear()

def test_acceleration_receipt_does_not_modify_answers():
 e=engine();q=questions(1);response,raw=e('',q)
 validate(q,response)
 assert raw['acceleration']['mode']=='native'
 assert raw['acceleration_delta']['fast_outputs']==0
