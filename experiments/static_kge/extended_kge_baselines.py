"""Frozen-protocol KGE baselines; official sources remain unmodified."""
from __future__ import annotations
import os,sys,json,hashlib,random,time,copy,types,importlib.util
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'outputs/official_kge_dependencies'))
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from pair_prototype_full_suite import load_data,metrics,save,save_npz
from train_baselines import sample_negative_tails
P=json.loads((HERE/'pairproto_extended_baselines_protocol.json').read_text(encoding='utf-8'))
OUT=HERE/'results'/'extended_baselines_20261024'
OUT.mkdir(parents=True,exist_ok=True)
PROTOCOL_SHA=hashlib.sha256((HERE/'pairproto_extended_baselines_protocol.json').read_bytes()).hexdigest()
def seed_all(seed):
 random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
def tag(source,name,seed,lr):return f'{source.lower()}_{P["datasets"][source]}_{name}_lr{lr:g}_seed{seed}'
def paths(source,name,seed,lr):
 stem=OUT/tag(source,name,seed,lr)
 return tuple(Path(str(stem)+s) for s in ['.pt','_dev.json','_test.json','_rr.npz'])
def cpu_state(model):return {k:v.detach().cpu() for k,v in model.state_dict().items()}
class MeanRealKGE(nn.Module):
 def __init__(self,data,name):
  super().__init__();self.name=name;width=P['classic']['real_dimension']
  self.entity=nn.Embedding(data.n_entities,width);self.relation=nn.Embedding(data.n_relations,width)
  if name=='mean_transe':
   lim=6/width**.5;nn.init.uniform_(self.entity.weight,-lim,lim);nn.init.uniform_(self.relation.weight,-lim,lim)
  else:nn.init.xavier_uniform_(self.entity.weight);nn.init.xavier_uniform_(self.relation.weight)
  for key,index in [('ingredients',0),('mask',1),('weights',2)]:
   self.register_buffer(key,torch.as_tensor(data.arrays[index]),persistent=False)
 def head(self,h):
  out=self.entity(h);formula=self.mask[h].any(-1)
  if formula.any():
   herbs=self.ingredients[h[formula]];weight=self.weights[h[formula]]
   out=out.clone();out[formula]=(self.entity(herbs)*weight[...,None]).sum(1)
  return out
 def bundle(self,h,r,tails):
  he=self.head(h);rel=self.relation(r);te=self.entity(tails)
  if self.name=='mean_transe':
   he=F.normalize(he,dim=-1);te=F.normalize(te,dim=-1)
   return 12-(he[:,None]+rel[:,None]-te).abs().sum(-1)
  return (he[:,None]*rel[:,None]*te).sum(-1)
 def score_tails(self,h,r,t):return self.bundle(h.reshape(1),r.reshape(1),t.reshape(1,-1))[0]
def composition_rows(data,heads):
 ingredients,mask,*_=data.arrays
 out=[(int(h),data.n_relations,int(x)) for h in sorted(set(map(int,heads))) for x in ingredients[h,mask[h]]]
 return np.asarray(out,dtype=np.int64).reshape(-1,3)
def graph_forward(data,phase):
 held=set(map(int,data.dev[:,0]))|set(map(int,data.test[:,0]))
 assert not np.isin(data.train[:,[0,2]],list(held)).any()
 heads=[int(h) for h in np.unique(data.train[:,0]) if data.arrays[1][int(h)].any()]
 rows=composition_rows(data,heads)
 if phase!='train':rows=np.concatenate([rows,composition_rows(data,np.unique(getattr(data,phase)[:,0]))])
 out=np.unique(np.concatenate([data.train,rows]),axis=0)
 for h,r,t in out:
  if int(h) in held:assert r==data.n_relations
  assert int(t) not in held
 return out
def inverse(rows,nrel):
 out=rows[:,[2,1,0]].copy();out[:,1]+=nrel
 return np.concatenate([rows,out])
def graph_audit(data):
 sets={}
 for phase in ['train','dev','test']:
  rows=graph_forward(data,phase);held=np.unique(np.concatenate([data.dev[:,0],data.test[:,0]]))
  sets[phase]={'forward_edges':len(rows),'held_composition_edges':int(np.isin(rows[:,0],held).sum()),'sha256':hashlib.sha256(rows.tobytes()).hexdigest()}
 return {'protocol_sha256':PROTOCOL_SHA,'split':data.audit,'graphs':sets,'forbidden_held_target_edges':0,
  'queries_sha256':{p:hashlib.sha256(getattr(data,p).tobytes()).hexdigest() for p in ['dev','test']},
  'candidate_sha256':hashlib.sha256(b''.join(np.asarray([r],np.int64).tobytes()+c.tobytes() for r,c in sorted(data.range_candidates.items()))).hexdigest()}
def official_module(path,name,remove_imports=(),replacements=()):
 text=Path(path).read_text(encoding='utf-8')
 for line in remove_imports:text=text.replace(line,'')
 for a,b in replacements:
  assert a in text,(path,a);text=text.replace(a,b)
 mod=types.ModuleType(name);mod.__file__=str(path);sys.modules[name]=mod
 exec(compile('from __future__ import annotations\n'+text,str(path),'exec'),mod.__dict__)
 return mod

class OfficialNodePiece(nn.Module):
 def __init__(self,data,source,seed):
  super().__init__();base=HERE/'third_party/NodePiece/oos_lp/src'
  tokmod=official_module(base/'vocab/nodepiece_tokenizer.py','tcm_nodepiece_tokenizer',
   ['from pykeen.triples import TriplesFactory'],
   [('self.triples_factory.num_entities','self.num_nodes')])
  encmod=official_module(base/'vocab/nodepiece_encoder.py','tcm_nodepiece_encoder',
   ['from vocab.nodepiece_tokenizer import NodePiece_Tokenizer','from common.dataset import Dataset'],
   [('self.no_anc = config.no_anc','self.no_anc = config.no_anc\n        self.ablate_anchors = config.no_anc')])
  fw=graph_forward(data,'train');self.nrel=data.n_relations+1;rows=inverse(fw,self.nrel)
  cache=HERE/'outputs/extended_nodepiece_tokens';(cache/'data').mkdir(parents=True,exist_ok=True)
  old=os.getcwd()
  try:
   os.chdir(cache)
   tokenizer=tokmod.NodePiece_Tokenizer(rows,f'{source}_{P["datasets"][source]}',min(500,len(np.unique(fw[:,[0,2]]))),
      {'degree':1.,'betweenness':0.,'pagerank':0.,'random':0.},limit_shortest=20,
      relation2id={str(i):i for i in range(2*self.nrel)},num_nodes=data.n_entities)
  finally:os.chdir(old)
  # Attach each unseen formula to the fixed training hashes through its known herbs only.
  base_vocab=copy.deepcopy(tokenizer.vocab)
  held=sorted(set(map(int,data.dev[:,0]))|set(map(int,data.test[:,0])))
  for h in held:
   choices={}
   for herb in data.arrays[0][h,data.arrays[1][h]]:
    for path in base_vocab[int(herb)]:
     anchor=path[0]
     if anchor<0:continue
     new=list(path)+[data.n_relations]
     if anchor not in choices or len(new)<len(choices[anchor]):choices[anchor]=new
   tokenizer.vocab[h]=sorted(choices.values(),key=lambda p:(len(p),p[0]))[:20] or [[tokenizer.NOTHING_TOKEN]]
  cfg=types.SimpleNamespace(pooler='cat',nearest_ancs=True,sample_rels=4,random_hashes=0,subbatch=256,emb_dim=256,
   max_path_len=max(32,max(len(p) for vv in tokenizer.vocab.values() for p in vv)+2),sample_size=20,anc_dist=True,
   t_hidden=512,t_drop=.1,t_heads=4,t_layers=1,no_anc=False)
  self.rel_embs=nn.Embedding(2*self.nrel+1,256,padding_idx=2*self.nrel)
  nn.init.xavier_uniform_(self.rel_embs.weight[:-1])
  graph=types.SimpleNamespace(init_num_ent=data.n_entities,rel2id={str(i):i for i in range(2*self.nrel)},train_data=rows)
  self.embedder=encmod.NodePieceEncoder(cfg,tokenizer,self.rel_embs,graph,torch.device('cuda'))
  self.embedder.reset_parameters()
  for h in held:
   self.embedder.unique_1hop_relations[h].fill_(2*self.nrel)
   self.embedder.unique_1hop_relations[h,0]=data.n_relations
 def bundle(self,h,r,tails):
  he=self.embedder.encode_by_index(h);re=self.rel_embs(r)
  flat=tails.reshape(-1);unique,inv=torch.unique(flat,return_inverse=True)
  te=self.embedder.encode_by_index(unique)[inv].reshape(*tails.shape,-1)
  return (he[:,None]*re[:,None]*te).sum(-1)
 def score_tails(self,h,r,t):
  cached=getattr(self,'eval_cache',None)
  if cached is not None:return (cached[h]*self.rel_embs(r)*cached[t]).sum(-1)
  return self.bundle(h.reshape(1),r.reshape(1),t.reshape(1,-1))[0]
@torch.inference_mode()
def embedding_ranks(model,data,phase):
 model.eval();rr=[];cache={}
 if isinstance(model,OfficialNodePiece):
  needed=np.unique(np.concatenate([getattr(data,phase)[:,0],*[data.range_candidates[int(r)] for r in np.unique(getattr(data,phase)[:,1])]]))
  model.eval_cache=torch.empty((data.n_entities,256),device='cuda')
  for begin in range(0,len(needed),512):
   ids=torch.tensor(needed[begin:begin+512],device='cuda')
   model.eval_cache[ids]=model.embedder.encode_by_index(ids)
 for h,r,t in getattr(data,phase):
  h,r,t=map(int,(h,r,t));c=data.range_candidates[r]
  if (h,r) not in cache:
   score=model.score_tails(torch.tensor(h,device='cuda'),torch.tensor(r,device='cuda'),torch.tensor(c,device='cuda')).float().cpu().numpy()
   if not np.isfinite(score).all():raise FloatingPointError('Nonfinite scores')
   cache[h,r]=score
  score=cache[h,r];pos=int(np.searchsorted(c,t));assert c[pos]==t
  allow=~np.isin(c,list(data.all_true[h,r]-{t}))
  rank=1+np.sum((score>score[pos])&allow)+.5*max(0,int(np.sum((score==score[pos])&allow))-1)
  rr.append(1/float(rank))
 if isinstance(model,OfficialNodePiece):model.eval_cache=None
 return np.asarray(rr,np.float64)

def train_embedding(source,name,seed,lr,smoke=False):
 data=load_data(source,P['datasets'][source]);seed_all(seed)
 model=(OfficialNodePiece(data,source,seed) if name=='nodepiece' else MeanRealKGE(data,name)).cuda()
 ckpt,devpath,_,_=paths(source,name,seed,lr)
 if devpath.exists() and ckpt.exists() and not smoke:return json.loads(devpath.read_text(encoding='utf-8'))
 optimizer=torch.optim.Adam(model.parameters(),lr=lr,weight_decay=1e-7);rng=np.random.default_rng(seed)
 history=[];best=-float('inf');stale=0;started=time.perf_counter()
 for epoch in range(1,(1 if smoke else 40)+1):
  model.train();order=rng.permutation(len(data.train));losses=[]
  if smoke:order=order[:32]
  for offset in range(0,len(order),2048):
   raw=data.train[order[offset:offset+2048]];neg=sample_negative_tails(data,raw,rng)
   optimizer.zero_grad(set_to_none=True);micro=512
   for begin in range(0,len(raw),micro):
    x=raw[begin:begin+micro];b=torch.as_tensor(x,device='cuda')
    tails=torch.tensor(np.concatenate([x[:,2:3],neg[begin:begin+len(x)]],axis=1),device='cuda')
    score=model.bundle(b[:,0],b[:,1],tails)
    loss=F.softplus(-score[:,0]).mean()+F.softplus(score[:,1:]).mean()
    if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss')
    (loss*len(x)/len(raw)).backward();losses.append(float(loss.detach()))
   optimizer.step()
  if epoch==1 or epoch%5==0:
   rr=embedding_ranks(model,data,'dev');dev=metrics(rr)
   row={'epoch':epoch,'loss':float(np.mean(losses)),'dev':dev};history.append(row)
   print('DEV',tag(source,name,seed,lr),row,flush=True)
   if dev['mrr']>best:
    best=dev['mrr'];selected=row;stale=0
    if not smoke:torch.save({'state_dict':cpu_state(model),'protocol_sha256':PROTOCOL_SHA},ckpt)
   else:stale+=1
   if stale>=4:break
 result={'source':source,'split':P['datasets'][source],'model':name,'seed':seed,'lr':lr,'protocol_sha256':PROTOCOL_SHA,
  'dev':selected['dev'],'selected':selected,'history':history,'parameters':sum(p.numel() for p in model.parameters()),
  'seconds':time.perf_counter()-started,'checkpoint':str(ckpt),'audit':graph_audit(data)}
 if not smoke:save(devpath,result)
 return result
def test_embedding(source,name,seed,lr):
 ckpt,devpath,testpath,rankpath=paths(source,name,seed,lr)
 if testpath.exists() and rankpath.exists():return json.loads(testpath.read_text(encoding='utf-8'))
 info=train_embedding(source,name,seed,lr)
 data=load_data(source,P['datasets'][source]);seed_all(seed)
 model=(OfficialNodePiece(data,source,seed) if name=='nodepiece' else MeanRealKGE(data,name)).cuda()
 model.load_state_dict(torch.load(ckpt,map_location='cuda',weights_only=False)['state_dict'])
 rr=embedding_ranks(model,data,'test')
 save_npz(rankpath,rr=rr,heads=data.test[:,0],relations=data.test[:,1])
 result={**info,'test':metrics(rr),'rank_path':str(rankpath)};save(testpath,result)
 print('TEST',source,name,seed,result['test'],flush=True)
 return result

def publish_selected_result(source,name,seed,result):
 import shutil
 stem=f'extended_baseline_{source.lower()}_{P["datasets"][source]}_{name}_seed{seed}'
 destination=HERE/'results'
 shutil.copyfile(result['rank_path'],destination/(stem+'_rr.npz'))
 save(destination/(stem+'_test.json'),result)
 return stem
def publish_selection_manifest():
 configs={}
 for source in P['datasets']:
  configs[source]={}
  for name in P['models']:
   path=OUT/f'{source.lower()}_{name}_selection.json'
   row=json.loads(path.read_text(encoding='utf-8'))
   assert row['protocol_sha256']==PROTOCOL_SHA
   configs[source][name]=row
 report={'protocol_sha256':PROTOCOL_SHA,'datasets':P['datasets'],'models':configs,
  'official_sources':P['official_sources'],'evaluation_seeds':P['evaluation_seeds']}
 destination=HERE/'results/pairproto_extended_baselines_selection.json'
 # Multiple selected workers publish the same immutable manifest. Use one unique
 # temporary file per process and retry transient Windows sharing locks.
 for attempt in range(20):
  try:
   existing=json.loads(destination.read_text(encoding='utf-8'))
  except FileNotFoundError:
   break
  except PermissionError:
   if attempt==19:raise
   time.sleep(.1)
   continue
  if existing==report:return report
  raise RuntimeError('Existing extension manifest differs from frozen selections')
 import uuid
 temporary=destination.with_name(destination.name+'.'+str(os.getpid())+'.'+uuid.uuid4().hex+'.tmp')
 temporary.write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
 for attempt in range(20):
  try:
   temporary.replace(destination)
   return report
  except PermissionError:
   if attempt==19:raise
   time.sleep(.1)
 return report
