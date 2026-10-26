"""Adapters around official InGram, ULTRA and TRIX, without held-label messages."""
from extended_kge_baselines import *
from functools import lru_cache
class ChunkedDistMultSum(torch.autograd.Function):
 """Exact edge-wise DistMult/sum with recomputed chunks in backward, avoiding the optional CUDA extension."""
 @staticmethod
 def forward(ctx,x,rel,boundary,edge_index,edge_type,weight):
  ctx.save_for_backward(x,rel,edge_index,edge_type,weight)
  out=boundary.clone()
  for begin in range(0,edge_type.numel(),32768):
   h,t=edge_index[:,begin:begin+32768];r=edge_type[begin:begin+32768];w=weight[begin:begin+32768]
   out.index_add_(1,t,x[:,h]*rel[:,r]*w[None,:,None])
  return out
 @staticmethod
 def backward(ctx,grad):
  x,rel,edge_index,edge_type,weight=ctx.saved_tensors
  gx=torch.zeros_like(x);gr=torch.zeros_like(rel)
  for begin in range(0,edge_type.numel(),32768):
   h,t=edge_index[:,begin:begin+32768];r=edge_type[begin:begin+32768];w=weight[begin:begin+32768]
   g=grad[:,t]*w[None,:,None]
   gx.index_add_(1,h,g*rel[:,r]);gr.index_add_(1,r,g*x[:,h])
  return gx,gr,grad,None,None,None
_ADJ_CACHE={}
def sparse_blocks(edge_index,edge_type,n,device,dtype):
 import weakref
 key=(id(edge_index),id(edge_type),dtype)
 if key in _ADJ_CACHE and _ADJ_CACHE[key][0]() is edge_index:return _ADJ_CACHE[key][1]
 blocks=[]
 for r in torch.unique(edge_type).tolist():
  selected=edge_type==r;h,t=edge_index[:,selected]
  src,si=torch.unique(h,return_inverse=True);dst,ti=torch.unique(t,return_inverse=True)
  adj=torch.sparse_coo_tensor(torch.stack([ti,si]),torch.ones(len(si),device=device,dtype=dtype),
   (len(dst),len(src))).coalesce().to_sparse_csr()
  blocks.append((int(r),src,dst,adj,adj.transpose(0,1).to_sparse_csr()))
 def cleanup(ref):_ADJ_CACHE.pop(key,None)
 _ADJ_CACHE[key]=(weakref.ref(edge_index,cleanup),blocks)
 return blocks
class SparseDistMultSum(torch.autograd.Function):
 @staticmethod
 def forward(ctx,x,rel,boundary,edge_index,edge_type,weight):
  blocks=sparse_blocks(edge_index,edge_type,x.shape[1],x.device,x.dtype)
  ctx.blocks=blocks;ctx.save_for_backward(x,rel)
  out=boundary.clone();batch,n,d=x.shape
  for r,src,dst,adj,adjT in blocks:
   features=x[:,src].transpose(0,1).reshape(len(src),batch*d)
   msg=torch.sparse.mm(adj,features).reshape(len(dst),batch,d).transpose(0,1)
   out.index_add_(1,dst,msg*rel[:,r:r+1])
  return out
 @staticmethod
 def backward(ctx,grad):
  x,rel=ctx.saved_tensors;batch,n,d=x.shape;gx=torch.zeros_like(x);gr=torch.zeros_like(rel)
  for r,src,dst,adj,adjT in ctx.blocks:
   g=grad[:,dst];features=x[:,src].transpose(0,1).reshape(len(src),batch*d)
   msg=torch.sparse.mm(adj,features).reshape(len(dst),batch,d).transpose(0,1)
   gr[:,r]=(g*msg).sum(1)
   weighted=(g*rel[:,r:r+1]).transpose(0,1).reshape(len(dst),batch*d)
   value=torch.sparse.mm(adjT,weighted).reshape(len(src),batch,d).transpose(0,1)
   gx.index_add_(1,src,value)
  return gx,gr,grad,None,None,None
def portable_propagate(self,edge_index,size=None,**kw):
 if self.message_func=='distmult' and self.aggregate_func=='sum' and not kw['edge_weight'].requires_grad:
  implementation=SparseDistMultSum if int(kw['edge_type'].max())<64 and bool((kw['edge_weight']==1).all()) else ChunkedDistMultSum
  out=implementation.apply(kw['input'],kw['relation'],kw['boundary'],edge_index,kw['edge_type'],kw['edge_weight'])
 else:
  msg=self.message(input_j=kw['input'].index_select(self.node_dim,edge_index[0]),
   relation=kw['relation'],boundary=kw['boundary'],edge_type=kw['edge_type'])
  out=self.aggregate(msg,kw['edge_weight'],edge_index[1],size[1])
 return self.update(out,kw['input'])
@lru_cache(None)
def graph_modules(name):
 sys.path.insert(0,str(HERE/'third_party/ULTRA'))
 sys.path.insert(0,str(HERE/'third_party/TRIX/src'))
 if name=='ultra':
  from ultra import models,layers,tasks
 else:
  from trix import models_entity as models,layers,tasks
 layers.GeneralizedRelationalConv.propagate=portable_propagate
 return models,layers,tasks
def make_graph_model(name):
 models,_,_=graph_modules(name);spec=P[name];d=spec['dimension']
 common={'input_dim':d,'message_func':'distmult','aggregate_func':'sum','short_cut':True,'layer_norm':True}
 if name=='ultra':
  return models.Ultra(dict(common,**{'class':'RelNBFNet','hidden_dims':[d]*6}),
                      dict(common,**{'class':'EntityNBFNet','hidden_dims':[d]*6})).cuda()
 return models.TRIX(dict(common,hidden_dims=[d]*3),dict(common,hidden_dims=[d]*2),dict(common,hidden_dims=[d]*4)).cuda()
@lru_cache(maxsize=12)
def message_graph(source,name,phase):
 from torch_geometric.data import Data
 data=load_data(source,P['datasets'][source]);rows=inverse(graph_forward(data,phase),data.n_relations+1)
 graph=Data(edge_index=torch.tensor(rows[:,[0,2]].T,dtype=torch.long),
  edge_type=torch.tensor(rows[:,1],dtype=torch.long),num_nodes=data.n_entities,num_relations=2*(data.n_relations+1))
 # Relation builders operate on CPU; avoid thousands of GPU .item() synchronizations in official TRIX.
 _,_,tasks=graph_modules(name);graph=tasks.build_relation_graph(graph)
 return graph.to('cuda')
@torch.inference_mode()
def graph_ranks(model,data,graph,phase,smoke=False):
 model.eval();triples=getattr(data,phase);cache={}
 keys=np.unique(triples[:,:2],axis=0)
 if smoke:keys=keys[:8]
 for relation in np.unique(keys[:,1]):
  heads=keys[keys[:,1]==relation,0];c=data.range_candidates[int(relation)]
  for begin in range(0,len(heads),8):
   hs=heads[begin:begin+8];b=np.empty((len(hs),len(c),3),np.int64)
   b[:,:,0]=hs[:,None];b[:,:,1]=c[None];b[:,:,2]=relation
   # Batch order in official models is h,t,r. No gold tail is passed as a query feature.
   scores=model(graph,torch.tensor(b,device='cuda')).float().cpu().numpy()
   if not np.isfinite(scores).all():raise FloatingPointError('Nonfinite graph scores')
   for h,score in zip(hs,scores):cache[int(h),int(relation)]=score
 rr=[]
 for h,r,t in triples:
  h,r,t=map(int,(h,r,t))
  if (h,r) not in cache:continue
  c=data.range_candidates[r];score=cache[h,r];pos=int(np.searchsorted(c,t));assert c[pos]==t
  allow=~np.isin(c,list(data.all_true[h,r]-{t}))
  rank=1+np.sum((score>score[pos])&allow)+.5*max(0,int(np.sum((score==score[pos])&allow))-1)
  rr.append(1/float(rank))
 return np.asarray(rr,np.float64)

def train_nbf(source,name,seed,lr,smoke=False):
 data=load_data(source,P['datasets'][source]);seed_all(seed)
 ckpt,devpath,_,_=paths(source,name,seed,lr)
 if devpath.exists() and ckpt.exists() and not smoke:return json.loads(devpath.read_text(encoding='utf-8'))
 model=make_graph_model(name);train_graph=message_graph(source,name,'train');dev_graph=message_graph(source,name,'dev')
 spec=P[name];optimizer=torch.optim.AdamW(model.parameters(),lr=lr)
 rng=np.random.default_rng(seed);history=[];best=-float('inf');stale=0;started=time.perf_counter()
 for epoch in range(1,(1 if smoke else spec['max_epochs'])+1):
  model.train();losses=[]
  for step in range(1 if smoke else spec['batches_per_epoch']):
   raw=data.train[rng.integers(len(data.train),size=spec['batch_size'])]
   neg=sample_negative_tails(data,raw,rng);tails=np.concatenate([raw[:,2:3],neg],axis=1)
   batch=np.empty((*tails.shape,3),np.int64);batch[:,:,0]=raw[:,0,None];batch[:,:,1]=tails;batch[:,:,2]=raw[:,1,None]
   optimizer.zero_grad(set_to_none=True)
   score=model(train_graph,torch.tensor(batch,device='cuda'))
   weights=F.softmax(score[:,1:].detach(),dim=-1)
   loss=F.softplus(-score[:,0]).mean()+(F.softplus(score[:,1:])*weights).sum(-1).mean()
   if not torch.isfinite(loss):raise FloatingPointError('Nonfinite graph loss')
   loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
   losses.append(float(loss.detach()))
  rr=graph_ranks(model,data,dev_graph,'dev',smoke);dev=metrics(rr)
  row={'epoch':epoch,'loss':float(np.mean(losses)),'dev':dev,'elapsed':time.perf_counter()-started};history.append(row)
  print('DEV',tag(source,name,seed,lr),row,flush=True)
  if dev['mrr']>best:
   best=dev['mrr'];selected=row;stale=0
   if not smoke:torch.save({'state_dict':cpu_state(model),'protocol_sha256':PROTOCOL_SHA},ckpt)
  else:stale+=1
  if stale>=spec['patience_evaluations']:break
 result={'source':source,'split':P['datasets'][source],'model':name,'seed':seed,'lr':lr,'protocol_sha256':PROTOCOL_SHA,
  'dev':selected['dev'],'selected':selected,'history':history,'parameters':sum(p.numel() for p in model.parameters()),
  'seconds':time.perf_counter()-started,'checkpoint':str(ckpt),'external_pretraining':False,
  'training_budget':spec,'audit':graph_audit(data)}
 if not smoke:save(devpath,result)
 return result

@lru_cache(None)
def ingram_modules():
 base=HERE/'third_party/InGram'
 mod=official_module(base/'model.py','tcm_official_ingram',replacements=[('/ent_freq','/ent_freq.clamp_min(1)')])
 # relgraph imports utils (only NumPy/math helper functions); load it under its expected name locally.
 old=sys.modules.get('utils')
 utils=official_module(base/'utils.py','utils')
 relgraph=official_module(base/'relgraph.py','tcm_ingram_relgraph')
 if old is not None:sys.modules['utils']=old
 else:sys.modules.pop('utils',None)
 return mod,relgraph
def make_ingram():
 mod,_=ingram_modules();s=P['ingram']
 model=mod.InGram(s['d_e'],s['hidden_ratio_e'],s['d_r'],s['hidden_ratio_r'],s['bins'],
  num_layer_ent=s['entity_layers'],num_layer_rel=s['relation_layers'],num_head=s['heads']).cuda()
 return checkpoint_ingram(model)
def ingram_init(data,message,seed):
 _,relmod=ingram_modules();s=P['ingram'];nr=data.n_relations+1
 reltrip=relmod.generate_relation_triplets(message,data.n_entities,nr,s['bins'])
 # Local generator prevents graph initialization from resetting training or model RNG streams.
 gen=torch.Generator(device='cuda').manual_seed(seed)
 ent=torch.empty((data.n_entities,s['d_e']),device='cuda');rel=torch.empty((2*nr,s['d_r']),device='cuda')
 for arr in [ent,rel]:nn.init.xavier_normal_(arr,gain=nn.init.calculate_gain('relu'),generator=gen)
 return ent,rel,torch.tensor(message,device='cuda'),torch.tensor(reltrip,dtype=torch.long,device='cuda')
@torch.inference_mode()
def ingram_ranks(model,data,cache,phase,smoke=False):
 model.eval();ent,rel=model(*cache);rr=[]
 triples=getattr(data,phase)[:32] if smoke else getattr(data,phase);scores={}
 for h,r,t in triples:
  h,r,t=map(int,(h,r,t));c=data.range_candidates[r]
  if (h,r) not in scores:
   re=model.rel_proj(rel[r]);score=(ent[h]*re*ent[torch.tensor(c,device='cuda')]).sum(-1).float().cpu().numpy()
   if not np.isfinite(score).all():raise FloatingPointError('InGram score nonfinite')
   scores[h,r]=score
  score=scores[h,r];pos=int(np.searchsorted(c,t));assert c[pos]==t;allow=~np.isin(c,list(data.all_true[h,r]-{t}))
  rank=1+np.sum((score>score[pos])&allow)+.5*max(0,int(np.sum((score==score[pos])&allow))-1);rr.append(1/float(rank))
 return np.asarray(rr,np.float64)

def episodic_ingram_message(data,rng):
 n=len(data.train);order=rng.permutation(n)
 # Keep an incident edge for each training entity and each relation, analogous to
 # the official spanning/relationship coverage split, then fill to 75%.
 shuffled=data.train[order]
 protected=set()
 for col in [0,2,1]:
  _,idx=np.unique(shuffled[:,col],return_index=True);protected.update(map(int,order[idx]))
 size=max(len(protected),int(.75*n));others=[int(i) for i in order if int(i) not in protected]
 chosen=np.asarray(sorted(protected)+others[:size-len(protected)],np.int64)
 sup=np.asarray(others[size-len(protected):],np.int64)
 if not len(sup):raise ValueError('No InGram supervision edges')
 heads=[int(h) for h in np.unique(data.train[:,0]) if data.arrays[1][int(h)].any()]
 fw=np.unique(np.concatenate([data.train[chosen],composition_rows(data,heads)]),axis=0)
 return inverse(fw,data.n_relations+1),data.train[sup]
def ingram_negatives(data,supervision,rng,k):
 # Typed training-only corruptions; reject true training tails.
 pieces=[]
 for _ in range((k+7)//8):pieces.append(sample_negative_tails(data,supervision,rng))
 tails=np.concatenate(pieces,axis=1)[:,:k]
 neg=np.repeat(supervision[:,None,:],k,axis=1);neg[:,:,2]=tails
 return neg.reshape(-1,3)
def train_ingram(source,seed,lr,smoke=False):
 name='ingram';data=load_data(source,P['datasets'][source]);seed_all(seed)
 ckpt,devpath,_,_=paths(source,name,seed,lr)
 if devpath.exists() and ckpt.exists() and not smoke:return json.loads(devpath.read_text(encoding='utf-8'))
 model=make_ingram();spec=P[name];optimizer=torch.optim.Adam(model.parameters(),lr=lr)
 rng=np.random.default_rng(seed)
 dev_cache=ingram_init(data,inverse(graph_forward(data,'dev'),data.n_relations+1),seed+700000)
 history=[];best=-float('inf');stale=0;started=time.perf_counter()
 for epoch in range(1,(1 if smoke else spec['max_epochs'])+1):
  model.train();msg,sup=episodic_ingram_message(data,rng)
  cache=ingram_init(data,msg,seed+1009*epoch)
  if smoke:sup=sup[:32]
  optimizer.zero_grad(set_to_none=True);emb_ent,emb_rel=model(*cache)
  positive=model.score(emb_ent,emb_rel,torch.tensor(sup,device='cuda'))
  neg=ingram_negatives(data,sup,rng,spec['negative_tails'])
  negative=model.score(emb_ent,emb_rel,torch.tensor(neg,device='cuda'))
  loss=F.margin_ranking_loss(positive.repeat_interleave(spec['negative_tails']),negative,torch.ones_like(negative),margin=spec['margin'])
  if not torch.isfinite(loss):raise FloatingPointError('Nonfinite InGram loss')
  loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),.1);optimizer.step()
  lossval=float(loss.detach());del cache,emb_ent,emb_rel,positive,negative,loss
  if epoch==1 or epoch%spec['eval_every']==0:
   rr=ingram_ranks(model,data,dev_cache,'dev',smoke);dev=metrics(rr)
   row={'epoch':epoch,'loss':lossval,'dev':dev,'elapsed':time.perf_counter()-started};history.append(row)
   print('DEV',tag(source,name,seed,lr),row,flush=True)
   if dev['mrr']>best:
    best=dev['mrr'];selected=row;stale=0
    if not smoke:torch.save({'state_dict':cpu_state(model),'protocol_sha256':PROTOCOL_SHA},ckpt)
   else:stale+=1
   if stale>=spec['patience_evaluations']:break
 result={'source':source,'split':P['datasets'][source],'model':name,'seed':seed,'lr':lr,'protocol_sha256':PROTOCOL_SHA,
  'dev':selected['dev'],'selected':selected,'history':history,'parameters':sum(p.numel() for p in model.parameters()),
  'seconds':time.perf_counter()-started,'checkpoint':str(ckpt),'training_budget':spec,'audit':graph_audit(data)}
 if not smoke:save(devpath,result)
 return result
def train_graph(source,name,seed,lr,smoke=False):
 return train_ingram(source,seed,lr,smoke) if name=='ingram' else train_nbf(source,name,seed,lr,smoke)
def test_graph(source,name,seed,lr):
 ckpt,devpath,testpath,rankpath=paths(source,name,seed,lr)
 if testpath.exists() and rankpath.exists():return json.loads(testpath.read_text(encoding='utf-8'))
 info=train_graph(source,name,seed,lr);seed_all(seed);data=load_data(source,P['datasets'][source])
 model=make_ingram() if name=='ingram' else make_graph_model(name)
 model.load_state_dict(torch.load(ckpt,map_location='cuda',weights_only=False)['state_dict'])
 if name=='ingram':
  cache=ingram_init(data,inverse(graph_forward(data,'test'),data.n_relations+1),seed+700000)
  rr=ingram_ranks(model,data,cache,'test')
 else:rr=graph_ranks(model,data,message_graph(source,name,'test'),'test')
 assert len(rr)==len(data.test)
 save_npz(rankpath,rr=rr,heads=data.test[:,0],relations=data.test[:,1])
 result={**info,'test':metrics(rr),'rank_path':str(rankpath)};save(testpath,result)
 print('TEST',source,name,seed,result['test'],flush=True)
 return result

def checkpoint_ingram(model):
 from torch.utils.checkpoint import checkpoint
 for layer in list(model.layers_ent)+list(model.layers_rel):
  original=layer.forward
  def wrapped(*args,_forward=original):
   if torch.is_grad_enabled():return checkpoint(_forward,*args,use_reentrant=False)
   return _forward(*args)
  layer.forward=wrapped
 return model
