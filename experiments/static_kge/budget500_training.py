"""Fresh, isolated 500-round comparisons; reuse the existing model and ranking implementations."""
from __future__ import annotations
import argparse,copy,hashlib,json,os,random,sys,time,gc
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'outputs/official_kge_dependencies'))
import numpy as np
import torch
from torch.nn import functional as F
from threadpoolctl import threadpool_limits
import pair_prototype_full_suite as suite
import extended_kge_baselines as extra

RUN=ROOT/'outputs/comparison_budget500_20261024'
PROTOCOL=RUN/'protocol.json'
P=json.loads(PROTOCOL.read_text(encoding='utf-8'))
SHA=hashlib.sha256(PROTOCOL.read_bytes()).hexdigest()
RESULTS=RUN/'results'
RESULTS.mkdir(parents=True,exist_ok=True)
torch.set_num_threads(2)

def save(path,value):
    temporary=path.with_name(path.name+f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
    temporary.replace(path)

def stem(source,family,name,seed):
    return RESULTS/f'{source.lower()}_{family}_{name}_seed{seed}'

def metrics(rr):
    rr=np.asarray(rr,dtype=np.float64)
    return {'mr':float(np.mean(1/rr)),'mrr':float(rr.mean()),
            **{f'hits{k}':float(np.mean(rr>=1/k)) for k in [1,3,5,10]}}

class EarlyStop:
    def __init__(self,patience=15):
        self.best=-float('inf');self.best_epoch=None;self.stale=0;self.patience=patience
    def update(self,value,epoch):
        improved=value>self.best
        if improved:self.best=value;self.best_epoch=epoch;self.stale=0
        else:self.stale+=1
        return improved,self.stale>=self.patience

def configure_graph():
    extra.P=copy.deepcopy(P['extended_architectures'])
    for name in ['classic','nodepiece','ingram','ultra','trix']:
        extra.P[name].update({k:P['training'][k] for k in ['max_epochs','eval_every','patience_evaluations']})
    extra.OUT=RESULTS;extra.PROTOCOL_SHA=SHA
    import extended_graph_baselines as graph
    graph.P=extra.P;graph.OUT=RESULTS;graph.PROTOCOL_SHA=SHA
    return graph

def configuration(source,family,name):
    return P['neural_configs' if family=='neural' else 'extended_configs'][source][name]

def check(record):
    assert record['protocol_sha256']==SHA,'Protocol mismatch; refuse stale cache'
    history=record.get('history',[])
    if history:
        assert [h['epoch'] for h in history]==list(range(1,history[-1]['epoch']+1))
        assert history[-1]['epoch']<=500
        stop=EarlyStop()
        for h in history:stop.update(h['dev']['mrr'],h['epoch'])
        assert record['selected']['epoch']==stop.best_epoch
        assert history[-1]['epoch']==500 or stop.stale>=15
    return record

def simple_model(data,source,family,name,seed,heads):
    if family=='neural':return suite.make_model(data,name,heads).cuda()
    return (extra.OfficialNodePiece(data,source,seed) if name=='nodepiece' else extra.MeanRealKGE(data,name)).cuda()

def simple_loss(model,data,raw,neg,family):
    micro=128 if getattr(model,'_budget_method','')=='set_transformer' else 512
    total=0.
    for begin in range(0,len(raw),micro):
        part=raw[begin:begin+micro];batch=torch.as_tensor(part,device='cuda')
        tails=torch.as_tensor(neg[begin:begin+len(part)],device='cuda')
        if family=='extended':
            score=model.bundle(batch[:,0],batch[:,1],torch.cat((batch[:,2:3],tails),dim=1))
            positive,negative=score[:,0],score[:,1:]
        elif isinstance(model,suite.CompositionalKGE):
            positive=model.score(batch[:,0],batch[:,1],batch[:,2])
            negative=model.score_negative_tails(batch[:,0],batch[:,1],tails)
        elif hasattr(model,'head_embedding'):
            bundle=suite.score_bundle(model,batch[:,0],batch[:,1],torch.cat((batch[:,2:3],tails),dim=1))
            positive,negative=bundle[:,0],bundle[:,1:]
        else:
            positive=model.score(batch[:,0],batch[:,1],batch[:,2])
            negative=model.score(batch[:,0,None].expand_as(tails).reshape(-1),
              batch[:,1,None].expand_as(tails).reshape(-1),tails.reshape(-1)).reshape_as(tails)
        loss=F.softplus(-positive).mean()+F.softplus(negative).mean()
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite training loss')
        weight=len(part)/len(raw);(loss*weight).backward();total+=float(loss.detach())*weight
    return total

def simple_ranks(model,data,family,phase):
    return suite.reciprocal_ranks(model,data,getattr(data,phase)) if family=='neural' else extra.embedding_ranks(model,data,phase)

def train_simple(source,family,name,seed):
    base=stem(source,family,name,seed);record_path=base.with_suffix('.train.json');checkpoint=base.with_suffix('.pt')
    if record_path.exists() and checkpoint.exists():
        return check(json.loads(record_path.read_text(encoding='utf-8')))
    cfg=configuration(source,family,name);data=suite.load_data(source,P['datasets'][source])
    extra.seed_all(seed)
    model=simple_model(data,source,family,name,seed,cfg.get('heads',4));model._budget_method=name
    initial=hashlib.sha256()
    for parameter in model.parameters():initial.update(parameter.detach().flatten()[:128].cpu().numpy().tobytes())
    optimizer=torch.optim.Adam(model.parameters(),lr=cfg['lr'],weight_decay=1e-7)
    rng=np.random.default_rng(seed);early=EarlyStop(P['training']['patience_evaluations'])
    history=[];started=time.perf_counter();torch.cuda.reset_peak_memory_stats()
    for epoch in range(1,P['training']['max_epochs']+1):
        model.train();losses=[];order=rng.permutation(len(data.train))
        for offset in range(0,len(order),2048):
            raw=data.train[order[offset:offset+2048]]
            negatives=suite.sample_negative_tails(data,raw,rng)
            optimizer.zero_grad(set_to_none=True)
            losses.append(simple_loss(model,data,raw,negatives,family));optimizer.step()
        dev=metrics(simple_ranks(model,data,family,'dev'))
        row={'epoch':epoch,'loss':float(np.mean(losses)),'dev':dev,'elapsed':time.perf_counter()-started}
        history.append(row);improved,finished=early.update(dev['mrr'],epoch)
        if improved:
            selected=copy.deepcopy(row)
            torch.save({'state_dict':extra.cpu_state(model),'protocol_sha256':SHA,
                        'selected_epoch':epoch,'fresh_initialization':True},checkpoint)
        save(base.with_suffix('.progress.json'),{'protocol_sha256':SHA,'phase':'training','latest':row,
             'selected':selected,'stale_evaluations':early.stale,'patience_evaluations':15})
        print('DEV',source,family,name,seed,row,'stale',early.stale,flush=True)
        if finished:break
    info={'source':source,'split':P['datasets'][source],'family':family,'model':name,'seed':seed,
          'config':cfg,'protocol_sha256':SHA,'training_budget':P['training'],'history':history,
          'selected':selected,'dev':selected['dev'],'stop_epoch':epoch,'stale_evaluations':early.stale,
          'stop_reason':'patience' if finished else 'max_epochs','fresh_initialization':True,
          'initial_parameter_sample_sha256':initial.hexdigest(),
          'parameters':sum(p.numel() for p in model.parameters()),'seconds':time.perf_counter()-started,
          'peak_gpu_allocated_bytes':torch.cuda.max_memory_allocated(),
          'peak_gpu_reserved_bytes':torch.cuda.max_memory_reserved(),'checkpoint':str(checkpoint)}
    check(info);save(record_path,info)
    return info

def save_evaluation(source,family,name,seed,data,rr,info):
    base=stem(source,family,name,seed);path=base.with_suffix('.ranks.npz')
    assert len(rr)==len(data.test) and np.isfinite(rr).all() and np.all((rr>0)&(rr<=1))
    suite.save_npz(path,rr=rr,heads=data.test[:,0],relations=data.test[:,1])
    record={**info,'source':source,'split':P['datasets'][source],'family':family,'model':name,
            'seed':seed,'protocol_sha256':SHA,'test':metrics(rr),'rank_path':str(path)}
    save(base.with_suffix('.evaluation.json'),record)
    print('TEST',source,family,name,seed,record['test'],flush=True)
    return record

def evaluate_proposed(source,seed,data,backbone,backbone_info):
    dest=stem(source,'proposed','pairproto',seed).with_suffix('.evaluation.json')
    if dest.exists():return check(json.loads(dest.read_text(encoding='utf-8')))
    spec=P['prototype_config'];started=time.perf_counter()
    model=suite.PairPrototypeKGE(data,backbone,relations=P['target_relations'][source],
           ridge=spec['ridge_relative'],shrinkage=spec['shrinkage'],min_support=spec['minimum_pair_support'])
    cache=suite.profile_cache(model,data,data.test)
    hw,pw=spec['weights'][source]
    rr=suite.weighted_ranks(data,data.test,cache,(1-hw-pw,hw,pw))
    info={'training_budget':P['training'],'history':backbone_info['history'],
          'selected':backbone_info['selected'],'backbone_checkpoint':backbone_info['checkpoint'],
          'backbone_stop_epoch':backbone_info['history'][-1]['epoch'],'prototype_config':spec,
          'prototype_build_seconds':model.build_seconds,'evaluation_seconds':time.perf_counter()-started,
          'note':'Shared fresh Mean-ComplEx backbone; no additional gradient epochs'}
    return save_evaluation(source,'proposed','pairproto',seed,data,rr,info)

def run_simple(source,family,name,seed):
    configure_graph()
    destination=stem(source,family,name,seed).with_suffix('.evaluation.json')
    info=train_simple(source,family,name,seed)
    if destination.exists() and not(family=='neural' and name=='mean'):
        return check(json.loads(destination.read_text(encoding='utf-8')))
    data=suite.load_data(source,P['datasets'][source]);cfg=configuration(source,family,name)
    extra.seed_all(seed);model=simple_model(data,source,family,name,seed,cfg.get('heads',4))
    saved=torch.load(info['checkpoint'],map_location='cuda',weights_only=False);assert saved['protocol_sha256']==SHA
    model.load_state_dict(saved['state_dict']);model.eval()
    if not destination.exists():save_evaluation(source,family,name,seed,data,simple_ranks(model,data,family,'test'),info)
    if family=='neural' and name=='mean':evaluate_proposed(source,seed,data,model,info)
    return info

def run_graph(source,name,seed):
    graph=configure_graph();base=stem(source,'extended',name,seed)
    destination=base.with_suffix('.evaluation.json')
    if destination.exists():return check(json.loads(destination.read_text(encoding='utf-8')))
    import builtins
    tracker=EarlyStop()
    def log(*args,**kwargs):
        builtins.print(*args,**kwargs)
        if args and args[0]=='DEV' and isinstance(args[-1],dict):
            row=args[-1];tracker.update(row['dev']['mrr'],row['epoch'])
            save(base.with_suffix('.progress.json'),{'protocol_sha256':SHA,'phase':'training',
                 'latest':row,'best_epoch':tracker.best_epoch,'best_dev_mrr':tracker.best,
                 'stale_evaluations':tracker.stale,'patience_evaluations':15})
    graph.print=log
    cfg=configuration(source,'extended',name)
    torch.cuda.reset_peak_memory_stats()
    record=graph.test_graph(source,name,seed,cfg['lr'])
    check(record);data=suite.load_data(source,P['datasets'][source])
    with np.load(record['rank_path']) as arrays:
        assert np.array_equal(arrays['heads'],data.test[:,0])
        assert np.array_equal(arrays['relations'],data.test[:,1])
        rr=arrays['rr'].copy()
    record.update({'fresh_initialization':True,'training_schedule':P['training'],
       'peak_gpu_allocated_bytes':torch.cuda.max_memory_allocated(),
       'peak_gpu_reserved_bytes':torch.cuda.max_memory_reserved()})
    return save_evaluation(source,'extended',name,seed,data,rr,record)

def smoke(source,family,name,seed):
    """Check one real optimizer step and finite development scores; no scientific artifact writes."""
    graph=configure_graph()
    if family=='extended' and name in ['ingram','ultra','trix']:
        record=graph.train_graph(source,name,seed,configuration(source,family,name)['lr'],smoke=True)
        assert record['history'][0]['epoch']==1 and np.isfinite(record['dev']['mrr'])
    else:
        data=suite.load_data(source,P['datasets'][source]);cfg=configuration(source,family,name)
        extra.seed_all(seed);model=simple_model(data,source,family,name,seed,cfg.get('heads',4))
        model._budget_method=name
        optimizer=torch.optim.Adam(model.parameters(),lr=cfg['lr'],weight_decay=1e-7)
        raw=data.train[:16];negative=suite.sample_negative_tails(data,raw,np.random.default_rng(seed))
        optimizer.zero_grad(set_to_none=True);loss=simple_loss(model,data,raw,negative,family)
        assert np.isfinite(loss);optimizer.step()
        rr=simple_ranks(model,data,family,'dev')[:16]
        assert np.isfinite(rr).all() and np.all((rr>0)&(rr<=1))
    print('SMOKE PASS',source,family,name,seed,flush=True)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',required=True,choices=list(P['datasets']))
    parser.add_argument('--family',required=True,choices=['neural','extended'])
    parser.add_argument('--model',required=True);parser.add_argument('--seed',type=int,required=True)
    parser.add_argument('--smoke',action='store_true');args=parser.parse_args()
    assert args.seed in P['seeds']
    assert args.model in P['neural_models' if args.family=='neural' else 'extended_models']
    for filename,digest in P['code_sha256'].items():
        assert hashlib.sha256((ROOT/filename).read_bytes()).hexdigest()==digest,'Frozen implementation changed: '+filename
    with threadpool_limits(limits=2):
        if args.smoke:smoke(args.source,args.family,args.model,args.seed)
        elif args.family=='extended' and args.model in ['ingram','ultra','trix']:
            run_graph(args.source,args.model,args.seed)
        else:run_simple(args.source,args.family,args.model,args.seed)

if __name__=='__main__':main()
