"""Isolated extension of the complete paper matrix to the frozen 500/1/15 budget."""
from __future__ import annotations
import argparse,copy,hashlib,json,os,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
BASE=ROOT/'outputs/paper_budget500_20261025'
RAW=BASE/'results'
RAW.mkdir(parents=True,exist_ok=True)
sys.path.insert(0,str(ROOT/'outputs/official_kge_dependencies'))
import numpy as np
import torch
from threadpoolctl import threadpool_limits
import budget500_training as bt
import pair_prototype_full_suite as suite
from pairproto_paper_plan import load_plan
P=json.loads((BASE/'protocol.json').read_text(encoding='utf-8')) if (BASE/'protocol.json').exists() else {}
SHA=hashlib.sha256((BASE/'protocol.json').read_bytes()).hexdigest() if P else None
PLAN=load_plan()
OLD=ROOT/'outputs/comparison_budget500_20261024'
CURRENT_JOB=None

def save(path,obj):
    bt.save(Path(path),obj)

def identity(source,split,name,seed,lr,heads=4):
    return suite.neural_tag(source,str(split),name,int(seed),float(lr),int(heads))

def verify_training(info,expected_sha):
    assert info['protocol_sha256']==expected_sha
    hist=info['history'];stop=bt.EarlyStop(15)
    assert [r['epoch'] for r in hist]==list(range(1,len(hist)+1))
    for row in hist: stop.update(row['dev']['mrr'],row['epoch'])
    assert stop.best_epoch==info['selected']['epoch']
    assert len(hist)<=500 and (len(hist)==500 or stop.stale>=15)
    return info

def load_backbone(data,source,split,name,seed,lr=.003,heads=4):
    tag=identity(source,split,name,seed,lr,heads)
    match=P['reuse'].get(tag)
    if match:
        info=json.loads(Path(match['record']).read_text(encoding='utf-8'))
        verify_training(info,P['comparison_protocol_sha256'])
        assert info['split']==str(split)
        assert info['config']['lr']==lr and info['config'].get('heads',4)==heads
    else:
        bt.P=copy.deepcopy(P['comparison_protocol'])
        bt.P['datasets'][source]=str(split)
        bt.P['neural_configs'][source][name]={'lr':lr,'heads':heads}
        bt.SHA=SHA
        bt.RESULTS=BASE/'training'/tag
        bt.RESULTS.mkdir(parents=True,exist_ok=True)
        info=bt.train_simple(source,'neural',name,seed)
        verify_training(info,SHA)
    model=suite.make_model(data,name,heads).cuda()
    ckpt=torch.load(info['checkpoint'],map_location='cuda',weights_only=False)
    assert ckpt['protocol_sha256']==info['protocol_sha256']
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    alias=copy.deepcopy(info)
    alias.update(config={**info['config'],'lr':lr,'heads':heads,'model':name,'seed':seed},
        parent_protocol_sha256=info['protocol_sha256'],protocol_sha256=SHA,
        reused_current_comparison=bool(match),training_budget=P['training'])
    save(RAW/(tag+'_dev.json'),alias)
    return model,alias

def configure():
    suite.OUT=RAW
    suite.train_or_load=load_backbone
    suite.metrics=bt.metrics
    suite.P=copy.deepcopy(suite.P)
    suite.P['seeds']=P['seeds']
    suite.P['replication_seeds']=P['seeds']
    torch.set_num_threads(2)

def prototype(source,split,seed,lr):
    result=suite.prototype_run(source,split,seed,backbone_lr=lr)
    tag=f'prototype_suite_{source.lower()}_{split}_prototype_seed{seed}'
    if lr!=.003:tag+=f'_backbone_lr{lr:g}'
    path=RAW/(tag+'.json')
    result.update(protocol_sha256=SHA,training_budget=P['training'])
    save(path,result)
    if str(split)==PLAN['datasets'][source]['main'] and lr==P['selected'][source]['mean']['lr']:
        external=OLD/'results'/f'{source.lower()}_proposed_pairproto_seed{seed}.ranks.npz'
        with np.load(RAW/(tag+'_rr.npz')) as a,np.load(external) as b:
            np.testing.assert_allclose(a['pair_prototype'],b['rr'],rtol=0,atol=1e-8)
    return result

def mechanism_module():
    import pairproto_paper_mechanisms as m
    m.OUT=RAW;m.PLAN=copy.deepcopy(PLAN);m.train_or_load=load_backbone
    m.metrics=bt.metrics
    return m

def mechanism_lock(source):
    m=mechanism_module()
    result=m.raw_development_lock(source)
    result['protocol_sha256']=SHA
    save(RAW/f'pairproto_paper_{source.lower()}_raw_score_lock.json',result)

def mechanism(source,seed):
    m=mechanism_module()
    lock=json.loads((RAW/f'pairproto_paper_{source.lower()}_raw_score_lock.json').read_text(encoding='utf-8'))
    assert lock['protocol_sha256']==SHA
    m.one(source,seed,lock)
    p=RAW/f'pairproto_paper_{source.lower()}_mechanisms_seed{seed}.json'
    r=json.loads(p.read_text(encoding='utf-8'));r.update(protocol_sha256=SHA,training_budget=P['training'])
    save(p,r)

def herb_lock(source):
    from pairproto_paper_mechanisms import build
    m=mechanism_module()
    spec=json.loads((ROOT/'pairproto_herb_control_amendment.json').read_text(encoding='utf-8'))
    path=RAW/f'pairproto_herb_control_{source.lower()}_selected.json'
    curves={w:[] for w in spec['weights_in_tie_order']}
    for seed in spec['selection_seeds']:
        data,model,info=m.build(source,seed)
        cache=suite.profile_cache(model,data,data.dev)
        for weight in curves:
            curves[weight].append(float(suite.weighted_ranks(data,data.dev,cache,(1-weight,weight,0)).mean()))
        del model
        torch.cuda.empty_cache()
    candidates=[{'weight':w,'development_mrr':float(np.mean(vals)),'per_seed':vals} for w,vals in curves.items()]
    save(path,{'source':source,'protocol_sha256':SHA,'amendment_sha256':hashlib.sha256((ROOT/'pairproto_herb_control_amendment.json').read_bytes()).hexdigest(),
        'candidates':candidates,'chosen':max(candidates,key=lambda r:r['development_mrr']),
        'selection_seeds':spec['selection_seeds'],'selection_partition':'development'})

def herb(source,seed):
    m=mechanism_module()
    selected=json.loads((RAW/f'pairproto_herb_control_{source.lower()}_selected.json').read_text(encoding='utf-8'))
    assert selected['protocol_sha256']==SHA
    weight=selected['chosen']['weight']
    data,model,info=m.build(source,seed);model.weights=(weight,0)
    rr=suite.weighted_ranks(data,data.test,suite.profile_cache(model,data,data.test),(1-weight,weight,0))
    np.testing.assert_allclose(rr,suite.reciprocal_ranks(model,data,data.test),rtol=0,atol=1e-8)
    tag=f'pairproto_herb_control_{source.lower()}_seed{seed}'
    suite.save_npz(RAW/(tag+'_rr.npz'),rr=rr,heads=data.test[:,0],relations=data.test[:,1])
    save(RAW/(tag+'.json'),{'source':source,'seed':seed,'checkpoint':info['checkpoint'],
        'selected_weight':weight,'protocol_sha256':SHA,'test':bt.metrics(rr)})

def sensitivity(source,seed):
    lr=P['selected'][source]['mean']['lr'];split=PLAN['datasets'][source]['main']
    report=suite.prototype_run(source,split,seed,True,lr)
    tag=f'prototype_suite_{source.lower()}_{split}_prototype_seed{seed}'
    if lr!=.003:tag+=f'_backbone_lr{lr:g}'
    report.update(protocol_sha256=SHA,training_budget=P['training'])
    save(RAW/(tag+'_sensitivity.json'),report)

def boundary(source,seed):
    split=PLAN['datasets'][source]['main'];lr=P['selected'][source]['mean']['lr']
    data=suite.load_data(source,split);base,info=load_backbone(data,source,split,'mean',seed,lr)
    gamma=PLAN['prototype'][source+'_pair_weight']
    model=suite.PairPrototypeKGE(data,base,weights=(0,gamma),relations=np.unique(np.concatenate((data.dev[:,1],data.test[:,1]))))
    mean=suite.reciprocal_ranks(base,data,data.test,all_entities=True)
    proposal=suite.reciprocal_ranks(model,data,data.test,all_entities=True)
    tag=f'pairproto_paper_{source.lower()}_boundary_seed{seed}'
    suite.save_npz(RAW/(tag+'_rr.npz'),mean=mean,pairproto=proposal,heads=data.test[:,0],relations=data.test[:,1])
    save(RAW/(tag+'.json'),{'source':source,'split':split,'seed':seed,'protocol_sha256':SHA,
        'training_budget':P['training'],'checkpoint':info['checkpoint'],
        'role':'All-entity ranking of typed-negative-trained models; no all-entity-negative retraining',
        'mean':bt.metrics(mean),'pairproto':bt.metrics(proposal)})

def measurements():
    import pairproto_paper_cases_costs as cases
    import platform
    cases.OUT=RAW;cases.PLAN=copy.deepcopy(PLAN);cases.train_or_load=load_backbone;cases.metrics=bt.metrics
    report={'protocol_sha256':SHA,'hardware':{'python':platform.python_version(),'torch':torch.__version__,
        'gpu':torch.cuda.get_device_name(0),'cuda':torch.version.cuda,'cpu_threads':2},
        'scope':'One-backbone engineering measurements and development-selected illustrative cases, not five independent timing claims','sources':{}}
    for source in PLAN['datasets']:
        report['sources'][source]=cases.one(source)
        save(RAW/'pairproto_paper_cases_costs.json',report)
    report['complete']=True;save(RAW/'pairproto_paper_cases_costs.json',report)
    import pairproto_moment_collision as collision
    witness=collision.experiment()
    witness['distinct_nonzero_embedding_witness']=collision.distinct_embedding_experiment()
    witness.update(protocol_sha256=SHA,role='Deterministic fixed-embedding mathematical witnesses; not a stochastic training comparison')
    save(RAW/'pairproto_moment_collision.json',witness)

def execute(job):
    kind=job['kind'];source=job.get('source');seed=job.get('seed')
    if kind in ('train','dev_audit'):
        split=job['split'];name=job['model'];lr=job['lr'];heads=job['heads']
        if kind=='train':
            info=suite.run_neural(source,split,name,seed,lr,heads)
            tag=identity(source,split,name,seed,lr,heads)
            path=RAW/(tag+'_test.json')
            dev=json.loads((RAW/(tag+'_dev.json')).read_text(encoding='utf-8'))
            info.update(protocol_sha256=SHA,training_budget=P['training'],best_epoch=dev['selected']['epoch'],stop_epoch=dev['stop_epoch'],
                training_parent_protocol_sha256=dev['parent_protocol_sha256'],reused_current_comparison=dev['reused_current_comparison'])
            save(path,info)
        else:
            data=suite.load_data(source,split)
            _,info=load_backbone(data,source,split,name,seed,lr,heads)
            save(BASE/'development_audit'/(job['id']+'.json'),{'protocol_sha256':SHA,'job':job,'dev':info['dev'],
                'checkpoint':info['checkpoint'],'selected_epoch':info['selected']['epoch'],
                'role':'Development-only sensitivity with frozen main configuration; does not choose a new main model'})
    elif kind=='prototype':prototype(source,job['split'],seed,job['lr'])
    elif kind=='mechanism_lock':mechanism_lock(source)
    elif kind=='mechanism':mechanism(source,seed)
    elif kind=='herb_lock':herb_lock(source)
    elif kind=='herb':herb(source,seed)
    elif kind=='sensitivity':sensitivity(source,seed)
    elif kind=='boundary':boundary(source,seed)
    elif kind=='measurements':measurements()
    else:raise ValueError(kind)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--job',required=True);parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args()
    for name,digest in P['code_sha256'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,'Frozen implementation changed: '+name
    job=next(row for row in P['jobs'] if row['id']==args.job)
    configure()
    with threadpool_limits(limits=2):
        if args.smoke:
            source=job['source'];split=job['split'];name=job['model'];seed=job['seed']
            bt.P=copy.deepcopy(P['comparison_protocol']);bt.P['datasets'][source]=split
            bt.P['neural_configs'][source][name]={'lr':job['lr'],'heads':job['heads']}
            bt.smoke(source,'neural',name,seed)
            return
        execute(job)
    marker=BASE/'completed'/(job['id']+'.json')
    save(marker,{'protocol_sha256':SHA,'job':job,'completed_local':time.strftime('%Y-%m-%d %H:%M:%S')})
    print('PAPER JOB COMPLETE',job['id'],flush=True)

if __name__=='__main__':main()
