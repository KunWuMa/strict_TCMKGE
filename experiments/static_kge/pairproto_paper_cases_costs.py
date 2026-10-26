"""Selected-backbone engineering, all-entity boundary and development cases.

No model setting is selected here. Test strata are descriptive; case selection
uses development queries only. Latency excludes prototype construction and is
reported separately for cold composition and an already cached composition.
"""
from __future__ import annotations

import itertools
import json
import platform
import time
from pathlib import Path

import numpy as np
from scipy import sparse
import torch
from threadpoolctl import threadpool_limits

from pair_memory_dev import ingredients,standardized
from pair_prototype_dev import tail_features
from pair_prototype_full_suite import (OUT, load_data, train_or_load,
    reciprocal_ranks, metrics, save, profile_cache, weighted_ranks,save_npz)
from pair_prototype_kge import PairPrototypeKGE

HERE=Path(__file__).resolve().parent
PLAN=json.loads((HERE/'pairproto_paper_protocol.json').read_text(encoding='utf-8'))


def nearest_recipe_similarity(data):
    """Maximum exact-set Jaccard to any training formula (including context-only)."""
    train=sorted({ingredients(data,h) for h in np.unique(data.train[:,[0,2]])
                  if data.arrays[1][int(h)].any()})
    heads=np.unique(data.test[:,0]);test=[ingredients(data,h) for h in heads]
    vocabulary={h:i for i,h in enumerate(sorted({h for recipe in train+test for h in recipe}))}
    def matrix(recipes):
        row=[];col=[]
        for i,recipe in enumerate(recipes):
            row.extend([i]*len(recipe));col.extend(vocabulary[h] for h in recipe)
        return sparse.csr_matrix((np.ones(len(row),dtype=np.int32),(row,col)),
                                 shape=(len(recipes),len(vocabulary)))
    tr=matrix(train);te=matrix(test);tr_len=np.asarray(tr.sum(1)).ravel()
    values=[]
    for start in range(0,len(test),128):
        count=(te[start:start+128]@tr.T).toarray()
        sizes=np.asarray(te[start:start+128].sum(1)).ravel()
        union=sizes[:,None]+tr_len[None,:]-count
        values.extend(np.max(count/np.maximum(union,1),axis=1).tolist())
    by_head=dict(zip(heads,values))
    return np.asarray([by_head[h] for h in data.test[:,0]])


def timing(model,data,raw_weight):
    pairs=list(dict.fromkeys((int(h),int(r)) for h,r,_ in data.test))[:100]
    records={}
    for mode in ('mean','prototype_cold','prototype_warm','raw_pair_cold'):
        samples=[]
        for repeat in range(3):
            for h,r in pairs:
                herbs=ingredients(data,h);tails=data.range_candidates[r]
                ht=torch.tensor(h,device='cuda');rt=torch.tensor(r,device='cuda')
                tt=torch.as_tensor(tails,device='cuda')
                if mode=='prototype_cold':model.cache.clear()
                if mode=='prototype_warm':model.compose(herbs,r)
                torch.cuda.synchronize();started=time.perf_counter()
                with torch.inference_mode():
                    if mode=='mean':model.base.score_tails(ht,rt,tt)
                    elif mode.startswith('prototype'):model.score_tails(ht,rt,tt)
                    else:
                        memory=model.memories[r]
                        ids=torch.as_tensor(herbs,device='cuda')
                        mu=torch.cat((model.base.er(ids).mean(0),model.base.ei(ids).mean(0))).cpu().numpy().astype(float)
                        mean=standardized(model.phi[r]@mu)
                        pids=[memory.pair_vocab[p] for p in itertools.combinations(herbs,2) if p in memory.pair_vocab]
                        profiles=memory.lift(memory.pair_counts,pids,memory.pair_n)
                        pair=profiles.mean(0) if len(profiles) else np.zeros(len(tails))
                        scores=(1-raw_weight)*mean+raw_weight*standardized(pair)
                torch.cuda.synchronize();samples.append(1000*(time.perf_counter()-started))
        records[mode]={'median_ms':float(np.median(samples)),
                       'p95_ms':float(np.percentile(samples,95)),
                       'observations':len(samples)}
    records['policy']='First 100 distinct (formula, relation) test queries, three repeats; full-catalogue score generation, excluding filtering and sorting, CUDA synchronized. Raw pair scores compute the same centered float64 Mean component and raw pair lift used in evaluation, with the dev-selected raw pair mixture; unused herb, top-k and residual profiles are not computed. Cache handling is declared separately.'
    records['raw_pair_mixture']=raw_weight
    return records


def decompose(model,data,h,r,t,gamma):
    herbs=ingredients(data,h);memory=model.memories[r]
    pairs=[p for p in itertools.combinations(herbs,2) if p in memory.pair_vocab]
    ids=[memory.pair_vocab[p] for p in pairs]
    pair=model.prototypes[r]['pair'][ids].mean(0) if ids else np.zeros(256)
    with torch.inference_mode():
        idx=torch.as_tensor(herbs,device='cuda')
        mu=torch.cat((model.base.er(idx).mean(0),model.base.ei(idx).mean(0))).cpu().numpy().astype(float)
    phi=tail_features(model.base,r,np.array([t]))[0]
    sm=max(float(np.std(model.phi[r]@mu)),1e-8)
    sp=max(float(np.std(model.phi[r]@pair)),1e-8)
    mean_term=float((1-gamma)*(mu@phi)/sm)
    terms=gamma*(model.prototypes[r]['pair'][ids]@phi)/(max(1,len(ids))*sp)
    with torch.inference_mode():
        actual=float(model.score(torch.tensor([h],device='cuda'),torch.tensor([r],device='cuda'),torch.tensor([t],device='cuda')))
    reconstructed=mean_term+float(terms.sum())
    if not np.isclose(actual,reconstructed,rtol=2e-4,atol=2e-4):
        raise AssertionError(('Additive score reconstruction',actual,reconstructed))
    order=np.argsort(-np.abs(terms));entries=[]
    for i in order[:8]:
        remaining=np.delete(model.prototypes[r]['pair'][ids],i,axis=0)
        removed=remaining.mean(0) if len(remaining) else np.zeros(256)
        removed_scale=max(float(np.std(model.phi[r]@removed)),1e-8)
        removed_score=mean_term+float(gamma*(removed@phi)/removed_scale)
        entries.append({'herb_ids':list(pairs[i]),'herbs':[str(data.entities[x]) for x in pairs[i]],
                        'training_recipe_support':int(memory.pair_n[ids[i]]),'contribution':float(terms[i]),
                        'score_after_prototype_removal_and_renormalization':removed_score,
                        'removal_score_difference':removed_score-actual})
    return {'head':h,'relation_id':r,'target_id':t,'relation':str(data.relations[r]),'target':str(data.entities[t]),
            'formula_herb_count':len(herbs),'supported_pairs':len(pairs),'mean_term':mean_term,
            'pair_term_total':float(terms.sum()),'score':actual,'reconstructed_score':reconstructed,
            'absolute_reconstruction_error':abs(actual-reconstructed),'top_pairs':entries,
            'unshown_pair_contribution':float(terms.sum()-sum(x['contribution'] for x in entries))}


def development_cases(model,data,gamma):
    cache=profile_cache(model,data,data.dev)
    baseline=weighted_ranks(data,data.dev,cache,(1,0,0))
    proposal=weighted_ranks(data,data.dev,cache,(1-gamma,0,gamma))
    eligible=[]
    for i,(h,r,_) in enumerate(data.dev):
        n=sum(p in model.memories[int(r)].pair_vocab for p in itertools.combinations(ingredients(data,h),2))
        if n>=3:eligible.append(i)
    if not eligible:return []
    result=[];delta=proposal-baseline
    for label,index in [('largest_gain',max(eligible,key=lambda i:delta[i])),
                        ('largest_loss',min(eligible,key=lambda i:delta[i]))]:
        h,r,t=map(int,data.dev[index]);row=decompose(model,data,h,r,t,gamma)
        row.update(selection=label,split='development',mean_rr=float(baseline[index]),
                   prototype_rr=float(proposal[index]),rr_difference=float(delta[index]))
        result.append(row)
    return result


def one(source):
    split=PLAN['datasets'][source]['main'];seed=PLAN['main_seeds'][0]
    selected=json.loads((OUT/f'pairproto_paper_{source.lower()}_selected_configs.json').read_text(encoding='utf-8'))
    lr=selected['models']['mean']['chosen']['lr'];data=load_data(source,split)
    base,info=train_or_load(data,source,split,'mean',seed,lr)
    gamma=PLAN['prototype'][source+'_pair_weight']
    raw_weight=json.loads((OUT/f'pairproto_paper_{source.lower()}_raw_score_lock.json').read_text(encoding='utf-8'))['selected']['pair']
    model=PairPrototypeKGE(data,base,weights=(0,gamma),relations=np.unique(np.concatenate((data.dev[:,1],data.test[:,1]))))
    trainable_bytes=sum(p.numel()*p.element_size() for p in base.parameters())
    pair_bytes=sum(v['pair'].nbytes for v in model.prototypes.values())
    herb_bytes=sum(v['herb'].nbytes for v in model.prototypes.values())
    count_bytes=sum(x.data.nbytes+x.indices.nbytes+x.indptr.nbytes for m in model.memories.values() for x in (m.herb_counts,m.pair_counts))
    phi_bytes=sum(x.nbytes for x in model.phi.values())
    report={'source':source,'split':split,'seed':seed,'backbone_lr':lr,'checkpoint':info['checkpoint'],
            'base_parameters':sum(p.numel() for p in base.parameters()),'base_parameter_bytes':trainable_bytes,
            'prototype_bytes_all':model.prototype_bytes,'pair_prototype_bytes':pair_bytes,
            'herb_control_prototype_bytes':herb_bytes,'sparse_count_bytes':count_bytes,
            'centered_candidate_feature_bytes':phi_bytes,'prototype_build_seconds':model.build_seconds,
            'storage_note':'Array payloads only; Python dictionary/tuple/index overhead, allocator overhead and cached formula compositions are excluded. PairProto deployment does not require herb-control vectors or raw count matrices after construction.',
            'latency':timing(model,data,raw_weight),'cases':development_cases(model,data,gamma)}
    all_mean=reciprocal_ranks(base,data,data.test,all_entities=True)
    all_proposal=reciprocal_ranks(model,data,data.test,all_entities=True)
    report['all_entity_boundary']={'mean':metrics(all_mean),'pairproto':metrics(all_proposal),
        'role':'One-seed diagnostic using the same typed-negative-trained backbone and the same retained queries; no all-entity-negative retraining, no claim of unrestricted KGE superiority. Normalization remains fixed to training relation ranges.'}
    save_npz(OUT/f'pairproto_paper_{source.lower()}_boundary_rr.npz',mean=all_mean,pairproto=all_proposal,
                        heads=data.test[:,0],relations=data.test[:,1])
    similarity=nearest_recipe_similarity(data)
    if (similarity>=1).any():raise AssertionError('Exact training composition in strict test')
    support=[];coverage=[];length=[]
    for h,r,t in data.test:
        memory=model.memories[int(r)];pairs=list(itertools.combinations(ingredients(data,h),2))
        covered=sum(p in memory.pair_vocab for p in pairs)
        coverage.append(covered/max(1,len(pairs)));length.append(len(ingredients(data,h)))
        pos=np.searchsorted(memory.candidates,int(t));support.append(int(memory.target_recipe_counts[pos]))
    save_npz(OUT/f'pairproto_paper_{source.lower()}_strata.npz',heads=data.test[:,0],relations=data.test[:,1],
                        nearest_jaccard=similarity,target_recipe_support=np.asarray(support),
                        coverage=np.asarray(coverage),lengths=np.asarray(length))
    report['strata']={'nearest_training_jaccard_median':float(np.median(similarity)),
                      'nearest_training_jaccard_max':float(similarity.max()),
                      'test_queries':len(similarity),'policy':'Fixed descriptive bins only; no setting is selected from these test strata.'}
    return report


def measurements():
    torch.set_num_threads(4)
    path=OUT/'pairproto_paper_cases_costs.json'
    report=json.loads(path.read_text(encoding='utf-8')) if path.exists() else {'sources':{}}
    report['hardware']={'python':platform.python_version(),'torch':torch.__version__,
        'gpu':torch.cuda.get_device_name(0),'cuda':torch.version.cuda,'cpu_threads':4}
    with threadpool_limits(limits=4):
        for source in PLAN['datasets']:
            artifacts=[OUT/f'pairproto_paper_{source.lower()}_{kind}.npz' for kind in ('boundary_rr','strata')]
            if source not in report['sources'] or not all(path.exists() for path in artifacts):
                report['sources'][source]=one(source);save(path,report)
    report['complete']=len(report['sources'])==len(PLAN['datasets']);save(path,report)
    print('CASES COSTS COMPLETE',flush=True)


def main():
    from pairproto_cache_lock import cache_lock
    # Timing starts only after any optional auxiliary training worker exits.
    with cache_lock('auxiliary_training'),cache_lock('pairproto_herb_control'),cache_lock('pairproto_full_deepsets'):measurements()


if __name__=='__main__':main()
