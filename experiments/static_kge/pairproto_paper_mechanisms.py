"""Mechanism controls for PairProto-KGE; all memories are training-derived.

The raw-score comparator receives a declared dev-only weight search. Shuffles
preserve memory entries; shared prototypes deliberately tie effective capacity.
"""
from __future__ import annotations

import argparse
import copy
import itertools
import json
from pathlib import Path

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from pair_memory_dev import ingredients, standardized
from pair_prototype_kge import PairPrototypeKGE
from pair_prototype_full_suite import (OUT, load_data, train_or_load, profile_cache,
                                      weighted_ranks, metrics, save,save_npz)

HERE=Path(__file__).resolve().parent
from pairproto_paper_plan import load_plan
PLAN=load_plan()


def chosen_lr(source):
    path=OUT/f"pairproto_paper_{source.lower()}_selected_configs.json"
    return json.loads(path.read_text(encoding="utf-8"))["models"]["mean"]["chosen"]["lr"]


def build(source,seed,backbone_lr=None):
    split=PLAN["datasets"][source]["main"]
    data=load_data(source,split)
    base,info=train_or_load(data,source,split,"mean",seed,chosen_lr(source) if backbone_lr is None else backbone_lr)
    relations=np.unique(np.concatenate((data.dev[:,1],data.test[:,1])))
    model=PairPrototypeKGE(data,base,relations=relations)
    return data,model,info


def raw_development_lock(source):
    path=OUT/f"pairproto_paper_{source.lower()}_raw_score_lock.json"
    if path.exists():return json.loads(path.read_text(encoding="utf-8"))
    curves={name:{str(w):[] for w in PLAN["raw_score_weight_grid"]} for name in ("herb","pair")}
    for seed in PLAN.get('selection_seeds',PLAN["ancillary_seeds"]):
        data,model,_=build(source,seed)
        cache=profile_cache(model,data,data.dev,raw=True)
        for name in curves:
            for weight in PLAN["raw_score_weight_grid"]:
                weights=(1-weight,weight,0) if name=="herb" else (1-weight,0,weight)
                rr=weighted_ranks(data,data.dev,cache,weights)
                curves[name][str(weight)].append(float(rr.mean()))
    selected={name:float(max(values,key=lambda w:np.mean(values[w]))) for name,values in curves.items()}
    report={"role":"Raw-profile comparator weights chosen on development only, before scoring these controls on test",
            "source":source,"split":PLAN["datasets"][source]["main"],"backbone_lr":chosen_lr(source),
            "curves":curves,"selected":selected,"seeds":PLAN.get('selection_seeds',PLAN["ancillary_seeds"])}
    save(path,report);print("RAW CONTROL LOCK",source,selected,flush=True)
    return report


def pair_cache(model,data,triples,base_cache):
    result={}
    for h,r,_ in triples:
        key=int(h),int(r)
        if key in result:continue
        herbs=ingredients(data,h);memory=model.memories[int(r)]
        ids=[memory.pair_vocab[p] for p in itertools.combinations(herbs,2) if p in memory.pair_vocab]
        vector=model.prototypes[int(r)]["pair"][ids].mean(0) if ids else np.zeros(256)
        pair=standardized(model.phi[int(r)]@vector)
        result[key]=(base_cache[key][0],base_cache[key][1],pair)
    return result


def shuffle_pair_identity(model,original,rng):
    eligible=0;total=0
    for r,memory in model.memories.items():
        values=model.prototypes[r]['pair'];support=memory.pair_n.astype(int)
        total+=len(values)
        for level in np.unique(support):
            ids=np.flatnonzero(support==level)
            if len(ids)>1:
                order=rng.permutation(ids);shift=int(rng.integers(1,len(ids)))
                values[order]=original[r]['pair'][np.roll(order,shift)]
                eligible+=len(ids)
    return eligible,total


def shuffle_relation_identity(model,original,multiple,rng):
    for entries in multiple.values():
        vectors=[original[r]['pair'][index] for r,index in entries]
        order=rng.permutation(len(vectors));shift=int(rng.integers(1,len(vectors)))
        for target,permuted in zip(order,np.roll(order,shift)):
            r,index=entries[int(target)]
            model.prototypes[r]['pair'][index]=vectors[int(permuted)]


def posterior_smoothing_only(model):
    """Keep the smoothed posterior, remove only the explicit support multiplier."""
    for r,memory in model.memories.items():
        phi=model.phi[r];gram=phi.T@phi
        strength=.01*np.trace(gram)/gram.shape[0]
        projection=np.linalg.solve(gram+max(strength,1e-10)*np.eye(256),phi.T).T
        for name,counts,totals in (("herb",memory.herb_counts,memory.herb_n),("pair",memory.pair_counts,memory.pair_n)):
            values=model.prototypes[r][name]
            for start in range(0,len(values),512):
                ids=list(range(start,min(start+512,len(values))))
                lift=memory.lift(counts,ids,totals).astype(float)
                lift/=totals[ids,None]/(totals[ids,None]+20)
                values[start:start+len(ids)]=(lift-lift.mean(1,keepdims=True))@projection


def one(source,seed,lock,backbone_lr=None,development_only=False,output_dir=None):
    destination=OUT if output_dir is None else Path(output_dir)
    destination.mkdir(parents=True,exist_ok=True)
    tag=f"pairproto_paper_{source.lower()}_mechanisms_seed{seed}"
    path=destination/(tag+".json")
    if path.exists() and (destination/(tag+"_rr.npz")).exists():return
    data,model,info=build(source,seed,backbone_lr)
    if development_only:
        data=copy.copy(data);data.test=data.dev.copy()
        data.audit={'role':'Implementation preflight on development only, not main test evidence',
                    'queries':len(data.test),'source_audit':data.audit}
    gamma=PLAN["prototype"][source+"_pair_weight"]
    weights=(1-gamma,0,gamma)
    cache=profile_cache(model,data,data.test)
    arrays={"heads":data.test[:,0],"relations":data.test[:,1]}
    arrays["full"]=weighted_ranks(data,data.test,cache,weights)
    raw=profile_cache(model,data,data.test,raw=True)
    arrays["raw_fixed_weight"]=weighted_ranks(data,data.test,raw,weights)
    for name in ("herb","pair"):
        w=lock["selected"][name]
        mix=(1-w,w,0) if name=="herb" else (1-w,0,w)
        arrays["raw_"+name+"_dev_selected"]=weighted_ranks(data,data.test,raw,mix)
    arrays["raw_pair_only"]=weighted_ranks(data,data.test,raw,(0,0,1))
    arrays["raw_herb_only"]=weighted_ranks(data,data.test,raw,(0,1,0))
    original={r:{name:values.copy() for name,values in proto.items()} for r,proto in model.prototypes.items()}
    locations={}
    for r,memory in model.memories.items():
        for pair,index in memory.pair_vocab.items():locations.setdefault(pair,[]).append((r,index))
    multiple={pair:entries for pair,entries in locations.items() if len(entries)>1}
    for shuffle_seed in PLAN["shuffle_seeds"]:
        rng=np.random.default_rng(shuffle_seed)
        pair_eligible,pair_total=shuffle_pair_identity(model,original,rng)
        arrays[f"pair_identity_shuffle_{shuffle_seed}"]=weighted_ranks(data,data.test,pair_cache(model,data,data.test,cache),weights)
        for r in original:model.prototypes[r]["pair"][:]=original[r]["pair"]
        shuffle_relation_identity(model,original,multiple,rng)
        arrays[f"relation_identity_shuffle_{shuffle_seed}"]=weighted_ranks(data,data.test,pair_cache(model,data,data.test,cache),weights)
        for r in original:model.prototypes[r]["pair"][:]=original[r]["pair"]
    for entries in locations.values():
        shared=np.mean([original[r]["pair"][index] for r,index in entries],axis=0)
        for r,index in entries:model.prototypes[r]["pair"][index]=shared
    arrays["shared_pair_prototypes"]=weighted_ranks(data,data.test,pair_cache(model,data,data.test,cache),weights)
    for r in original:model.prototypes[r]["pair"][:]=original[r]["pair"]
    posterior_smoothing_only(model)
    arrays["posterior_smoothing_only"]=weighted_ranks(data,data.test,pair_cache(model,data,data.test,cache),weights)
    for name,kwargs in (("no_shrinkage",{"shrinkage":0}),("support_one",{"min_support":1})):
        alternate=PairPrototypeKGE(data,model.base,relations=list(model.memories),**kwargs)
        arrays[name]=weighted_ranks(data,data.test,pair_cache(alternate,data,data.test,cache),weights)
    fallback=[];coverage=[]
    for h,r,_ in data.test:
        pairs=list(itertools.combinations(ingredients(data,h),2));vocab=model.memories[int(r)].pair_vocab
        n=sum(p in vocab for p in pairs);coverage.append(n/max(1,len(pairs)));fallback.append(n==0)
    arrays["coverage"]=np.asarray(coverage)
    arrays["lengths"]=data.arrays[1].sum(1)[data.test[:,0]]
    no_pair=np.asarray(fallback)
    mean=weighted_ranks(data,data.test,cache,(1,0,0))
    if not np.array_equal(arrays["full"][no_pair],mean[no_pair]):raise AssertionError("Unsupported-pair fallback failed")
    scores={name:metrics(v) for name,v in arrays.items() if name not in ("heads","relations","coverage","lengths")}
    report={"source":source,"split":PLAN["datasets"][source]["main"],"seed":seed,"audit":data.audit,
            "backbone":info["checkpoint"],"gamma":gamma,"raw_control_lock":lock["selected"],
            "evaluation_partition":"development" if development_only else "test",
            "relation_shuffle_eligible_pairs":len(multiple),"unique_supported_pairs":len(locations),
            "relation_shuffle_eligible_fraction":len(multiple)/max(1,len(locations)),
            "pair_shuffle_eligible_entries":pair_eligible,"pair_relation_entries":pair_total,
            "pair_shuffle_policy":"Exact support-count groups within each relation, random cyclic derangements for groups of at least two; all vectors and support counts retained",
            "no_supported_pair_queries":int(no_pair.sum()),"fallback_ranks_equal":True,
            "shared_prototype_note":"Tied pair vectors across relations, identical decoder and mixture; smaller effective dictionary capacity, not a parameter-matched removal of all relation information"}
    report['development' if development_only else 'test']=scores
    save_npz(destination/(tag+"_rr.npz"),**arrays);save(path,report)
    print("MECHANISMS",source,seed,report['evaluation_partition'],{k:round(v["mrr"],5) for k,v in scores.items()},flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--source",choices=["PRIMARY","KTKP"])
    parser.add_argument('--preflight-development',action='store_true');args=parser.parse_args()
    torch.set_num_threads(4)
    with threadpool_limits(limits=4):
        for source in [args.source] if args.source else PLAN["datasets"]:
            if args.preflight_development:
                lock={'selected':{'herb':.1,'pair':PLAN['prototype'][source+'_pair_weight']}}
                one(source,PLAN['main_seeds'][0],lock,backbone_lr=.003,development_only=True,
                    output_dir=OUT/'pairproto_development_preflight')
                continue
            lock=raw_development_lock(source)
            for seed in PLAN["mechanism_seeds"]:one(source,seed,lock)


if __name__=="__main__":main()
