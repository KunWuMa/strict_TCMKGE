"""Constructive, balanced-marginal collision example at the default support=3.

This tests representational non-reducibility for fixed collided herb embeddings,
not the impossibility of a learned mean encoder fitting arbitrary training data.
The two query recipes never occur among the 12 training recipe groups.
"""
from types import SimpleNamespace
import json
from pathlib import Path

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from compositional_kge import CompositionalComplEx
from pair_prototype_kge import PairPrototypeKGE


def experiment():
    herbs=np.zeros((23,3),dtype=np.int64);mask=np.zeros_like(herbs,dtype=bool)
    triples=[];head=7
    for pair,target in [((0,1),21),((2,3),21),((0,2),22),((1,3),22)]:
        for context in (4,5,6):
            herbs[head]=[*pair,context];mask[head]=True
            triples.append((head,0,target));head+=1
    herbs[19,:2]=[0,1];herbs[20,:2]=[0,2];mask[19:21,:2]=True
    uniform=(mask/np.maximum(1,mask.sum(1,keepdims=True))).astype(np.float32)
    arrays=(herbs,mask,uniform,uniform.copy(),np.zeros(23,dtype=bool))
    data=SimpleNamespace(n_entities=23,n_relations=1,arrays=arrays,train=np.asarray(triples),
                         dev=np.array([[19,0,21],[20,0,22]]),test=np.array([[19,0,21],[20,0,22]]),
                         range_candidates={0:np.array([21,22])})
    base=CompositionalComplEx(data,"uniform_mean",arrays)
    with torch.no_grad():
        for p in base.parameters():p.zero_()
        base.rr.weight[0,0]=1.;base.er.weight[21,0]=1.;base.er.weight[22,0]=-1.
    model=PairPrototypeKGE(data,base,weights=(0.,.1),min_support=3)
    tails=torch.tensor([21,22])
    mean_scores=[base.score_tails(torch.tensor(h),torch.tensor(0),tails).detach().numpy() for h in (19,20)]
    full_scores=[model.score_tails(torch.tensor(h),torch.tensor(0),tails).numpy() for h in (19,20)]
    assert np.allclose(mean_scores[0],mean_scores[1]) and np.allclose(mean_scores,0)
    assert max(np.max(np.abs(v)) for v in [model.prototypes[0]["herb"]])<1e-10
    assert full_scores[0][0]>full_scores[0][1] and full_scores[1][1]>full_scores[1][0]
    train_recipes={tuple(sorted(herbs[h][mask[h]])) for h,_,_ in data.train}
    assert all(tuple(sorted(herbs[h][mask[h]])) not in train_recipes for h in (19,20))
    assert model.memories[0].pair_n[model.memories[0].pair_vocab[(0,1)]]==3
    data.test[:,2]=data.test[::-1,2].copy()
    rebuilt=PairPrototypeKGE(data,base,weights=(0.,.1),min_support=3)
    np.testing.assert_allclose(model.prototypes[0]["pair"],rebuilt.prototypes[0]["pair"])
    return {"role":"Synthetic mathematical witness, no medical annotations or clinical validation",
            "training_recipe_groups":12,"query_recipes":["AB","AC"],"query_recipes_seen_in_training":False,
            "support_threshold":3,"minimum_pair_support_met":True,"herb_profile_max_abs":float(np.abs(model.prototypes[0]["herb"]).max()),
            "herb_mean_and_all_elementwise_moments":"Identical zero vectors for both queries",
            "mean_scores":np.asarray(mean_scores).tolist(),"pairproto_scores":np.asarray(full_scores).tolist(),
            "interpretation":"Pair identity profiles can separate colliding moment summaries; this is not a universal learning-theoretic superiority claim"}


def distinct_embedding_experiment():
    """A stronger witness: four distinct nonzero constituent embeddings."""
    herbs=np.zeros((35,3),dtype=np.int64);mask=np.zeros_like(herbs,dtype=bool)
    triples=[];head=13
    for j in range(3):
        e,f,g=4+j,7+j,10+j
        recipes=[([0,1,e],33),([2,3,e],34),([0,f],34),
                 ([1,g],34),([2,f],33),([3,g],33)]
        for recipe,target in recipes:
            herbs[head,:len(recipe)]=recipe;mask[head,:len(recipe)]=True
            triples.append([head,0,target]);head+=1
    herbs[31,:2]=[0,1];herbs[32,:2]=[2,3];mask[31:33,:2]=True
    weights=(mask/np.maximum(1,mask.sum(1,keepdims=True))).astype(np.float32)
    arrays=(herbs,mask,weights,weights.copy(),np.zeros(35,dtype=bool))
    data=SimpleNamespace(n_entities=35,n_relations=1,arrays=arrays,train=np.array(triples),
        dev=np.array([[31,0,33],[32,0,34]]),test=np.array([[31,0,33],[32,0,34]]),
        range_candidates={0:np.array([33,34])})
    base=CompositionalComplEx(data,'uniform_mean',arrays)
    with torch.no_grad():
        for p in base.parameters():p.zero_()
        base.er.weight[:4,:2]=torch.tensor([[1.,1.],[-1.,-1.],[1.,-1.],[-1.,1.]])
        base.rr.weight[0,0]=1; base.er.weight[33,0]=1; base.er.weight[34,0]=-1
    query_vectors=[base.er.weight[[0,1]].detach().numpy(),base.er.weight[[2,3]].detach().numpy()]
    assert len(np.unique(base.er.weight[:4].detach().numpy(),axis=0))==4
    assert (np.linalg.norm(base.er.weight[:4].detach().numpy(),axis=1)>0).all()
    for power in range(1,7):
        np.testing.assert_allclose((query_vectors[0]**power).mean(0),(query_vectors[1]**power).mean(0))
    model=PairPrototypeKGE(data,base,weights=(0,.1),min_support=3)
    assert np.max(np.abs(model.prototypes[0]['herb']))<1e-10
    scores=[model.score_tails(torch.tensor(h),torch.tensor(0),torch.tensor([33,34])).numpy() for h in (31,32)]
    assert scores[0][0]>scores[0][1] and scores[1][1]>scores[1][0]
    train_recipes={tuple(sorted(herbs[h][mask[h]])) for h,_,_ in data.train}
    assert len(train_recipes)==18
    assert all(tuple(sorted(herbs[h][mask[h]])) not in train_recipes for h in (31,32))
    return {'training_recipe_groups':18,'query_recipes':['AB','CD'],
        'query_herb_embeddings':[[1,1],[-1,-1],[1,-1],[-1,1]],
        'query_herb_embeddings_all_distinct_and_nonzero':True,
        'equal_elementwise_moments':'All orders, because each coordinate has the same {-1,+1} multiset; powers 1 through 6 checked numerically',
        'minimum_pair_support':3,'balanced_herb_profiles':True,
        'mean':query_vectors[0].mean(0)[:2].tolist(),
        'second_moment':(query_vectors[0]**2).mean(0)[:2].tolist(),
        'pairproto_scores':np.asarray(scores).tolist(),
        'interpretation':'Fixed coordinatewise moment summaries also collide with distinct nonzero queried herb embeddings; still not universal learning superiority.'}


if __name__=="__main__":
    torch.set_num_threads(2)
    with threadpool_limits(limits=2):
        result=experiment()
        result['distinct_nonzero_embedding_witness']=distinct_embedding_experiment()
    out=Path(__file__).with_name("results")/"pairproto_moment_collision.json"
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps(result,indent=2))
