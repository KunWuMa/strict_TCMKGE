# -*- coding: utf-8 -*-
"""Dev-only prototype projection audit: train-derived pair evidence in ComplEx space.

This tests whether statistical pair signal survives a genuine compositional head
embedding. No candidate-specific nonlinear decoder or test calibration is used.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from compositional_kge import OUT, CompositionalComplEx, make_strict_inductive, read_compositions
from inpetm_independent import InPETMDataset
from hybrid_pair_kge import HybridPairComplEx
from pair_memory_dev import PairMemory, ingredients, standardized
from train_baselines import Dataset


def load_data(source, seed, split_seed, base="mean"):
    if source == "PRIMARY":
        if split_seed != 20261021:
            raise ValueError("PRIMARY uses hybrid_holdout")
        data = Dataset("hybrid_holdout")
        data.arrays = read_compositions(data)
        make_strict_inductive(data)
        tag = (f"expanded_mean_complex_seed{seed}" if base == "mean" else
               f"hybridconfirm_hybrid_seed{seed}")
    else:
        data = InPETMDataset(source, split_seed, 400)
        tag = f"inpetm_{source.lower()}_g400_{base}_k3_b0.25_seed{seed}"
        if split_seed != 20261021:
            tag += f"_split{split_seed}"
    return data, tag


def tail_features(model, relation, candidates):
    with torch.inference_mode():
        ids = torch.as_tensor(candidates, device=model.er.weight.device)
        rr, ri = model.rr.weight[relation], model.ri.weight[relation]
        tr, ti = model.er(ids), model.ei(ids)
        return torch.cat((rr*tr + ri*ti, rr*ti - ri*tr), dim=-1).cpu().numpy().astype(np.float64)


def evaluate_dev(data, queries, model, ridge=0.01):
    relations = [int(r) for r in np.unique(queries[:, 1])]
    memories = {r: PairMemory(data, r) for r in relations}
    projections = {}
    for r in relations:
        phi = tail_features(model, r, data.range_candidates[r])
        centered = phi - phi.mean(0, keepdims=True)
        gram = centered.T @ centered
        strength = ridge * np.trace(gram) / gram.shape[0]
        projector = np.linalg.solve(gram + max(strength, 1e-10)*np.eye(gram.shape[0]), centered.T).T
        projections[r] = phi, centered, projector
    single = [(0.,0.), (.1,0.), (.25,0.), (.5,0.), (.75,0.), (1.,0.),
              (0.,.1), (0.,.25), (0.,.5), (0.,.75), (0.,1.)]
    mixtures = single + [(h,p) for h in (.1,.25,.5) for p in (.1,.25,.5) if h+p <= 1]
    reciprocal = {str((h,p)): [] for h,p in mixtures}
    cache = {}
    equivalence_error = 0.
    with torch.inference_mode():
        for h,r,t in queries:
            key = int(h),int(r)
            candidates = data.range_candidates[int(r)]
            if key not in cache:
                phi, centered, projector = projections[int(r)]
                base = model.score_tails(torch.tensor(int(h),device="cuda"), torch.tensor(int(r),device="cuda"), torch.as_tensor(candidates,device="cuda")).cpu().numpy()
                signals, _ = memories[int(r)].scores(ingredients(data,h))
                projected = {}
                for name in ("herb_mean","pair_mean"):
                    signal = signals[name].astype(np.float64)
                    embedding = (signal-signal.mean()) @ projector
                    scores = embedding @ phi.T
                    # Centering only changes a query-constant offset, preserving rank.
                    equivalence_error = max(equivalence_error,float(np.max(np.abs((scores-scores.mean())-embedding@centered.T))))
                    projected[name] = standardized(scores)
                cache[key] = standardized(base),projected["herb_mean"],projected["pair_mean"]
            base,herb,pair = cache[key]
            idx = np.searchsorted(candidates,int(t))
            known = data.all_true[key]-{int(t)}
            allowed = ~np.isin(candidates,np.fromiter(known,dtype=np.int64))
            for hw,pw in mixtures:
                score = (1-hw-pw)*base + hw*herb + pw*pair
                target = score[idx]
                rank = 1+np.sum((score>target)&allowed)+.5*max(0,int(np.sum((score==target)&allowed))-1)
                reciprocal[str((hw,pw))].append(1/rank)
    return {"mrr":{k:float(np.mean(v)) for k,v in reciprocal.items()},
            "embedding_score_equivalence_max_error":equivalence_error,
            "memory":{str(r):{"recipes":m.n_recipes,"herbs":len(m.herb_vocab),"pairs":len(m.pair_vocab)} for r,m in memories.items()}}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--source",choices=["KTKP","DRAGON","PRIMARY"],required=True)
    parser.add_argument("--seed",type=int,default=20260915)
    parser.add_argument("--split-seed",type=int,default=20261021)
    parser.add_argument("--base",choices=["mean","pair"],default="mean")
    args=parser.parse_args()
    torch.set_num_threads(4)
    data,tag=load_data(args.source,args.seed,args.split_seed,args.base)
    model=(CompositionalComplEx(data,"uniform_mean",data.arrays) if args.base=="mean" else
           HybridPairComplEx(data,"hybrid_pair",data.arrays)).cuda()
    model.load_state_dict(torch.load(OUT/f"{tag}.pt",map_location="cuda",weights_only=False)["state_dict"])
    model.eval()
    report={"role":"development only", "source":args.source,"config":vars(args),
            "ridge_relative":0.01,"shrinkage":20,"minimum_pair_support":3,
            "base_checkpoint":tag+".pt", "n_dev":len(data.dev)}
    report.update(evaluate_dev(data,data.dev,model))
    name=f"pair_prototype_{args.source.lower()}_seed{args.seed}"
    if args.base!="mean":name+=f"_base{args.base}"
    if args.split_seed!=20261021:name+=f"_split{args.split_seed}"
    (OUT/f"{name}_dev.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report,indent=2))


if __name__=="__main__":main()
