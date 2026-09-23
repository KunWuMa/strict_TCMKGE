# -*- coding: utf-8 -*-
"""Ingredient-pair occlusion sanity check for gradient-based local sensitivity."""
from __future__ import annotations
import json, math
import numpy as np
import torch
from analyze_composition_results import typed_ranks
from analyze_hybrid_confirmatory import load
from compositional_kge import OUT, make_strict_inductive, read_compositions
from pair_attribution_cases import one_model_attribution
from train_baselines import Dataset

SEEDS=(20260915,20260916,20260917)

@torch.inference_mode()
def score_subset(model,herbs,r,t):
    device=model.er.weight.device
    ids=torch.tensor(herbs,device=device)
    ar,ai=model.er(ids),model.ei(ids); n=len(herbs)
    mr,mi=ar.mean(0),ai.mean(0)
    ur,ui=model._transform(mr,mi,model.unary_r,model.unary_i)
    pr,pi,count=model._pair_moment(ar[None],ai[None],torch.ones((1,n),dtype=torch.bool,device=device))
    br,bi=model._transform(pr[0],pi[0],model.branch_r,model.branch_i)
    beta=torch.sigmoid(model.pair_gate_intercept+model.pair_gate_log_size*(torch.log(count[0])-math.log(5.0)))
    hr=mr+torch.sigmoid(model.unary_gate)*ur+beta*br
    hi=mi+torch.sigmoid(model.unary_gate)*ui+beta*bi
    rt=torch.tensor([r],device=device); tt=torch.tensor([t],device=device)
    rr,ri=model.rr(rt)[0],model.ri(rt)[0]; tr,ti=model.er(tt)[0],model.ei(tt)[0]
    return float((hr*rr*tr+hi*rr*ti+hr*ri*ti-hi*ri*tr).sum())

def bootstrap(values,seed=20261101):
    rng=np.random.default_rng(seed); a=np.asarray(values,float)
    means=np.asarray([rng.choice(a,size=len(a),replace=True).mean() for _ in range(20000)])
    return [float(np.quantile(means,.025)),float(np.quantile(means,.975))]

def main():
    if not torch.cuda.is_available(): raise RuntimeError("CUDA required")
    device=torch.device("cuda"); data=Dataset("hybrid_holdout"); arrays=read_compositions(data); make_strict_inductive(data)
    models=[load("hybrid",data,arrays,s,device) for s in SEEDS]
    ranks=np.stack([typed_ranks(m,data,data.test,device) for m in models])
    rng=np.random.default_rng(20261100); rows=[]
    for edge,(h,r,t) in enumerate(data.test):
        h,r,t=map(int,(h,r,t)); herbs=list(map(int,arrays[0][h][arrays[1][h]]))
        if not 3<=len(herbs)<=10 or np.mean(ranks[:,edge])>3: continue
        for si,model in enumerate(models):
            values,full,_,_,_=one_model_attribution(model,h,r,t)
            ordered=sorted(values,key=lambda x:x[2],reverse=True)
            top=ordered[0]; alternatives=ordered[1:]
            random_pair=alternatives[int(rng.integers(len(alternatives)))]
            low=ordered[-1]
            def drop(pair):
                remaining=list(herbs)
                remaining.remove(pair[0]); remaining.remove(pair[1])
                return full-score_subset(model,remaining,r,t)
            rows.append({"formula":h,"edge":edge,"seed":SEEDS[si],"size":len(herbs),"top_drop":drop(top),"random_drop":drop(random_pair),"lowest_drop":drop(low),"top_attribution":top[2]})
    formula_values={}
    for row in rows: formula_values.setdefault(row["formula"],[]).append(row["top_drop"]-row["random_drop"])
    clustered=[float(np.mean(v)) for v in formula_values.values()]
    report={"definition":"remove the two herbs in a gradient-selected pair, recompute all normalized first- and second-order statistics, and compare target-score drop with a random within-formula pair","eligible_seed_query_rows":len(rows),"formula_clusters":len(clustered),"mean_target_score_drop":{"top_pair":float(np.mean([x["top_drop"] for x in rows])),"random_pair":float(np.mean([x["random_drop"] for x in rows])),"lowest_pair":float(np.mean([x["lowest_drop"] for x in rows]))},"top_minus_random_cluster_mean":float(np.mean(clustered)),"top_minus_random_formula_clustered_95ci":bootstrap(clustered),"scope":"perturbation sanity check for local model sensitivity; not evidence of clinical herb-pair validity"}
    (OUT/"pair_attribution_occlusion_sanity.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report,indent=2))
if __name__=="__main__": main()
