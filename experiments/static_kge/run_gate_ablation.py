# -*- coding: utf-8 -*-
"""Gate ablations and branch-magnitude analysis for PairComp-KGE."""
from __future__ import annotations
import json, math
import numpy as np
import torch
from compositional_kge import OUT, make_strict_inductive, read_compositions
from hybrid_pair_kge import HybridPairComplEx, train_hybrid
from train_baselines import Dataset, evaluate

SEEDS=(20260915,20260916,20260917)
ABLATIONS=("fixed_pair_gate","ungated_pair")

def fresh():
    data=Dataset("hybrid_holdout"); arrays=read_compositions(data); make_strict_inductive(data)
    return data,arrays

def load_model(data,arrays,variant,path,device):
    model=HybridPairComplEx(data,variant,arrays).to(device)
    model.load_state_dict(torch.load(OUT/path,map_location=device,weights_only=True)["state_dict"])
    model.eval(); return model

@torch.inference_mode()
def branch_rows(model,heads):
    rows=[]
    for h in heads:
        ht=torch.tensor([h],device=model.er.weight.device)
        herbs=model.ingredients[ht]; valid=model.ingredient_mask[ht]
        ar,ai=model.er(herbs),model.ei(herbs)
        w=model.uniform_weight[ht]
        mr,mi=(w[...,None]*ar).sum(1),(w[...,None]*ai).sum(1)
        ur,ui=model._transform(mr,mi,model.unary_r,model.unary_i)
        pr,pi,n=model._pair_moment(ar,ai,valid)
        br,bi=model._transform(pr,pi,model.branch_r,model.branch_i)
        alpha=torch.sigmoid(model.unary_gate)
        if model.hybrid_variant=="fixed_pair_gate": beta=torch.sigmoid(model.pair_gate_intercept)
        elif model.hybrid_variant=="ungated_pair": beta=torch.tensor(1.0,device=mr.device)
        else: beta=torch.sigmoid(model.pair_gate_intercept+model.pair_gate_log_size*(torch.log(n)-math.log(5.0)))[0]
        cn=lambda x,y: float(torch.sqrt(x.square().sum()+y.square().sum()))
        rows.append({"formula":int(h),"size":int(n[0]),"mu_norm":cn(mr,mi),"unary_norm":cn(alpha*ur,alpha*ui),"pair_norm":cn(beta*br,beta*bi),"pair_gate":float(beta)})
    return rows

def main():
    if not torch.cuda.is_available(): raise RuntimeError("CUDA required")
    device=torch.device("cuda")
    dev_path=OUT/"gate_ablation_dev.json"
    state=json.loads(dev_path.read_text(encoding="utf-8")) if dev_path.exists() else {"models":{}}
    for variant in ABLATIONS:
        state["models"].setdefault(variant,{})
        for seed in SEEDS:
            old=state["models"][variant].get(str(seed))
            if old and (OUT/old["checkpoint"]).exists(): continue
            data,arrays=fresh()
            state["models"][variant][str(seed)]=train_hybrid(data,variant,arrays,device,seed,f"gate_{variant}_seed{seed}.pt")
            dev_path.write_text(json.dumps(state,indent=2),encoding="utf-8")
    primary=json.loads((OUT/"hybrid_confirmatory_dev_selection.json").read_text(encoding="utf-8"))
    report={"models":{},"branch_magnitudes":{}}
    paths={"learned_size_gate":{str(s):primary["hybrid"][str(s)]["checkpoint"] for s in SEEDS}}
    for variant in ABLATIONS: paths[variant]={str(s):state["models"][variant][str(s)]["checkpoint"] for s in SEEDS}
    for label,seed_paths in paths.items():
        variant="hybrid_pair" if label=="learned_size_gate" else label
        scores={}; all_rows=[]
        for seed in SEEDS:
            data,arrays=fresh(); model=load_model(data,arrays,variant,seed_paths[str(seed)],device)
            scores[str(seed)]={"typed":evaluate(model,data,data.test,device,True),"untyped":evaluate(model,data,data.test,device,False)}
            all_rows.extend(branch_rows(model,sorted(set(map(int,data.test[:,0])))))
        report["models"][label]={"seeds":scores,"summary":{m:{"mean":float(np.mean([scores[str(s)]["typed"][m] for s in SEEDS])),"sample_std":float(np.std([scores[str(s)]["typed"][m] for s in SEEDS],ddof=1))} for m in ("mrr","hits1","hits3","hits10")}}
        bins={"1-5":lambda n:n<=5,"6-10":lambda n:6<=n<=10,"11+":lambda n:n>=11}
        report["branch_magnitudes"][label]={}
        for name,predicate in bins.items():
            selected=[x for x in all_rows if predicate(x["size"])]
            report["branch_magnitudes"][label][name]={"n_seed_formula_rows":len(selected),**{k:float(np.mean([x[k] for x in selected])) for k in ("mu_norm","unary_norm","pair_norm","pair_gate")}}
        print(label,report["models"][label]["summary"],flush=True)
        (OUT/"gate_ablation_results.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
if __name__=="__main__": main()
