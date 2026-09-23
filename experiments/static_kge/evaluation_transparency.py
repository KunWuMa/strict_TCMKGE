# -*- coding: utf-8 -*-
"""Transparent relation-wise typed and global-untyped evaluation."""
from __future__ import annotations
import json
import numpy as np
import torch
from compositional_kge import OUT, make_strict_inductive, read_compositions
from expanded_kge_baselines import initialise_model
from hybrid_pair_kge import HybridPairComplEx
from train_baselines import Dataset, evaluate

SEEDS=(20260915,20260916,20260917)

def load(name,seed,data,arrays,device):
    if name=="paircomp":
        model=HybridPairComplEx(data,"hybrid_pair",arrays)
        path=OUT/f"hybridconfirm_hybrid_seed{seed}.pt"
    elif name=="matched":
        model=HybridPairComplEx(data,"dual_mean_control",arrays)
        path=OUT/f"hybridconfirm_parameter_control_seed{seed}.pt"
    else:
        model=initialise_model(data,arrays,"mean_complex")
        path=OUT/f"expanded_mean_complex_seed{seed}.pt"
    model=model.to(device)
    model.load_state_dict(torch.load(path,map_location=device,weights_only=True)["state_dict"])
    return model

def average(items):
    return {m:{"mean":float(np.mean([x[m] for x in items])),"sample_std":float(np.std([x[m] for x in items],ddof=1))} for m in ("mrr","hits1","hits3","hits10","mean_rank")}

def main():
    if not torch.cuda.is_available(): raise RuntimeError("CUDA required")
    device=torch.device("cuda"); data=Dataset("hybrid_holdout"); arrays=read_compositions(data); make_strict_inductive(data)
    report={"protocol":{"typed_candidates":"relation ranges observed in strict training graph","untyped_candidates":data.n_entities,"filtered":True},"relations":{},"overall":{}}

    primary=json.loads((OUT/"hybrid_confirmatory_results.json").read_text())
    expanded=json.loads((OUT/"expanded_kge_baselines_results.json").read_text())
    for name in ("paircomp","mean_complex","matched"):
        models=[load(name,s,data,arrays,device) for s in SEEDS]
        if name == "paircomp":
            typed_all=[primary["hybrid"][str(s)]["typed"] for s in SEEDS]
            untyped_all=[primary["hybrid"][str(s)]["untyped"] for s in SEEDS]
        elif name == "matched":
            typed_all=[primary["parameter_control"][str(s)]["typed"] for s in SEEDS]
            untyped_all=[primary["parameter_control"][str(s)]["untyped"] for s in SEEDS]
        else:
            typed_all=[expanded["models"]["mean_complex"][str(s)]["typed"] for s in SEEDS]
            untyped_all=[evaluate(model,data,data.test,device,False) for model in models]
        report["overall"][name]={"typed":average(typed_all),"untyped":average(untyped_all)}
        for rid,relation in enumerate(data.relations):
            subset=data.test[data.test[:,1]==rid]
            if not len(subset): continue
            entry=report["relations"].setdefault(relation,{"queries":int(len(subset)),"typed_candidates":int(len(data.range_candidates[rid])),"untyped_candidates":data.n_entities})
            vals=[evaluate(model,data,subset,device,True) for model in models]
            entry[name]=average(vals)
            print(name,relation,entry[name]["mrr"],flush=True)
            (OUT/"evaluation_transparency.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=="__main__": main()
