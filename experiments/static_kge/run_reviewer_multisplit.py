# -*- coding: utf-8 -*-
"""Three-split by three-seed robustness experiment requested during review."""
from __future__ import annotations
import json
import numpy as np
import torch
from compositional_kge import OUT, CompositionalComplEx, make_strict_inductive, read_compositions, train_one
from hybrid_pair_kge import HybridPairComplEx, train_hybrid
from train_baselines import Dataset, evaluate

SPLITS=("review_split_2","review_split_3")
SEEDS=(20260915,20260916,20260917)
MODELS={"paircomp":"hybrid_pair","matched":"dual_mean_control","mean":"uniform_mean"}

def fresh(split):
    data=Dataset(split); arrays=read_compositions(data); integrity=make_strict_inductive(data)
    return data,arrays,integrity

def main():
    if not torch.cuda.is_available(): raise RuntimeError("CUDA required")
    device=torch.device("cuda")
    dev_path=OUT/"reviewer_multisplit_dev.json"
    state=json.loads(dev_path.read_text(encoding="utf-8")) if dev_path.exists() else {"protocol":{},"runs":{}}
    state["protocol"]={"splits":list(SPLITS),"seeds":list(SEEDS),"selection":"typed filtered dev MRR","same_hyperparameters":True}
    for split in SPLITS:
        state["runs"].setdefault(split,{})
        for name,variant in MODELS.items():
            state["runs"][split].setdefault(name,{})
            for seed in SEEDS:
                old=state["runs"][split][name].get(str(seed))
                if old and (OUT/old["checkpoint"]).exists():
                    print("RESUME SKIP",split,name,seed,flush=True); continue
                data,arrays,integrity=fresh(split)
                tag=f"review_{split}_{name}_seed{seed}.pt"
                result=train_one(data,variant,arrays,device,seed,tag) if name=="mean" else train_hybrid(data,variant,arrays,device,seed,tag)
                result["integrity"]=integrity
                state["runs"][split][name][str(seed)]=result
                dev_path.write_text(json.dumps(state,indent=2),encoding="utf-8")
    report={"protocol":state["protocol"],"splits":{}}
    for split in SPLITS:
        report["splits"][split]={}
        for name,variant in MODELS.items():
            per_seed={}
            for seed in SEEDS:
                data,arrays,integrity=fresh(split)
                model=CompositionalComplEx(data,variant,arrays) if name=="mean" else HybridPairComplEx(data,variant,arrays)
                model=model.to(device)
                chosen=state["runs"][split][name][str(seed)]
                saved=torch.load(OUT/chosen["checkpoint"],map_location=device,weights_only=True)
                model.load_state_dict(saved["state_dict"])
                per_seed[str(seed)]={"typed":evaluate(model,data,data.test,device,True),"untyped":evaluate(model,data,data.test,device,False),"selected":chosen["selected"],"parameters":chosen["parameters"]}
                (OUT/"reviewer_multisplit_results.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
            summary={metric:{"mean":float(np.mean([per_seed[str(s)]["typed"][metric] for s in SEEDS])),"sample_std":float(np.std([per_seed[str(s)]["typed"][metric] for s in SEEDS],ddof=1))} for metric in ("mrr","hits1","hits3","hits10")}
            report["splits"][split][name]={"seeds":per_seed,"summary":summary}
            print("TEST",split,name,summary,flush=True)
            (OUT/"reviewer_multisplit_results.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print("MULTISPLIT COMPLETE",flush=True)

if __name__=="__main__":
    main()
