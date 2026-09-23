# -*- coding: utf-8 -*-
"""Formula-clustered uncertainty for PairComp versus Set Transformer."""
import json
import numpy as np
import torch
from analyze_composition_results import typed_ranks
from analyze_hybrid_confirmatory import load
from compositional_kge import OUT, make_strict_inductive, read_compositions
from expanded_kge_baselines import initialise_model
from expert_systems_robustness import clustered_bootstrap
from train_baselines import Dataset

SEEDS=(20260915,20260916,20260917)

def main():
    if not torch.cuda.is_available(): raise RuntimeError("CUDA required")
    device=torch.device("cuda"); data=Dataset("hybrid_holdout"); arrays=read_compositions(data); make_strict_inductive(data)
    pair=[]; st=[]
    for seed in SEEDS:
        pm=load("hybrid",data,arrays,seed,device)
        sm=initialise_model(data,arrays,"set_transformer").to(device)
        state=torch.load(OUT/f"expanded_set_transformer_seed{seed}.pt",map_location=device,weights_only=True)
        sm.load_state_dict(state["state_dict"])
        pair.append(typed_ranks(pm,data,data.test,device))
        st.append(typed_ranks(sm,data,data.test,device))
    pair=np.stack(pair);st=np.stack(st)
    report={"paircomp_vs_set_transformer":clustered_bootstrap(pair,st,data.test[:,0]),"seed_mrr":{"paircomp":[float(np.mean(1/x)) for x in pair],"set_transformer":[float(np.mean(1/x)) for x in st]},"protocol":"seed-average each query, then formula-clustered bootstrap with all queries retained"}
    (OUT/"set_transformer_clustered_comparison.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report,indent=2))
if __name__=="__main__":main()
