# -*- coding: utf-8 -*-
"""Formula-clustered uncertainty after global-negative training."""
from __future__ import annotations
import json
import numpy as np
import torch
from compositional_kge import OUT, make_strict_inductive, read_compositions
from expert_systems_robustness import clustered_bootstrap
from run_global_negative_sensitivity import SEEDS, make_model
from train_baselines import Dataset

@torch.inference_mode()
def untyped_ranks(model, data, triples, device):
    model.eval()
    candidates=np.arange(data.n_entities,dtype=np.int64)
    tails=torch.as_tensor(candidates,device=device)
    ranks=[]
    for h,r,t in triples:
        scores=model.score_tails(torch.tensor(int(h),device=device),torch.tensor(int(r),device=device),tails).float().cpu().numpy()
        known=data.all_true[(int(h),int(r))]-{int(t)}
        if known:
            scores[np.isin(candidates,np.fromiter(known,dtype=np.int64))]=-np.inf
        target=scores[int(t)]
        ranks.append(1.0+np.sum(scores>target)+0.5*max(0,int(np.sum(scores==target))-1))
    return np.asarray(ranks,dtype=np.float64)

def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    device=torch.device("cuda")
    data=Dataset("hybrid_holdout")
    arrays=read_compositions(data)
    make_strict_inductive(data)
    ranks={}
    for name in ("paircomp","mean_complex","matched"):
        rows=[]
        for seed in SEEDS:
            model=make_model(name,data,arrays).to(device)
            state=torch.load(OUT/f"globalneg_{name}_seed{seed}.pt",map_location=device,weights_only=True)
            model.load_state_dict(state["state_dict"])
            rows.append(untyped_ranks(model,data,data.test,device))
        ranks[name]=np.stack(rows)
    report={
        "paircomp_vs_mean_complex":clustered_bootstrap(ranks["paircomp"],ranks["mean_complex"],data.test[:,0]),
        "paircomp_vs_matched":clustered_bootstrap(ranks["paircomp"],ranks["matched"],data.test[:,0]),
        "protocol":"seed-average each query, then 20,000 formula-clustered bootstrap replicates; unrestricted filtered ranking after global-negative training",
    }
    (OUT/"global_negative_clustered_comparison.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report,indent=2))

if __name__=="__main__":
    main()
