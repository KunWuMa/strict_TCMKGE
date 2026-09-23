# -*- coding: utf-8 -*-
"""Inference-time causal ablation of the learned pair branch."""

import json
import numpy as np
import torch

from analyze_composition_results import typed_ranks
from analyze_hybrid_confirmatory import paired_bootstrap
from compositional_kge import OUT, make_strict_inductive, read_compositions
from hybrid_pair_kge import HybridPairComplEx
from train_baselines import Dataset

SEEDS = (20260915, 20260916, 20260917)


def metric_table(ranks):
    return {"mrr": float((1.0/ranks).mean()), "hits1": float((ranks <= 1).mean()),
            "hits3": float((ranks <= 3).mean()), "hits10": float((ranks <= 10).mean())}


def main():
    if not torch.cuda.is_available(): raise RuntimeError("CUDA required")
    device = torch.device("cuda"); data = Dataset("hybrid_holdout")
    arrays = read_compositions(data); make_strict_inductive(data)
    full, ablated = [], []
    for seed in SEEDS:
        model = HybridPairComplEx(data, "hybrid_pair", arrays).to(device)
        state = torch.load(OUT/f"hybridconfirm_hybrid_seed{seed}.pt",
                           map_location=device, weights_only=True)
        model.load_state_dict(state["state_dict"])
        full.append(typed_ranks(model, data, data.test, device))
        with torch.no_grad(): model.pair_gate_intercept.fill_(-100.0)
        ablated.append(typed_ranks(model, data, data.test, device))
    full, ablated = np.stack(full), np.stack(ablated)
    report = {"full": metric_table(full), "pair_branch_zeroed": metric_table(ablated),
              "full_minus_zeroed": {
                  "mrr": paired_bootstrap(full, ablated, lambda x: 1.0/x),
                  "hits1": paired_bootstrap(full, ablated, lambda x: (x <= 1).astype(float)),
                  "hits3": paired_bootstrap(full, ablated, lambda x: (x <= 3).astype(float)),
                  "hits10": paired_bootstrap(full, ablated, lambda x: (x <= 10).astype(float))},
              "note": "post-hoc inference ablation of frozen checkpoints; no retraining"}
    (OUT/"pair_branch_ablation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__": main()
