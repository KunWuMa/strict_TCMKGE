# -*- coding: utf-8 -*-
"""Per-query statistics for the expanded strict-inductive KGE baselines."""

from __future__ import annotations

import json

import numpy as np
import torch

from analyze_composition_results import typed_ranks
from analyze_hybrid_confirmatory import compare
from compositional_kge import OUT, make_strict_inductive, read_compositions
from expanded_kge_baselines import SEEDS, initialise_model
from expert_systems_robustness import clustered_bootstrap
from hybrid_pair_kge import HybridPairComplEx
from train_baselines import Dataset

BASELINES = ("mean_complex", "rotate", "pairre", "nodepiece_transformer")
COMPARISONS = BASELINES + ("ingram",)


def load_baseline(name, seed, data, arrays, device):
    model = initialise_model(data, arrays, name).to(device)
    state = torch.load(
        OUT / f"expanded_{name}_seed{seed}.pt", map_location=device, weights_only=True
    )
    model.load_state_dict(state["state_dict"])
    return model


def load_paircomp(seed, data, arrays, device):
    model = HybridPairComplEx(data, "hybrid_pair", arrays).to(device)
    state = torch.load(
        OUT / f"hybridconfirm_hybrid_seed{seed}.pt",
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(state["state_dict"])
    return model


def metrics(ranks):
    return {
        "mrr": float((1.0 / ranks).mean()),
        "hits1": float((ranks <= 1).mean()),
        "hits3": float((ranks <= 3).mean()),
        "hits10": float((ranks <= 10).mean()),
        "mean_rank": float(ranks.mean()),
        "per_seed_mrr": [float((1.0 / row).mean()) for row in ranks],
    }


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    device = torch.device("cuda")
    data = Dataset("hybrid_holdout")
    arrays = read_compositions(data)
    make_strict_inductive(data)

    ranks = {}
    ranks["paircomp_kge"] = np.stack(
        [typed_ranks(load_paircomp(seed, data, arrays, device), data, data.test, device)
         for seed in SEEDS]
    )
    for name in BASELINES:
        ranks[name] = np.stack(
            [typed_ranks(load_baseline(name, seed, data, arrays, device), data, data.test, device)
             for seed in SEEDS]
        )
    ingram = json.loads((OUT / "ingram_tcm_full200_results.json").read_text(encoding="utf-8"))
    ranks["ingram"] = np.asarray(ingram["per_query_ranks"], dtype=np.float64)

    report = {
        "protocol": {
            "split": "hybrid_holdout",
            "unit": "query after averaging the three seed-specific reciprocal ranks",
            "bootstrap_repetitions": 20000,
            "typed_filtered": True,
        },
        "overall": {name: metrics(value) for name, value in ranks.items()},
        "paired_bootstrap": {
            f"paircomp_vs_{name}": compare(ranks["paircomp_kge"], ranks[name])
            for name in COMPARISONS
        },
        "clustered_bootstrap": {
            f"paircomp_vs_{name}": clustered_bootstrap(
                ranks["paircomp_kge"], ranks[name], data.test[:, 0]
            )
            for name in COMPARISONS
        },
        "integrity": {
            "queries": int(len(data.test)),
            "test_formulae": int(len(set(map(int, data.test[:, 0])))),
            "seeds": list(SEEDS),
        },
    }
    (OUT / "expanded_kge_baselines_detailed_analysis.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
