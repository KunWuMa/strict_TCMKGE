# -*- coding: utf-8 -*-
"""Paired query- and formula-cluster inference for decoder transfer experiments."""

from __future__ import annotations

import json

import numpy as np
import torch

from analyze_composition_results import typed_ranks
from compositional_kge import OUT, make_strict_inductive, read_compositions
from expert_systems_robustness import clustered_bootstrap
from real_pair_decoders import DECODERS, RealPairKGE, SEEDS
from train_baselines import Dataset


def paired_bootstrap(a, b, seed=20261301, repetitions=20000):
    """Average model seeds first, then resample the same test queries."""
    delta = (1.0 / a).mean(0) - (1.0 / b).mean(0)
    rng = np.random.default_rng(seed)
    samples = np.empty(repetitions)
    for i in range(repetitions):
        idx = rng.integers(0, len(delta), len(delta))
        samples[i] = delta[idx].mean()
    lo, hi = np.quantile(samples, (0.025, 0.975))
    return {
        "mrr_difference": float(delta.mean()),
        "paired_query_bootstrap_95ci": [float(lo), float(hi)],
        "probability_difference_le_zero": float((samples <= 0).mean()),
        "repetitions": repetitions,
    }


def load_model(data, arrays, decoder, encoder, seed, device):
    model = RealPairKGE(data, arrays, decoder, encoder).to(device)
    state = torch.load(
        OUT / f"decoder_{decoder}_{encoder}_seed{seed}.pt",
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(state["state_dict"])
    return model


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    device = torch.device("cuda")
    data = Dataset("hybrid_holdout")
    arrays = read_compositions(data)
    make_strict_inductive(data)
    heads = data.test[:, 0]
    report = {"seeds": list(SEEDS), "test_edges": int(len(data.test)), "formula_heads": int(len(np.unique(heads))), "decoders": {}}
    for decoder in DECODERS:
        ranks = {}
        for encoder in ("mean", "pair"):
            ranks[encoder] = np.stack([
                typed_ranks(load_model(data, arrays, decoder, encoder, seed, device), data, data.test, device)
                for seed in SEEDS
            ])
        report["decoders"][decoder] = {
            "pair_vs_mean_query_bootstrap": paired_bootstrap(ranks["pair"], ranks["mean"]),
            "pair_vs_mean_formula_cluster_bootstrap": clustered_bootstrap(ranks["pair"], ranks["mean"], heads),
        }
    (OUT / "decoder_transfer_analysis.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
