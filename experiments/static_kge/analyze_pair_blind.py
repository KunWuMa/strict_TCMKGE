# -*- coding: utf-8 -*-
"""Paired and stratified analysis for the first PairComp-KGE blind study."""

from __future__ import annotations

import json

import numpy as np
import torch

from analyze_composition_results import metrics, typed_ranks
from compositional_kge import CompositionalComplEx, OUT, make_strict_inductive, read_compositions
from pair_interaction_kge import PairCompositionalComplEx
from train_baselines import Dataset

SPLIT = "pair_holdout"
SEEDS = (20260915, 20260916, 20260917)


def load_pair(data, arrays, variant, filename, device):
    model = PairCompositionalComplEx(data, variant, arrays).to(device)
    state = torch.load(OUT / filename, map_location=device, weights_only=True)
    model.load_state_dict(state["state_dict"])
    return model


def load_uniform(data, arrays, filename, device):
    model = CompositionalComplEx(data, "uniform_mean", arrays).to(device)
    state = torch.load(OUT / filename, map_location=device, weights_only=True)
    model.load_state_dict(state["state_dict"])
    return model


def bootstrap_rr(a, b, seed=20261004, repetitions=20000):
    """a/b are seed-by-query reciprocal ranks; resample queries after seed averaging."""
    delta = a.mean(0) - b.mean(0)
    rng = np.random.default_rng(seed)
    samples = np.empty(repetitions)
    for start in range(0, repetitions, 1000):
        k = min(1000, repetitions - start)
        idx = rng.integers(0, len(delta), size=(k, len(delta)))
        samples[start:start+k] = delta[idx].mean(1)
    lo, hi = np.quantile(samples, (0.025, 0.975))
    return {"queries": len(delta), "seeds": int(a.shape[0]),
            "mrr_difference": float(delta.mean()),
            "bootstrap_95ci": [float(lo), float(hi)],
            "probability_difference_le_zero": float(np.mean(samples <= 0)),
            "repetitions": repetitions}


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    device = torch.device("cuda")
    data = Dataset(SPLIT)
    arrays = read_compositions(data)
    make_strict_inductive(data)
    ranks = {"pair": [], "uniform": [], "deepsets": []}
    gates = []
    for seed in SEEDS:
        pair = load_pair(data, arrays, "pair_residual", f"pairblind_pair_seed{seed}.pt", device)
        uniform = load_uniform(data, arrays, f"pairblind_uniform_seed{seed}.pt", device)
        deepsets = load_pair(data, arrays, "deepsets_residual", f"pairblind_deepsets_seed{seed}.pt", device)
        ranks["pair"].append(typed_ranks(pair, data, data.test, device))
        ranks["uniform"].append(typed_ranks(uniform, data, data.test, device))
        ranks["deepsets"].append(typed_ranks(deepsets, data, data.test, device))
        gates.append(float(torch.sigmoid(pair.interaction_gate).item()))
    ranks = {k: np.stack(v) for k, v in ranks.items()}
    rr = {k: 1.0 / v for k, v in ranks.items()}
    report = {
        "seed_averaged": {k: {"mrr": float(v.mean()),
                                "per_seed_mrr": [float(x.mean()) for x in v]}
                          for k, v in rr.items()},
        "paired_bootstrap": {
            "pair_vs_uniform": bootstrap_rr(rr["pair"], rr["uniform"]),
            "pair_vs_deepsets": bootstrap_rr(rr["pair"], rr["deepsets"]),
            "deepsets_vs_uniform": bootstrap_rr(rr["deepsets"], rr["uniform"]),
        },
        "learned_pair_gate": {"per_seed": gates, "mean": float(np.mean(gates))},
        "by_formula_size": {},
        "by_relation": {},
    }
    mask = arrays[1]
    sizes = mask[data.test[:, 0]].sum(1)
    for name, subset in (("1_to_5", sizes <= 5),
                         ("6_to_10", (sizes >= 6) & (sizes <= 10)),
                         ("over_10", sizes > 10)):
        report["by_formula_size"][name] = {
            "n": int(subset.sum()),
            **{k + "_mrr": float(v[:, subset].mean()) for k, v in rr.items()},
            "pair_vs_uniform": bootstrap_rr(rr["pair"][:, subset], rr["uniform"][:, subset]),
            "pair_vs_deepsets": bootstrap_rr(rr["pair"][:, subset], rr["deepsets"][:, subset]),
        }
    relations = data.test[:, 1]
    for rid in sorted(set(map(int, relations))):
        subset = relations == rid
        report["by_relation"][data.relations[rid]] = {
            "n": int(subset.sum()),
            **{k + "_mrr": float(v[:, subset].mean()) for k, v in rr.items()}}
    (OUT / "pair_blind_detailed_analysis.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
