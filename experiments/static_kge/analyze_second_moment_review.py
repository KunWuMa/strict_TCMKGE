# -*- coding: utf-8 -*-
"""Formula-clustered uncertainty for the second-moment review experiment."""

from __future__ import annotations

import json

import numpy as np

from compositional_kge import OUT, make_strict_inductive, read_compositions
from train_baselines import Dataset


def cluster_interval(a, b, heads, metric, seed=20261021, repetitions=20000):
    """Average matched seeds per query, then resample formula heads."""
    assert a.shape == b.shape and a.shape[1] == len(heads)
    if metric == "mrr":
        values = 1.0 / a - 1.0 / b
    elif metric == "hits1":
        values = (a <= 1).astype(float) - (b <= 1).astype(float)
    else:
        raise ValueError(metric)
    delta = values.mean(0)
    _, inverse = np.unique(heads, return_inverse=True)
    counts = np.bincount(inverse)
    sums = np.bincount(inverse, weights=delta)
    rng = np.random.default_rng(seed)
    samples = np.empty(repetitions)
    for start in range(0, repetitions, 500):
        idx = rng.integers(0, len(counts), size=(min(500, repetitions-start), len(counts)))
        samples[start:start+len(idx)] = sums[idx].sum(1) / counts[idx].sum(1)
    lo, hi = np.quantile(samples, (0.025, 0.975))
    return {"difference": float(delta.mean()), "cluster_bootstrap_95ci": [float(lo), float(hi)],
            "clusters": len(counts), "queries": len(heads), "seeds": a.shape[0],
            "repetitions": repetitions}


def main():
    with np.load(OUT / "second_moment_review_ranks.npz") as f:
        heads = f["heads"].copy()
        ranks = {k: f[k].copy() for k in ("paircomp", "mean_complex", "matched", "second_moment")}
    report = {"five_seed_direct_comparison": {}, "three_seed_direct_comparison": {},
              "three_seed_deepsets": {}, "by_formula_size": {},
              "moment_information_equivalence": "q_f = n_f mu_f^2 - (n_f-1) p_f for n_f > 1"}
    for other in ("mean_complex", "matched", "second_moment"):
        report["five_seed_direct_comparison"][f"paircomp_vs_{other}"] = {
            metric: cluster_interval(ranks["paircomp"], ranks[other], heads, metric,
                                     seed=20261021 + i)
            for i, metric in enumerate(("mrr", "hits1"))}
        report["three_seed_direct_comparison"][f"paircomp_vs_{other}"] = {
            metric: cluster_interval(ranks["paircomp"][:3], ranks[other][:3], heads,
                                     metric, seed=20261024 + i)
            for i, metric in enumerate(("mrr", "hits1"))}
    data = Dataset("hybrid_holdout")
    arrays = read_compositions(data)
    make_strict_inductive(data)
    assert np.array_equal(heads, data.test[:, 0])
    sizes = arrays[1][heads].sum(1)
    for label, mask in (("singleton", sizes == 1), ("2_to_5", (sizes >= 2) & (sizes <= 5)),
                        ("6_to_10", (sizes >= 6) & (sizes <= 10)), ("over_10", sizes > 10)):
        if not mask.any():
            continue
        report["by_formula_size"][label] = {
            "queries": int(mask.sum()),
            **{family: {"mrr": float((1.0 / value[:3, mask]).mean()),
                        "hits1": float((value[:3, mask] <= 1).mean())}
               for family, value in ranks.items()}}
    deep_path = OUT / "true_deepsets_review_ranks.npz"
    if deep_path.exists():
        with np.load(deep_path) as f:
            assert np.array_equal(heads, f["heads"])
            deep = f["ranks"].copy()
        report["three_seed_deepsets"] = {
            metric: cluster_interval(ranks["paircomp"][:3], deep, heads, metric,
                                     seed=20261031 + i)
            for i, metric in enumerate(("mrr", "hits1"))}
        for label, mask in (("singleton", sizes == 1), ("2_to_5", (sizes >= 2) & (sizes <= 5)),
                            ("6_to_10", (sizes >= 6) & (sizes <= 10)), ("over_10", sizes > 10)):
            if mask.any():
                report["by_formula_size"][label]["true_deepsets"] = {
                    "mrr": float((1.0 / deep[:, mask]).mean()),
                    "hits1": float((deep[:, mask] <= 1).mean())}
    path = OUT / "second_moment_review_analysis.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
