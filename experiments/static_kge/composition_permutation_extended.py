# -*- coding: utf-8 -*-
"""Extended size-matched wrong-composition randomization test (999 permutations)."""

from __future__ import annotations

import json

import numpy as np
import torch

from analyze_hybrid_confirmatory import load
from compositional_kge import OUT, make_strict_inductive, read_compositions
from expert_systems_robustness import metrics, perturbed_arrays
from train_baselines import Dataset

SEEDS = (20260915, 20260916, 20260917)
N_PERMUTATIONS = 999


@torch.inference_mode()
def fast_typed_ranks(model, data, triples, device):
    """Vectorized equivalent of typed_ranks, grouped by relation."""
    model.eval()
    unique_heads, inverse = np.unique(triples[:, 0], return_inverse=True)
    head_tensor = torch.as_tensor(unique_heads, device=device)
    hr_all, hi_all = model.head_embedding(head_tensor, torch.zeros_like(head_tensor))
    ranks = np.empty(len(triples), dtype=np.float64)
    for rid in np.unique(triples[:, 1]):
        rows = np.flatnonzero(triples[:, 1] == rid)
        local = torch.as_tensor(inverse[rows], device=device)
        hr, hi = hr_all[local], hi_all[local]
        relation = torch.full((len(rows),), int(rid), device=device, dtype=torch.long)
        rr, ri = model.rr(relation), model.ri(relation)
        qr, qi = hr * rr - hi * ri, hi * rr + hr * ri
        candidates = data.range_candidates[int(rid)]
        tails = torch.as_tensor(candidates, device=device)
        scores = (qr @ model.er(tails).T + qi @ model.ei(tails).T).float().cpu().numpy()
        for j, row in enumerate(rows):
            h, _, t = map(int, triples[row])
            known = data.all_true[(h, int(rid))] - {t}
            if known:
                scores[j, np.isin(candidates, np.fromiter(known, dtype=np.int64))] = -np.inf
            pos = int(np.searchsorted(candidates, t))
            target = scores[j, pos]
            ranks[row] = 1.0 + np.sum(scores[j] > target) + 0.5 * max(0, int(np.sum(scores[j] == target)) - 1)
    return ranks


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    device = torch.device("cuda")
    data = Dataset("hybrid_holdout")
    arrays = read_compositions(data)
    make_strict_inductive(data)
    heads = data.test[:, 0]
    models = [load("hybrid", data, arrays, seed, device) for seed in SEEDS]
    observed_ranks = np.stack([
        fast_typed_ranks(model, data, data.test, device) for model in models
    ])
    observed = metrics(observed_ranks)["mrr"]
    null = []
    for repetition in range(N_PERMUTATIONS):
        altered = perturbed_arrays(
            arrays, heads, np.random.default_rng(20261400 + repetition), permute=True
        )
        ing, mask, uniform, _, _ = altered
        rr_rows = []
        for model in models:
            model.ingredients.copy_(torch.from_numpy(ing).to(device))
            model.ingredient_mask.copy_(torch.from_numpy(mask).to(device))
            model.uniform_weight.copy_(torch.from_numpy(uniform).to(device))
            rr_rows.append(fast_typed_ranks(model, data, data.test, device))
        rr = np.stack(rr_rows)
        value = metrics(rr)["mrr"]
        null.append(value)
        print(f"permutation {repetition + 1}/{N_PERMUTATIONS}: {value:.6f}", flush=True)
    null_array = np.asarray(null)
    report = {
        "observed_mrr": observed,
        "permutations": N_PERMUTATIONS,
        "null_mrr_mean": float(null_array.mean()),
        "null_mrr_std": float(null_array.std(ddof=1)),
        "null_mrr_95percentile": float(np.quantile(null_array, 0.95)),
        "null_mrr_values": null,
        "empirical_p_upper": float((1 + np.sum(null_array >= observed)) / (N_PERMUTATIONS + 1)),
        "preserved": "formula size, test queries, and trained model",
        "permuted": "ingredient set among exact-size formulas",
        "seed_aggregation": "three-model-seed mean before permutation comparison",
    }
    (OUT / "composition_permutation_extended.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
