# -*- coding: utf-8 -*-
"""Per-edge analyses and paired bootstrap for the strict composition experiment."""

from __future__ import annotations

import json
from collections import Counter, defaultdict

import numpy as np
import torch

from compositional_kge import (OUT, CompositionalComplEx, drop_training_relations,
                               make_strict_inductive, read_compositions)
from train_baselines import Dataset


@torch.inference_mode()
def typed_ranks(model, data, triples, device):
    model.eval(); ranks = []
    for h, r, t in triples:
        candidates = data.range_candidates[int(r)]
        tails = torch.as_tensor(candidates, device=device)
        scores = model.score_tails(torch.tensor(int(h), device=device), torch.tensor(int(r), device=device), tails).float().cpu().numpy()
        known = data.all_true[(int(h), int(r))] - {int(t)}
        if known:
            scores[np.isin(candidates, np.fromiter(known, dtype=np.int64))] = -np.inf
        pos = int(np.searchsorted(candidates, int(t)))
        target = scores[pos]
        ranks.append(1.0 + np.sum(scores > target) + 0.5 * max(0, int(np.sum(scores == target))-1))
    return np.asarray(ranks, dtype=np.float64)


def frequency_ranks(data, triples):
    counts = defaultdict(Counter)
    for _, r, t in data.train:
        counts[int(r)][int(t)] += 1
    ranks = []
    for h, r, t in triples:
        candidates = data.range_candidates[int(r)]
        scores = np.asarray([counts[int(r)][int(x)] for x in candidates], dtype=np.float64)
        known = data.all_true[(int(h), int(r))] - {int(t)}
        if known:
            scores[np.isin(candidates, np.fromiter(known, dtype=np.int64))] = -np.inf
        pos = int(np.searchsorted(candidates, int(t))); target = scores[pos]
        ranks.append(1.0 + np.sum(scores > target) + 0.5 * max(0, int(np.sum(scores == target))-1))
    return np.asarray(ranks, dtype=np.float64)


def metrics(ranks):
    return {"n": len(ranks), "mrr": float(np.mean(1/ranks)), "hits1": float(np.mean(ranks <= 1)),
            "hits3": float(np.mean(ranks <= 3)), "hits10": float(np.mean(ranks <= 10)),
            "mean_rank": float(np.mean(ranks))}


def bootstrap(a, b, seed=20260915, repetitions=20000):
    rng = np.random.default_rng(seed); delta = 1/a - 1/b; n = len(delta)
    samples = np.empty(repetitions)
    for start in range(0, repetitions, 1000):
        size = min(1000, repetitions-start)
        indices = rng.integers(0, n, size=(size, n))
        samples[start:start+size] = delta[indices].mean(1)
    lo, hi = np.quantile(samples, [0.025, 0.975])
    return {"n": n, "mrr_difference": float(delta.mean()), "bootstrap_95ci": [float(lo), float(hi)],
            "probability_difference_le_zero": float(np.mean(samples <= 0)), "repetitions": repetitions}


def load_model(data, arrays, variant, filename, device):
    model = CompositionalComplEx(data, variant, arrays).to(device)
    state = torch.load(OUT / filename, map_location=device, weights_only=True)
    model.load_state_dict(state["state_dict"])
    return model


def main():
    if not torch.cuda.is_available(): raise RuntimeError("CUDA is required")
    device = torch.device("cuda")
    data = Dataset("formula_holdout"); arrays = read_compositions(data); make_strict_inductive(data)
    ingredients, mask, uniform, dosage, complete = arrays
    uniform_model = load_model(data, arrays, "uniform_mean", "strict_composition_uniform_mean_seed20260915.pt", device)
    dose_model = load_model(data, arrays, "dose_mean", "strict_composition_dose_mean_seed20260915.pt", device)
    uniform_r = typed_ranks(uniform_model, data, data.test, device)
    dose_r = typed_ranks(dose_model, data, data.test, device)
    freq_r = frequency_ranks(data, data.test)

    # Target-only uses a different training graph and therefore its own candidate/filter indexes.
    target_data = Dataset("formula_holdout"); target_arrays = read_compositions(target_data); make_strict_inductive(target_data)
    drops = ("cpm_treats_icd11", "chp_has_property", "chp_from_species", "chp_contains_compound")
    drop_training_relations(target_data, drops)
    target_model = load_model(target_data, target_arrays, "uniform_mean", "strict_ablation_target_only_seed20260915.pt", device)
    target_r = typed_ranks(target_model, target_data, target_data.test, device)

    report = {
        "overall": {"uniform_mean": metrics(uniform_r), "dose_mean": metrics(dose_r),
                    "frequency": metrics(freq_r), "target_only": metrics(target_r)},
        "paired_bootstrap": {"uniform_vs_frequency": bootstrap(uniform_r, freq_r),
                             "uniform_vs_target_only": bootstrap(uniform_r, target_r)},
        "by_relation": {}, "by_dosage_availability": {}, "by_formula_size": {},
    }
    relations = data.test[:, 1]
    for relation_id in sorted(set(map(int, relations))):
        subset = relations == relation_id
        report["by_relation"][data.relations[relation_id]] = {
            "uniform_mean": metrics(uniform_r[subset]), "dose_mean": metrics(dose_r[subset]),
            "frequency": metrics(freq_r[subset]), "candidate_tails": len(data.range_candidates[relation_id])}

    test_heads = data.test[:, 0]
    available = complete[test_heads]
    for name, subset in (("complete_dosage", available), ("dosage_absent", ~available)):
        report["by_dosage_availability"][name] = {
            "uniform_mean": metrics(uniform_r[subset]), "dose_mean": metrics(dose_r[subset]),
            "dose_minus_uniform": bootstrap(dose_r[subset], uniform_r[subset])}

    sizes = mask[test_heads].sum(1)
    for name, subset in (("1_to_5", sizes <= 5), ("6_to_10", (sizes >= 6) & (sizes <= 10)), ("over_10", sizes > 10)):
        report["by_formula_size"][name] = {"uniform_mean": metrics(uniform_r[subset]),
                                                   "dose_mean": metrics(dose_r[subset])}
    report["integrity"] = {"test_edges": len(data.test), "unique_test_heads": len(set(map(int, test_heads))),
                           "all_strict_test_heads_have_composition": bool(mask[test_heads].any(1).all()),
                           "complete_dosage_test_edges": int(available.sum())}
    (OUT / "composition_detailed_analysis.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
