# -*- coding: utf-8 -*-
"""Frozen-model permutation control: shuffle compositions among equal-size test formulas."""

import json
from collections import defaultdict

import numpy as np
import torch

from compositional_kge import CompositionalComplEx, OUT, make_strict_inductive, read_compositions
from train_baselines import Dataset, evaluate

SPLIT = "composition_holdout"
REPETITIONS = 50
SEED = 20261002


def main():
    if not torch.cuda.is_available(): raise RuntimeError("CUDA required")
    device = torch.device("cuda")
    data = Dataset(SPLIT); arrays = read_compositions(data); make_strict_inductive(data)
    model = CompositionalComplEx(data, "uniform_mean", arrays).to(device)
    state = torch.load(OUT/"final_uniform_mean_seed20260915.pt", map_location=device, weights_only=True)
    model.load_state_dict(state["state_dict"])
    observed = evaluate(model, data, data.test, device, True)["mrr"]
    heads = np.asarray(sorted(set(map(int, data.test[:, 0]))), dtype=np.int64)
    sizes = model.ingredient_mask[torch.as_tensor(heads, device=device)].sum(1).cpu().numpy()
    groups = defaultdict(list)
    for head, size in zip(heads, sizes): groups[int(size)].append(int(head))
    original_i = model.ingredients.clone(); original_m = model.ingredient_mask.clone()
    original_u = model.uniform_weight.clone(); original_d = model.dosage_weight.clone()
    rng = np.random.default_rng(SEED); null = []; changed = []
    for _ in range(REPETITIONS):
        source = heads.copy()
        changed_count = 0
        for members in groups.values():
            if len(members) < 2: continue
            shuffled = np.asarray(members)[rng.permutation(len(members))]
            if np.array_equal(shuffled, members): shuffled = np.roll(shuffled, 1)
            for target, donor in zip(members, shuffled):
                source[np.where(heads == target)[0][0]] = donor
                changed_count += target != donor
        target_t = torch.as_tensor(heads, device=device); source_t = torch.as_tensor(source, device=device)
        model.ingredients[target_t] = original_i[source_t]
        model.ingredient_mask[target_t] = original_m[source_t]
        model.uniform_weight[target_t] = original_u[source_t]
        model.dosage_weight[target_t] = original_d[source_t]
        null.append(evaluate(model, data, data.test, device, True)["mrr"]); changed.append(changed_count/len(heads))
    model.ingredients.copy_(original_i); model.ingredient_mask.copy_(original_m)
    model.uniform_weight.copy_(original_u); model.dosage_weight.copy_(original_d)
    null = np.asarray(null)
    report = {"observed_mrr": observed, "null_repetitions": REPETITIONS,
              "null_mean_mrr": float(null.mean()), "null_std_mrr": float(null.std(ddof=1)),
              "null_min_mrr": float(null.min()), "null_max_mrr": float(null.max()),
              "empirical_p_ge_observed": float((1 + np.sum(null >= observed))/(REPETITIONS+1)),
              "mean_fraction_test_heads_changed": float(np.mean(changed)),
              "control": "compositions permuted among held-out formulas with exactly equal ingredient counts"}
    (OUT/"composition_permutation_test.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__": main()
