# -*- coding: utf-8 -*-
"""Three-seed confirmation and frozen auxiliary-relation ablations."""

import json
from pathlib import Path

import numpy as np
import torch

from compositional_kge import (OUT, CompositionalComplEx, drop_training_relations,
                               make_strict_inductive, read_compositions, train_one)
from train_baselines import Dataset, evaluate

SEEDS = (20260915, 20260916, 20260917)
ABLATIONS = {
    "no_icd": ("cpm_treats_icd11",),
    "no_herb_attributes": ("chp_has_property", "chp_from_species", "chp_contains_compound"),
    "target_only": ("cpm_treats_icd11", "chp_has_property", "chp_from_species", "chp_contains_compound"),
}


def load_and_test(data, arrays, chosen, device, seed, variant="uniform_mean"):
    model = CompositionalComplEx(data, variant, arrays).to(device)
    state = torch.load(OUT / chosen["checkpoint"], map_location=device, weights_only=True)
    model.load_state_dict(state["state_dict"])
    return {"typed": evaluate(model, data, data.test, device, True),
            "untyped": evaluate(model, data, data.test, device, False),
            "selected": chosen["selected"], "seed": seed}


def fresh(drops=()):
    data = Dataset("formula_holdout")
    arrays = read_compositions(data)
    strict = make_strict_inductive(data)
    if drops:
        drop_training_relations(data, drops)
    return data, arrays, strict


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda")
    selections = {"seeds": {}, "ablations": {}}

    # Reuse the already selected first seed; train only the two new seeds.
    prior = json.loads((OUT / "composition_dev_selection.json").read_text(encoding="utf-8"))
    selections["seeds"][str(SEEDS[0])] = prior["models"]["uniform_mean"]
    for seed in SEEDS[1:]:
        data, arrays, _ = fresh()
        chosen = train_one(data, "uniform_mean", arrays, device, seed=seed)
        selections["seeds"][str(seed)] = chosen
        (OUT / "confirmatory_dev_selection.json").write_text(json.dumps(selections, indent=2), encoding="utf-8")

    for name, drops in ABLATIONS.items():
        data, arrays, _ = fresh(drops)
        chosen = train_one(data, "uniform_mean", arrays, device, seed=SEEDS[0],
                           tag=f"strict_ablation_{name}_seed{SEEDS[0]}.pt")
        chosen["dropped_relations"] = list(drops)
        chosen["train_edges"] = len(data.train)
        selections["ablations"][name] = chosen
        (OUT / "confirmatory_dev_selection.json").write_text(json.dumps(selections, indent=2), encoding="utf-8")

    # Test only after every selection above is frozen.
    results = {"seeds": {}, "ablations": {}}
    for seed in SEEDS:
        data, arrays, _ = fresh()
        result = load_and_test(data, arrays, selections["seeds"][str(seed)], device, seed)
        results["seeds"][str(seed)] = result
        print("SEED TEST", seed, result, flush=True)
    metrics = ("mrr", "hits1", "hits3", "hits10")
    results["seed_summary"] = {
        metric: {"mean": float(np.mean([results["seeds"][str(s)]["typed"][metric] for s in SEEDS])),
                 "sample_std": float(np.std([results["seeds"][str(s)]["typed"][metric] for s in SEEDS], ddof=1))}
        for metric in metrics
    }
    for name, drops in ABLATIONS.items():
        data, arrays, _ = fresh(drops)
        result = load_and_test(data, arrays, selections["ablations"][name], device, SEEDS[0])
        result["dropped_relations"] = list(drops)
        results["ablations"][name] = result
        print("ABLATION TEST", name, result, flush=True)
    (OUT / "confirmatory_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("CONFIRMATORY EXPERIMENTS COMPLETE", flush=True)


if __name__ == "__main__":
    main()
