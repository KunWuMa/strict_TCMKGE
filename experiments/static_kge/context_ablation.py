# -*- coding: utf-8 -*-
"""Run the preregistered context ablation on formula-held-out completion."""

from __future__ import annotations

import json

import torch

import train_baselines as k


VARIANTS = {
    "no_formula_herb": ["cpm_contains_chp"],
    "no_icd": ["cpm_treats_icd11"],
    "herb_identity_only": [
        "cpm_treats_icd11", "chp_has_property", "chp_from_species", "chp_contains_compound"
    ],
    "no_formula_context": ["cpm_contains_chp", "cpm_treats_icd11"],
}


def head_support(data: k.Dataset) -> dict:
    target_heads = {int(h) for h, _, _ in data.test}
    incident = set()
    for h, _, t in data.train:
        if int(h) in target_heads: incident.add(int(h))
        if int(t) in target_heads: incident.add(int(t))
    return {"test_heads": len(target_heads), "test_heads_with_train_incident_edge": len(incident)}


def main() -> None:
    if not torch.cuda.is_available(): raise RuntimeError("CUDA required")
    device = torch.device("cuda"); selections = {}; datasets = {}
    for name, dropped in VARIANTS.items():
        data = k.Dataset("formula_holdout", dropped); datasets[name] = data
        selections[name] = k.train_one("formula_holdout", "complex", data, device, tag="ablation_" + name)
        selections[name]["dropped_relations"] = dropped; selections[name]["head_support"] = head_support(data)
        (k.OUT / "context_ablation_dev.json").write_text(json.dumps(selections, indent=2), encoding="utf-8")

    results = {}
    full = json.loads((k.OUT / "results.json").read_text(encoding="utf-8"))["formula_holdout"]["complex"]
    results["full"] = full
    for name, data in datasets.items():
        checkpoint = torch.load(k.OUT / selections[name]["checkpoint"], map_location=device, weights_only=True)
        model = k.ComplEx(data.n_entities, data.n_relations, checkpoint["dim"]).to(device)
        model.load_state_dict(checkpoint["state_dict"])
        results[name] = {
            "typed": k.evaluate(model, data, data.test, device, True),
            "untyped": k.evaluate(model, data, data.test, device, False),
            "selected": selections[name]["selected"],
            "dropped_relations": selections[name]["dropped_relations"],
            "head_support": selections[name]["head_support"],
        }
        print("TEST", name, results[name], flush=True)
    (k.OUT / "context_ablation_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("CONTEXT ABLATION COMPLETE", flush=True)


if __name__ == "__main__":
    main()
