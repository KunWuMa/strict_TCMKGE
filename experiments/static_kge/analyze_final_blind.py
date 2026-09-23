# -*- coding: utf-8 -*-
"""Detailed, paired analysis of the untouched final composition benchmark."""

import json

import numpy as np
import torch

from analyze_composition_results import bootstrap, frequency_ranks, metrics, typed_ranks
from compositional_kge import CompositionalComplEx, OUT, drop_training_relations, make_strict_inductive, read_compositions
from train_baselines import ComplEx, Dataset

SPLIT = "composition_holdout"
DROPS = ("cpm_treats_icd11", "chp_has_property", "chp_from_species", "chp_contains_compound")


def comp_model(data, arrays, variant, name, device):
    state = torch.load(OUT/name, map_location=device, weights_only=True)
    model = CompositionalComplEx(data, variant, arrays).to(device); model.load_state_dict(state["state_dict"]); return model


def id_model(data, name, device):
    state = torch.load(OUT/name, map_location=device, weights_only=True)
    model = ComplEx(data.n_entities, data.n_relations, state["dim"]).to(device); model.load_state_dict(state["state_dict"]); return model


def main():
    if not torch.cuda.is_available(): raise RuntimeError("CUDA required")
    device = torch.device("cuda")
    data = Dataset(SPLIT); arrays = read_compositions(data); make_strict_inductive(data)
    ingredients, mask, uniform, dosage, complete = arrays
    assert mask[data.dev[:, 0]].any(1).all() and mask[data.test[:, 0]].any(1).all()
    uniform_model = comp_model(data, arrays, "uniform_mean", "final_uniform_mean_seed20260915.pt", device)
    dose_model = comp_model(data, arrays, "dose_mean", "final_dose_mean_seed20260915.pt", device)
    strict_model = id_model(data, "final_strict_id_complex.pt", device)
    r_uniform = typed_ranks(uniform_model, data, data.test, device)
    r_dose = typed_ranks(dose_model, data, data.test, device)
    r_strict = typed_ranks(strict_model, data, data.test, device)
    r_freq = frequency_ranks(data, data.test)

    target_data = Dataset(SPLIT); target_arrays = read_compositions(target_data); make_strict_inductive(target_data); drop_training_relations(target_data, DROPS)
    target_model = comp_model(target_data, target_arrays, "uniform_mean", "final_target_only_seed20260915.pt", device)
    r_target = typed_ranks(target_model, target_data, target_data.test, device)

    upper_data = Dataset(SPLIT); upper_arrays = read_compositions(upper_data)
    upper_model = id_model(upper_data, "final_context_upper_complex.pt", device)
    r_upper = typed_ranks(upper_model, upper_data, upper_data.test, device)

    ranks = {"uniform_mean": r_uniform, "dose_mean": r_dose, "frequency": r_freq,
             "strict_id_complex": r_strict, "target_only": r_target, "context_upper_complex": r_upper}
    report = {"overall": {name: metrics(values) for name, values in ranks.items()},
              "paired_bootstrap": {
                  "uniform_vs_frequency": bootstrap(r_uniform, r_freq),
                  "uniform_vs_strict_id": bootstrap(r_uniform, r_strict),
                  "uniform_vs_target_only": bootstrap(r_uniform, r_target),
                  "uniform_vs_dose": bootstrap(r_uniform, r_dose),
                  "uniform_vs_context_upper": bootstrap(r_uniform, r_upper),
              }, "by_relation": {}, "by_dosage_availability": {}, "by_formula_size": {}}
    relations = data.test[:, 1]
    for rid in sorted(set(map(int, relations))):
        subset = relations == rid
        report["by_relation"][data.relations[rid]] = {name: metrics(values[subset]) for name, values in ranks.items() if name in {"uniform_mean", "frequency", "target_only", "context_upper_complex"}}
        report["by_relation"][data.relations[rid]]["candidate_tails"] = len(data.range_candidates[rid])
    heads = data.test[:, 0]; has_dose = complete[heads]
    for name, subset in (("complete_dosage", has_dose), ("dosage_absent", ~has_dose)):
        report["by_dosage_availability"][name] = {"uniform_mean": metrics(r_uniform[subset]), "dose_mean": metrics(r_dose[subset]),
            "uniform_minus_dose": bootstrap(r_uniform[subset], r_dose[subset])}
    sizes = mask[heads].sum(1)
    for name, subset in (("1_to_5", sizes <= 5), ("6_to_10", (sizes >= 6)&(sizes <= 10)), ("over_10", sizes > 10)):
        report["by_formula_size"][name] = {"uniform_mean": metrics(r_uniform[subset]), "dose_mean": metrics(r_dose[subset])}
    report["integrity"] = {"test_edges": len(data.test), "test_heads": len(set(map(int, heads))),
        "all_test_heads_have_composition": bool(mask[heads].any(1).all()), "complete_dosage_edges": int(has_dose.sum())}
    (OUT/"final_blind_detailed_analysis.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
