# -*- coding: utf-8 -*-
"""Execute every frozen model on the final composition-only eligible split."""

import json
import numpy as np
import torch

from compositional_kge import (OUT, CompositionalComplEx, drop_training_relations,
                               make_strict_inductive, read_compositions, train_one as train_composition)
from train_baselines import (ComplEx, Dataset, evaluate, frequency_baseline,
                             train_one as train_standard)

SPLIT = "composition_holdout"
SEEDS = (20260915, 20260916, 20260917)
SINGLE_VARIANTS = ("dose_mean", "relation_attention", "dose_relation_attention",
                   "residual_attention", "dose_residual_attention")
TARGET_ONLY_DROPS = ("cpm_treats_icd11", "chp_has_property", "chp_from_species", "chp_contains_compound")


def fresh(strict=True, drops=()):
    data = Dataset(SPLIT); arrays = read_compositions(data)
    info = make_strict_inductive(data) if strict else {"train": len(data.train)}
    if drops: drop_training_relations(data, drops)
    return data, arrays, info


def load_comp(data, arrays, variant, checkpoint, device):
    model = CompositionalComplEx(data, variant, arrays).to(device)
    state = torch.load(OUT / checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state["state_dict"]); return model


def load_standard(data, checkpoint, device):
    state = torch.load(OUT / checkpoint, map_location=device, weights_only=True)
    model = ComplEx(data.n_entities, data.n_relations, state["dim"]).to(device)
    model.load_state_dict(state["state_dict"]); return model


def score(model, data, device):
    return {"typed": evaluate(model, data, data.test, device, True),
            "untyped": evaluate(model, data, data.test, device, False)}


def main():
    if not torch.cuda.is_available(): raise RuntimeError("CUDA is required")
    device = torch.device("cuda")
    selections = {"split": SPLIT, "primary": {}, "variants": {}, "baselines": {}}

    # Frozen primary model, three seeds.
    for seed in SEEDS:
        data, arrays, info = fresh(strict=True)
        assert arrays[1][data.dev[:, 0]].any(1).all() and arrays[1][data.test[:, 0]].any(1).all()
        chosen = train_composition(data, "uniform_mean", arrays, device, seed=seed,
                                   tag=f"final_uniform_mean_seed{seed}.pt")
        selections["primary"][str(seed)] = chosen
        (OUT / "final_blind_dev_selection.json").write_text(json.dumps(selections, indent=2), encoding="utf-8")

    for variant in SINGLE_VARIANTS:
        data, arrays, _ = fresh(strict=True)
        chosen = train_composition(data, variant, arrays, device, seed=SEEDS[0],
                                   tag=f"final_{variant}_seed{SEEDS[0]}.pt")
        selections["variants"][variant] = chosen
        (OUT / "final_blind_dev_selection.json").write_text(json.dumps(selections, indent=2), encoding="utf-8")

    data, arrays, _ = fresh(strict=True, drops=TARGET_ONLY_DROPS)
    chosen = train_composition(data, "uniform_mean", arrays, device, seed=SEEDS[0],
                               tag=f"final_target_only_seed{SEEDS[0]}.pt")
    chosen["dropped_relations"] = list(TARGET_ONLY_DROPS)
    selections["variants"]["target_only"] = chosen

    strict_data, _, strict_info = fresh(strict=True)
    selections["baselines"]["strict_id_complex"] = train_standard(
        SPLIT, "complex", strict_data, device, tag="final_strict_id")
    full_data, _, _ = fresh(strict=False)
    selections["baselines"]["context_upper_complex"] = train_standard(
        SPLIT, "complex", full_data, device, tag="final_context_upper")
    (OUT / "final_blind_dev_selection.json").write_text(json.dumps(selections, indent=2), encoding="utf-8")

    # Test is first touched here, after every checkpoint has been selected.
    results = {"split": SPLIT, "primary": {}, "variants": {}, "baselines": {}, "integrity": strict_info}
    for seed in SEEDS:
        data, arrays, _ = fresh(strict=True)
        chosen = selections["primary"][str(seed)]
        results["primary"][str(seed)] = {**score(load_comp(data, arrays, "uniform_mean", chosen["checkpoint"], device), data, device),
                                            "selected": chosen["selected"]}
        print("FINAL PRIMARY TEST", seed, results["primary"][str(seed)], flush=True)
    keys = ("mrr", "hits1", "hits3", "hits10")
    results["primary_summary"] = {key: {"mean": float(np.mean([results["primary"][str(s)]["typed"][key] for s in SEEDS])),
                                               "sample_std": float(np.std([results["primary"][str(s)]["typed"][key] for s in SEEDS], ddof=1))}
                                  for key in keys}
    for variant in SINGLE_VARIANTS:
        data, arrays, _ = fresh(strict=True); chosen = selections["variants"][variant]
        results["variants"][variant] = {**score(load_comp(data, arrays, variant, chosen["checkpoint"], device), data, device),
                                           "selected": chosen["selected"]}
        print("FINAL VARIANT TEST", variant, results["variants"][variant], flush=True)
    data, arrays, _ = fresh(strict=True, drops=TARGET_ONLY_DROPS); chosen = selections["variants"]["target_only"]
    results["variants"]["target_only"] = {**score(load_comp(data, arrays, "uniform_mean", chosen["checkpoint"], device), data, device),
                                              "selected": chosen["selected"]}
    data, _, _ = fresh(strict=True); chosen = selections["baselines"]["strict_id_complex"]
    results["baselines"]["strict_id_complex"] = {**score(load_standard(data, chosen["checkpoint"], device), data, device), "selected": chosen["selected"]}
    results["baselines"]["frequency"] = {"typed": frequency_baseline(data, data.test, True), "untyped": frequency_baseline(data, data.test, False)}
    data, _, _ = fresh(strict=False); chosen = selections["baselines"]["context_upper_complex"]
    results["baselines"]["context_upper_complex"] = {**score(load_standard(data, chosen["checkpoint"], device), data, device), "selected": chosen["selected"]}
    (OUT / "final_blind_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("FINAL BLIND EXPERIMENT COMPLETE", flush=True)


if __name__ == "__main__":
    main()
