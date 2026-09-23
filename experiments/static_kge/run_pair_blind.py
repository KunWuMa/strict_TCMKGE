# -*- coding: utf-8 -*-
"""Frozen PairComp-KGE confirmation experiment on the new blind split."""

from __future__ import annotations

import json

import numpy as np
import torch

from compositional_kge import CompositionalComplEx, OUT, make_strict_inductive, read_compositions
from pair_interaction_kge import PairCompositionalComplEx, train_pair
from train_baselines import ComplEx, Dataset, evaluate, frequency_baseline, train_one as train_standard
from compositional_kge import train_one as train_composition

SPLIT = "pair_holdout"
SEEDS = (20260915, 20260916, 20260917)


def fresh(strict=True):
    data = Dataset(SPLIT)
    arrays = read_compositions(data)
    info = make_strict_inductive(data) if strict else {"train": len(data.train)}
    return data, arrays, info


def load_composition(data, arrays, variant, checkpoint, device):
    model = CompositionalComplEx(data, variant, arrays).to(device)
    state = torch.load(OUT / checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state["state_dict"])
    return model


def load_pair(data, arrays, variant, checkpoint, device):
    model = PairCompositionalComplEx(data, variant, arrays).to(device)
    state = torch.load(OUT / checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state["state_dict"])
    return model


def load_standard(data, checkpoint, device):
    state = torch.load(OUT / checkpoint, map_location=device, weights_only=True)
    model = ComplEx(data.n_entities, data.n_relations, state["dim"]).to(device)
    model.load_state_dict(state["state_dict"])
    return model


def score(model, data, device):
    return {"typed": evaluate(model, data, data.test, device, True),
            "untyped": evaluate(model, data, data.test, device, False)}


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    device = torch.device("cuda")
    selections = {"split": SPLIT, "frozen_before_split": "pair_residual",
                  "pair": {}, "uniform": {}, "deepsets_control": {}, "baselines": {}}

    # Select all checkpoints using train and dev only.
    for seed in SEEDS:
        data, arrays, _ = fresh(True)
        selections["pair"][str(seed)] = train_pair(
            data, "pair_residual", arrays, device, seed,
            tag=f"pairblind_pair_seed{seed}.pt")
        (OUT / "pair_blind_dev_selection.json").write_text(json.dumps(selections, indent=2), encoding="utf-8")
    for seed in SEEDS:
        data, arrays, _ = fresh(True)
        selections["uniform"][str(seed)] = train_composition(
            data, "uniform_mean", arrays, device, seed,
            tag=f"pairblind_uniform_seed{seed}.pt")
        (OUT / "pair_blind_dev_selection.json").write_text(json.dumps(selections, indent=2), encoding="utf-8")
    for seed in SEEDS:
        data, arrays, _ = fresh(True)
        selections["deepsets_control"][str(seed)] = train_pair(
            data, "deepsets_residual", arrays, device, seed,
            tag=f"pairblind_deepsets_seed{seed}.pt")
        (OUT / "pair_blind_dev_selection.json").write_text(json.dumps(selections, indent=2), encoding="utf-8")

    data, _, strict_info = fresh(True)
    selections["baselines"]["strict_id_complex"] = train_standard(
        SPLIT, "complex", data, device, tag="pairblind_strict_id")
    data, _, _ = fresh(False)
    selections["baselines"]["context_upper_complex"] = train_standard(
        SPLIT, "complex", data, device, tag="pairblind_context_upper")
    (OUT / "pair_blind_dev_selection.json").write_text(json.dumps(selections, indent=2), encoding="utf-8")

    # First test access occurs below, after architecture and every checkpoint are frozen.
    results = {"split": SPLIT, "test_access": "after_all_checkpoint_selection",
               "pair": {}, "uniform": {}, "deepsets_control": {}, "baselines": {},
               "integrity": strict_info}
    for seed in SEEDS:
        data, arrays, _ = fresh(True)
        chosen = selections["pair"][str(seed)]
        results["pair"][str(seed)] = {**score(load_pair(data, arrays, "pair_residual", chosen["checkpoint"], device), data, device),
                                        "selected": chosen["selected"], "parameters": chosen["parameters"]}
        chosen = selections["uniform"][str(seed)]
        results["uniform"][str(seed)] = {**score(load_composition(data, arrays, "uniform_mean", chosen["checkpoint"], device), data, device),
                                           "selected": chosen["selected"], "parameters": chosen["parameters"]}
        chosen = selections["deepsets_control"][str(seed)]
        results["deepsets_control"][str(seed)] = {**score(load_pair(data, arrays, "deepsets_residual", chosen["checkpoint"], device), data, device),
                                                    "selected": chosen["selected"], "parameters": chosen["parameters"]}
        print("BLIND TEST SEED", seed, {k: results[k][str(seed)]["typed"] for k in ("pair", "uniform", "deepsets_control")}, flush=True)

    keys = ("mrr", "hits1", "hits3", "hits10")
    for family in ("pair", "uniform", "deepsets_control"):
        results[family + "_summary"] = {
            key: {"mean": float(np.mean([results[family][str(s)]["typed"][key] for s in SEEDS])),
                  "sample_std": float(np.std([results[family][str(s)]["typed"][key] for s in SEEDS], ddof=1))}
            for key in keys}

    data, _, _ = fresh(True)
    chosen = selections["baselines"]["strict_id_complex"]
    results["baselines"]["strict_id_complex"] = {
        **score(load_standard(data, chosen["checkpoint"], device), data, device), "selected": chosen["selected"]}
    results["baselines"]["frequency"] = {
        "typed": frequency_baseline(data, data.test, True),
        "untyped": frequency_baseline(data, data.test, False)}
    data, _, _ = fresh(False)
    chosen = selections["baselines"]["context_upper_complex"]
    results["baselines"]["context_upper_complex"] = {
        **score(load_standard(data, chosen["checkpoint"], device), data, device), "selected": chosen["selected"]}
    (OUT / "pair_blind_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("PAIR BLIND EXPERIMENT COMPLETE", flush=True)


if __name__ == "__main__":
    main()
