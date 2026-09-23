# -*- coding: utf-8 -*-
"""Confirmatory three-seed experiment for frozen Hybrid PairComp-KGE."""

from __future__ import annotations

import json
import numpy as np
import torch

from compositional_kge import OUT, make_strict_inductive, read_compositions
from hybrid_pair_kge import HybridPairComplEx, train_hybrid
from pair_interaction_kge import PairCompositionalComplEx, train_pair
from train_baselines import Dataset, evaluate, frequency_baseline

SPLIT = "hybrid_holdout"
SEEDS = (20260915, 20260916, 20260917)
FAMILIES = {"hybrid": "hybrid_pair", "parameter_control": "dual_mean_control",
            "deepsets": "deepsets_residual", "pair_only": "pair_residual"}


def fresh():
    data = Dataset(SPLIT); arrays = read_compositions(data); integrity = make_strict_inductive(data)
    return data, arrays, integrity


def train_family(family, variant, data, arrays, device, seed):
    tag = f"hybridconfirm_{family}_seed{seed}.pt"
    if family in ("hybrid", "parameter_control"):
        return train_hybrid(data, variant, arrays, device, seed, tag)
    return train_pair(data, variant, arrays, device, seed, tag)


def load_family(family, variant, data, arrays, checkpoint, device):
    cls = HybridPairComplEx if family in ("hybrid", "parameter_control") else PairCompositionalComplEx
    model = cls(data, variant, arrays).to(device)
    state = torch.load(OUT/checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state["state_dict"])
    return model


def main():
    if not torch.cuda.is_available(): raise RuntimeError("CUDA required")
    device = torch.device("cuda")
    selections = {"split": SPLIT, "architecture_frozen": True,
                  "test_scored_after_all_selection": True, **{k: {} for k in FAMILIES}}
    for family, variant in FAMILIES.items():
        for seed in SEEDS:
            data, arrays, _ = fresh()
            selections[family][str(seed)] = train_family(family, variant, data, arrays, device, seed)
            (OUT/"hybrid_confirmatory_dev_selection.json").write_text(
                json.dumps(selections, indent=2), encoding="utf-8")

    results = {"split": SPLIT, "role": "same-corpus confirmatory resplit",
               "test_access": "after_all_checkpoint_selection", **{k: {} for k in FAMILIES}}
    _, _, integrity = fresh(); results["integrity"] = integrity
    for family, variant in FAMILIES.items():
        for seed in SEEDS:
            data, arrays, _ = fresh(); chosen = selections[family][str(seed)]
            model = load_family(family, variant, data, arrays, chosen["checkpoint"], device)
            results[family][str(seed)] = {
                "typed": evaluate(model, data, data.test, device, True),
                "untyped": evaluate(model, data, data.test, device, False),
                "selected": chosen["selected"], "parameters": chosen["parameters"]}
        results[family+"_summary"] = {
            metric: {"mean": float(np.mean([results[family][str(s)]["typed"][metric] for s in SEEDS])),
                     "sample_std": float(np.std([results[family][str(s)]["typed"][metric] for s in SEEDS], ddof=1))}
            for metric in ("mrr", "hits1", "hits3", "hits10")}
        print("CONFIRMATORY", family, results[family+"_summary"], flush=True)
    data, _, _ = fresh()
    results["frequency"] = {"typed": frequency_baseline(data, data.test, True),
                            "untyped": frequency_baseline(data, data.test, False)}
    (OUT/"hybrid_confirmatory_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("HYBRID CONFIRMATORY EXPERIMENT COMPLETE", flush=True)


if __name__ == "__main__":
    main()
