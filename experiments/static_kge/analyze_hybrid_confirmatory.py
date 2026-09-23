# -*- coding: utf-8 -*-
"""Paired query-level analysis of Hybrid PairComp-KGE confirmation results."""

from __future__ import annotations

import json
import numpy as np
import torch

from analyze_composition_results import typed_ranks
from compositional_kge import OUT, make_strict_inductive, read_compositions
from hybrid_pair_kge import HybridPairComplEx
from pair_interaction_kge import PairCompositionalComplEx
from train_baselines import Dataset

SPLIT = "hybrid_holdout"
SEEDS = (20260915, 20260916, 20260917)
FAMILIES = {"hybrid": (HybridPairComplEx, "hybrid_pair"),
            "parameter_control": (HybridPairComplEx, "dual_mean_control"),
            "deepsets": (PairCompositionalComplEx, "deepsets_residual"),
            "pair_only": (PairCompositionalComplEx, "pair_residual")}


def load(family, data, arrays, seed, device):
    cls, variant = FAMILIES[family]
    model = cls(data, variant, arrays).to(device)
    state = torch.load(OUT/f"hybridconfirm_{family}_seed{seed}.pt",
                       map_location=device, weights_only=True)
    model.load_state_dict(state["state_dict"])
    return model


def paired_bootstrap(a, b, transform, seed=20261007, repetitions=20000):
    # Seed-average each query first; the independent resampling unit is a query.
    delta = transform(a).mean(0)-transform(b).mean(0)
    rng = np.random.default_rng(seed); samples = np.empty(repetitions)
    for start in range(0, repetitions, 1000):
        k = min(1000, repetitions-start)
        idx = rng.integers(0, len(delta), size=(k, len(delta)))
        samples[start:start+k] = delta[idx].mean(1)
    lo, hi = np.quantile(samples, (0.025, 0.975))
    return {"queries": len(delta), "seeds": int(a.shape[0]),
            "difference": float(delta.mean()), "bootstrap_95ci": [float(lo), float(hi)],
            "probability_difference_le_zero": float((samples <= 0).mean()),
            "repetitions": repetitions}


def compare(a, b):
    return {"mrr": paired_bootstrap(a, b, lambda x: 1.0/x),
            "hits1": paired_bootstrap(a, b, lambda x: (x <= 1).astype(float)),
            "hits3": paired_bootstrap(a, b, lambda x: (x <= 3).astype(float)),
            "hits10": paired_bootstrap(a, b, lambda x: (x <= 10).astype(float))}


def main():
    if not torch.cuda.is_available(): raise RuntimeError("CUDA required")
    device = torch.device("cuda"); data = Dataset(SPLIT); arrays = read_compositions(data)
    make_strict_inductive(data)
    ranks = {family: [] for family in FAMILIES}; learned = {family: [] for family in FAMILIES}
    for family in FAMILIES:
        for seed in SEEDS:
            model = load(family, data, arrays, seed, device)
            ranks[family].append(typed_ranks(model, data, data.test, device))
            if family in ("hybrid", "parameter_control"):
                learned[family].append({
                    "unary_gate": float(torch.sigmoid(model.unary_gate)),
                    "pair_gate_at_5": float(torch.sigmoid(model.pair_gate_intercept)),
                    "pair_gate_log_size": float(model.pair_gate_log_size)})
            else:
                learned[family].append({"residual_gate": float(torch.sigmoid(model.interaction_gate).mean())})
    ranks = {k: np.stack(v) for k, v in ranks.items()}
    report = {"overall": {}, "paired_bootstrap": {}, "by_formula_size": {},
              "learned_gates": learned, "complexity": {}}
    for family, values in ranks.items():
        rr = 1.0/values
        report["overall"][family] = {
            "mrr": float(rr.mean()), "hits1": float((values <= 1).mean()),
            "hits3": float((values <= 3).mean()), "hits10": float((values <= 10).mean()),
            "mean_rank": float(values.mean()),
            "per_seed_mrr": [float(x.mean()) for x in rr]}
    for other in ("parameter_control", "deepsets", "pair_only"):
        report["paired_bootstrap"]["hybrid_vs_"+other] = compare(ranks["hybrid"], ranks[other])
    report["paired_bootstrap"]["pair_only_vs_deepsets"] = compare(ranks["pair_only"], ranks["deepsets"])
    sizes = arrays[1][data.test[:, 0]].sum(1)
    for label, subset in (("1_to_5", sizes <= 5),
                          ("6_to_10", (sizes >= 6)&(sizes <= 10)),
                          ("over_10", sizes > 10)):
        report["by_formula_size"][label] = {"n": int(subset.sum())}
        for family, values in ranks.items():
            report["by_formula_size"][label][family+"_mrr"] = float((1.0/values[:, subset]).mean())
        report["by_formula_size"][label]["hybrid_vs_deepsets"] = compare(
            ranks["hybrid"][:, subset], ranks["deepsets"][:, subset])["mrr"]
    selections = json.loads((OUT/"hybrid_confirmatory_dev_selection.json").read_text(encoding="utf-8"))
    for family in FAMILIES:
        rows = [selections[family][str(s)] for s in SEEDS]
        report["complexity"][family] = {"parameters": rows[0]["parameters"],
                                         "mean_training_seconds": float(np.mean([x["seconds"] for x in rows]))}
    report["integrity"] = {"queries": len(data.test), "test_heads": len(set(map(int, data.test[:, 0]))),
                           "all_test_heads_have_composition": bool(arrays[1][data.test[:, 0]].any(1).all()),
                           "same_corpus_confirmatory_resplit": True}
    (OUT/"hybrid_confirmatory_detailed_analysis.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
