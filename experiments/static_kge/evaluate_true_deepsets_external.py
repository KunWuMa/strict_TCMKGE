# -*- coding: utf-8 -*-
"""Frozen external stress test for the corrected Deep Sets architecture."""

from __future__ import annotations

import json

import numpy as np
import torch

from analyze_hybrid_confirmatory import load
from compositional_kge import OUT, make_strict_inductive, read_compositions
from external_zero_shot_evaluation import (cluster_bootstrap, load_external,
                                            rank_query, seed_summary)
from pair_interaction_kge import PairCompositionalComplEx
from train_baselines import Dataset


SEEDS = (20260915, 20260916, 20260917)


def main():
    torch.set_num_threads(2)
    device = torch.device("cpu")
    data = Dataset("hybrid_holdout")
    arrays = read_compositions(data)
    make_strict_inductive(data)
    _, records, queries, _ = load_external(data)
    truth = {}
    for index, relation, target, _ in queries:
        truth.setdefault((index, relation), set()).add(target)
    ranks = {}
    for family in ("true_deepsets", "paircomp"):
        per_seed = []
        for seed in SEEDS:
            if family == "true_deepsets":
                model = PairCompositionalComplEx(data, "deepsets_elementwise", arrays).to(device)
                state = torch.load(OUT / f"review_true_deepsets_seed{seed}.pt",
                                   map_location=device, weights_only=True)
                model.load_state_dict(state["state_dict"])
            else:
                model = load("hybrid", data, arrays, seed, device)
            model.eval()
            values = []
            for index, relation, target, _ in queries:
                values.append(rank_query(model, records[index]["mapped_herbs"],
                                         relation, target, data.range_candidates[relation],
                                         truth[(index, relation)]))
            per_seed.append(values)
        ranks[family] = np.asarray(per_seed, dtype=float)
    patients = np.asarray([records[index]["patient_id"] for index, _, _, _ in queries])
    report = {
        "role": "exploratory frozen-model domain-shift stress test",
        "queries": len(queries), "patients": len(set(patients)),
        "model_state": "no external fitting or tuning",
        "evaluation_device": "cpu",
        "paircomp": seed_summary(ranks["paircomp"]),
        "true_deepsets": seed_summary(ranks["true_deepsets"]),
        "paircomp_minus_true_deepsets": cluster_bootstrap(
            ranks["paircomp"], ranks["true_deepsets"], patients, seed=20261022),
    }
    path = OUT / "external_true_deepsets_review.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
