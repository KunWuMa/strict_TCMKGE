# -*- coding: utf-8 -*-
"""Exploratory frozen-model external comparison with the second-moment control."""

from __future__ import annotations

import json

import numpy as np
import torch

from analyze_hybrid_confirmatory import load
from compositional_kge import OUT, make_strict_inductive, read_compositions
from external_zero_shot_evaluation import (cluster_bootstrap, load_external,
                                            rank_query, seed_summary)
from hybrid_pair_kge import HybridPairComplEx
from train_baselines import Dataset


SEEDS = (20260915, 20260916, 20260917)


def main():
    # CPU inference avoids competing with the independent GPU training run.
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
    for family in ("second_moment", "paircomp"):
        per_seed = []
        for seed in SEEDS:
            if family == "second_moment":
                model = HybridPairComplEx(data, "second_moment_control", arrays).to(device)
                state = torch.load(OUT / f"review_second_moment_seed{seed}.pt",
                                   map_location=device, weights_only=True)
                model.load_state_dict(state["state_dict"])
            else:
                model = load("hybrid", data, arrays, seed, device)
            model.eval()
            values = [rank_query(model, records[index]["mapped_herbs"], relation,
                                 target, data.range_candidates[relation],
                                 truth[(index, relation)])
                      for index, relation, target, _ in queries]
            per_seed.append(values)
        ranks[family] = np.asarray(per_seed, dtype=float)

    patients = np.asarray([records[index]["patient_id"]
                           for index, _, _, _ in queries])
    report = {
        "role": "exploratory frozen-model domain-shift stress test",
        "queries": len(queries), "patients": len(set(patients)),
        "model_state": "no external fitting or tuning",
        "evaluation_device": "cpu",
        "paircomp": seed_summary(ranks["paircomp"]),
        "second_moment": seed_summary(ranks["second_moment"]),
        "paircomp_minus_second_moment": cluster_bootstrap(
            ranks["paircomp"], ranks["second_moment"], patients,
            seed=20261023),
    }
    path = OUT / "external_second_moment_review.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
