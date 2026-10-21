# -*- coding: utf-8 -*-
"""Dev-selected second-moment control and five-seed direct comparison.

The architecture and seeds are fixed here. All training/checkpoint selection
finishes before any test query is evaluated. Existing three-seed checkpoints are
reused; only absent runs are trained. This script never chooses a model by test.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from analyze_composition_results import typed_ranks
from compositional_kge import make_strict_inductive, read_compositions
from expanded_kge_baselines import CompositionalKGE, train_one as train_mean
from hybrid_pair_kge import HybridPairComplEx, train_hybrid
from train_baselines import Dataset, evaluate


HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
SPLIT = "hybrid_holdout"
SEEDS = (20260915, 20260916, 20260917, 20260918, 20260919)
EXISTING_SEEDS = SEEDS[:3]
FAMILIES = ("paircomp", "mean_complex", "matched", "second_moment")
DEV_FILE = OUT / "second_moment_review_dev.json"
RESULT_FILE = OUT / "second_moment_review_results.json"
RANK_FILE = OUT / "second_moment_review_ranks.npz"


def fresh():
    data = Dataset(SPLIT)
    arrays = read_compositions(data)
    make_strict_inductive(data)
    return data, arrays


def checkpoint(family, seed):
    if family == "second_moment":
        return f"review_second_moment_seed{seed}.pt"
    if seed in EXISTING_SEEDS:
        return ({"paircomp": "hybridconfirm_hybrid",
                 "mean_complex": "expanded_mean_complex",
                 "matched": "hybridconfirm_parameter_control"}[family]
                + f"_seed{seed}.pt")
    return ({"paircomp": "review5_paircomp",
             "mean_complex": "expanded_mean_complex",
             "matched": "review5_matched"}[family]
            + f"_seed{seed}.pt")


def model_for(family, data, arrays):
    if family == "mean_complex":
        return CompositionalKGE(data, arrays, "mean_complex")
    variant = {"paircomp": "hybrid_pair", "matched": "dual_mean_control",
               "second_moment": "second_moment_control"}[family]
    return HybridPairComplEx(data, variant, arrays)


def train_missing(family, seed, device):
    path = OUT / checkpoint(family, seed)
    if path.exists():
        return None
    data, arrays = fresh()
    if family == "mean_complex":
        return train_mean(data, arrays, "mean_complex", seed, device)
    variant = {"paircomp": "hybrid_pair", "matched": "dual_mean_control",
               "second_moment": "second_moment_control"}[family]
    return train_hybrid(data, variant, arrays, device, seed, path.name)


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the frozen experiment budget")
    device = torch.device("cuda")
    OUT.mkdir(parents=True, exist_ok=True)
    dev = json.loads(DEV_FILE.read_text(encoding="utf-8")) if DEV_FILE.exists() else {
        "split": SPLIT, "seeds": SEEDS, "selection": "typed filtered development MRR",
        "architecture": "same maps, gates, decoder and parameter count as PairComp-KGE",
        "second_moment": "mean of elementwise complex herb squares",
        "runs": {family: {} for family in FAMILIES},
    }
    # Train all missing runs before touching the test set.
    for family in FAMILIES:
        for seed in SEEDS:
            path = OUT / checkpoint(family, seed)
            if path.exists():
                print("REUSE", family, seed, path.name, flush=True)
                continue
            chosen = train_missing(family, seed, device)
            dev["runs"][family][str(seed)] = chosen
            DEV_FILE.write_text(json.dumps(dev, indent=2), encoding="utf-8")
            print("DEV SELECTED", family, seed, chosen["selected"]["epoch"], flush=True)

    data, arrays = fresh()
    ranks = {}
    results = {"split": SPLIT, "seeds": SEEDS, "models": {},
               "test_scored_after_all_selection": True,
               "candidate_scope": "training-derived relation ranges, filtered",
               "queries": len(data.test), "formula_heads": len(set(map(int, data.test[:, 0])))}
    for family in FAMILIES:
        results["models"][family] = {}
        rows = []
        for seed in SEEDS:
            model = model_for(family, data, arrays).to(device)
            state = torch.load(OUT / checkpoint(family, seed), map_location=device,
                               weights_only=True)
            model.load_state_dict(state["state_dict"])
            model.eval()
            row = evaluate(model, data, data.test, device, typed=True)
            results["models"][family][str(seed)] = row
            rows.append(typed_ranks(model, data, data.test, device))
            print("TEST", family, seed, row, flush=True)
        ranks[family] = np.stack(rows)
        results["models"][family]["summary"] = {
            metric: {"mean": float(np.mean([results["models"][family][str(s)][metric]
                                       for s in SEEDS])),
                     "sample_std": float(np.std([results["models"][family][str(s)][metric]
                                                 for s in SEEDS], ddof=1))}
            for metric in ("mrr", "hits1", "hits3", "hits10")}
    np.savez_compressed(RANK_FILE, heads=data.test[:, 0], **ranks)
    RESULT_FILE.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("COMPLETE", RESULT_FILE, RANK_FILE, flush=True)


if __name__ == "__main__":
    main()
