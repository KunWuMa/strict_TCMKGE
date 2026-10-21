# -*- coding: utf-8 -*-
"""Latency audit for newly added second-moment and true Deep Sets encoders."""

from __future__ import annotations

import json

import numpy as np
import torch

from benchmark_efficiency import timed_encoder
from compositional_kge import OUT, make_strict_inductive, read_compositions
from hybrid_pair_kge import HybridPairComplEx
from pair_interaction_kge import PairCompositionalComplEx
from train_baselines import Dataset


SEEDS = (20260915, 20260916, 20260917)


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    device = torch.device("cuda")
    data = Dataset("hybrid_holdout")
    arrays = read_compositions(data)
    make_strict_inductive(data)
    first_relation = {}
    for h, r, _ in data.test:
        first_relation.setdefault(int(h), int(r))
    unique_heads = np.asarray(sorted(first_relation), dtype=np.int64)
    heads = torch.as_tensor(unique_heads, device=device)
    relations = torch.as_tensor([first_relation[int(h)] for h in unique_heads], device=device)
    report = {"hardware": torch.cuda.get_device_name(device), "torch": torch.__version__,
              "batch_formulas": len(unique_heads), "models": {}}
    for name in ("second_moment", "true_deepsets"):
        rows = []
        for seed in SEEDS:
            if name == "second_moment":
                model = HybridPairComplEx(data, "second_moment_control", arrays).to(device)
                checkpoint = OUT / f"review_second_moment_seed{seed}.pt"
            else:
                model = PairCompositionalComplEx(data, "deepsets_elementwise", arrays).to(device)
                checkpoint = OUT / f"review_true_deepsets_seed{seed}.pt"
            state = torch.load(checkpoint, map_location=device, weights_only=True)
            model.load_state_dict(state["state_dict"])
            rows.append({"seed": seed, "parameters": sum(p.numel() for p in model.parameters()),
                         **timed_encoder(model, heads, relations)})
            del model
        report["models"][name] = {
            "seeds": rows, "parameters": rows[0]["parameters"],
            "batch_milliseconds_mean": float(np.mean([x["batch_milliseconds_mean"] for x in rows])),
            "batch_milliseconds_std": float(np.std([x["batch_milliseconds_mean"] for x in rows], ddof=1))}
    path = OUT / "review_encoder_efficiency.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
