# -*- coding: utf-8 -*-
"""Parameter, checkpoint, and encoder-latency audit for Expert Systems."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from analyze_hybrid_confirmatory import FAMILIES, load
from compositional_kge import OUT, make_strict_inductive, read_compositions
from train_baselines import Dataset

SEEDS = (20260915, 20260916, 20260917)


def timed_encoder(model, heads, relations, repeats=200, warmup=30):
    model.eval()
    with torch.inference_mode():
        for _ in range(warmup):
            model.head_embedding(heads, relations)
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(repeats):
            model.head_embedding(heads, relations)
        torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) * 1000.0 / repeats
    return {
        "batch_milliseconds_mean": elapsed,
        "milliseconds_per_formula": elapsed / len(heads),
        "batch_formulas": int(len(heads)),
        "warmup": warmup,
        "timed_repeats": repeats,
    }


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
    report = {
        "hardware": torch.cuda.get_device_name(device),
        "torch": torch.__version__,
        "encoder_scope": "one generated embedding per unique unseen test formula",
        "models": {},
        "asymptotic_encoder_complexity": {
            "all_implemented_set_encoders": "O(n d)",
            "explicit_pair_enumeration_avoided": "O(n^2 d)",
            "identity": "sum_{i!=j} z_i*z_j = (sum_i z_i)^2 - sum_i z_i^2",
        },
    }
    for family in FAMILIES:
        seed_reports = []
        for seed in SEEDS:
            model = load(family, data, arrays, seed, device)
            checkpoint = {
                "hybrid": f"hybridconfirm_hybrid_seed{seed}.pt",
                "parameter_control": f"hybridconfirm_parameter_control_seed{seed}.pt",
                "deepsets": f"hybridconfirm_deepsets_seed{seed}.pt",
                "pair_only": f"hybridconfirm_pair_only_seed{seed}.pt",
            }[family]
            seed_reports.append({
                "seed": seed,
                "parameters": sum(p.numel() for p in model.parameters()),
                "trainable_megabytes_fp32": sum(p.numel() for p in model.parameters()) * 4 / 1024**2,
                "checkpoint_megabytes": (OUT / checkpoint).stat().st_size / 1024**2,
                **timed_encoder(model, heads, relations),
            })
            del model
        report["models"][family] = {
            "seeds": seed_reports,
            "batch_milliseconds_mean": float(np.mean([x["batch_milliseconds_mean"] for x in seed_reports])),
            "batch_milliseconds_std": float(np.std([x["batch_milliseconds_mean"] for x in seed_reports], ddof=1)),
            "parameters": seed_reports[0]["parameters"],
        }
    hybrid = report["models"]["hybrid"]["parameters"]
    deep = report["models"]["deepsets"]["parameters"]
    report["hybrid_over_deepsets"] = {
        "additional_parameters": hybrid - deep,
        "relative_parameter_increase_percent": 100.0 * (hybrid - deep) / deep,
        "latency_ratio": report["models"]["hybrid"]["batch_milliseconds_mean"] / report["models"]["deepsets"]["batch_milliseconds_mean"],
    }
    (OUT / "efficiency_results.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
