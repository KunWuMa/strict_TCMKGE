# -*- coding: utf-8 -*-
"""Gradient-based local pair sensitivity for selected correct predictions."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch

from analyze_composition_results import typed_ranks
from analyze_hybrid_confirmatory import load
from compositional_kge import MKG, OUT, make_strict_inductive, read_compositions
from train_baselines import Dataset

SEEDS = (20260915, 20260916, 20260917)


def table(path, key, value, extras=()):
    out = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            out[row[key].strip()] = {"name": row[value].strip(), **{column: row.get(column, "").strip() for column in extras}}
    return out


def labels():
    return {
        "cpm": table(MKG / "D2_Chinese_patent_medicine.tsv", "CPM_ID", "Chinese_patent_medicine", ("Pinyin_term",)),
        "chp": table(MKG / "D6_Chinese_herbal_pieces.tsv", "CHP_ID", "Chinese_herbal_pieces", ("English_term",)),
        "tcmt": table(MKG / "D1_TCM_terminology.tsv", "TCMT_ID", "Chinese_term", ("English_term", "Chinese_group", "English_group")),
    }


def describe(entity, maps):
    prefix, identifier = entity.split(":", 1)
    return {"id": identifier, **maps.get(prefix, {}).get(identifier, {"name": entity})}


def one_model_attribution(model, h, r, t):
    device = model.er.weight.device
    h_tensor = torch.tensor([h], device=device)
    r_tensor = torch.tensor([r], device=device)
    t_tensor = torch.tensor([t], device=device)
    herbs = model.ingredients[h_tensor][0]
    valid = model.ingredient_mask[h_tensor][0]
    herbs = herbs[valid]
    ar, ai = model.er(herbs), model.ei(herbs)
    n = len(herbs)
    mu_r, mu_i = ar.mean(0).detach(), ai.mean(0).detach()
    pair_r0, pair_i0, count = model._pair_moment(ar[None], ai[None], torch.ones((1, n), dtype=torch.bool, device=device))
    pair_r = pair_r0[0].detach().requires_grad_(True)
    pair_i = pair_i0[0].detach().requires_grad_(True)
    unary_r, unary_i = model._transform(mu_r, mu_i, model.unary_r, model.unary_i)
    branch_r, branch_i = model._transform(pair_r, pair_i, model.branch_r, model.branch_i)
    gate = torch.sigmoid(
        model.pair_gate_intercept + model.pair_gate_log_size * (torch.log(count[0].clamp_min(1.0)) - np.log(5.0))
    )
    head_r = mu_r + torch.sigmoid(model.unary_gate) * unary_r + gate * branch_r
    head_i = mu_i + torch.sigmoid(model.unary_gate) * unary_i + gate * branch_i
    rr, ri, tr, ti = model.rr(r_tensor)[0], model.ri(r_tensor)[0], model.er(t_tensor)[0], model.ei(t_tensor)[0]
    score = (head_r * rr * tr + head_i * rr * ti + head_r * ri * ti - head_i * ri * tr).sum()
    grad_r, grad_i = torch.autograd.grad(score, (pair_r, pair_i))
    pair_values = []
    denominator = float(n * (n - 1))
    for i in range(n):
        for j in range(i + 1, n):
            contribution_r = 2.0 * (ar[i] * ar[j] - ai[i] * ai[j]) / denominator
            contribution_i = 2.0 * (ar[i] * ai[j] + ai[i] * ar[j]) / denominator
            attribution = (grad_r * contribution_r + grad_i * contribution_i).sum()
            pair_values.append((int(herbs[i]), int(herbs[j]), float(attribution.detach())))
    # A finite intervention answers whether the complete learned pair branch raises this target score.
    no_pair_r = mu_r + torch.sigmoid(model.unary_gate) * unary_r
    no_pair_i = mu_i + torch.sigmoid(model.unary_gate) * unary_i
    no_pair_score = (no_pair_r * rr * tr + no_pair_i * rr * ti + no_pair_r * ri * ti - no_pair_i * ri * tr).sum()
    linearized_pair_total = float((grad_r * pair_r + grad_i * pair_i).sum().detach())
    return pair_values, float(score.detach()), float(no_pair_score.detach()), float(gate.detach()), linearized_pair_total


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    device = torch.device("cuda")
    data = Dataset("hybrid_holdout")
    arrays = read_compositions(data)
    make_strict_inductive(data)
    models = [load("hybrid", data, arrays, seed, device) for seed in SEEDS]
    ranks = np.stack([typed_ranks(model, data, data.test, device) for model in models])
    maps = labels()
    candidates = []
    for edge_index, (h, r, t) in enumerate(data.test):
        h, r, t = int(h), int(r), int(t)
        size = int(arrays[1][h].sum())
        if size < 2 or size > 10 or not np.all(ranks[:, edge_index] <= 1):
            continue
        per_seed = [one_model_attribution(model, h, r, t) for model in models]
        delta = float(np.mean([score - no_pair for _, score, no_pair, _, _ in per_seed]))
        aggregated = {}
        for values, _, _, _, _ in per_seed:
            for herb_a, herb_b, attribution in values:
                aggregated.setdefault((herb_a, herb_b), []).append(attribution)
        pairs = sorted(
            [
                {
                    "herb_a": describe(data.entities[a], maps),
                    "herb_b": describe(data.entities[b], maps),
                    "attribution_mean": float(np.mean(values)),
                    "attribution_std": float(np.std(values, ddof=1)),
                }
                for (a, b), values in aggregated.items()
            ],
            key=lambda item: item["attribution_mean"],
            reverse=True,
        )
        candidates.append({
            "edge_index": edge_index,
            "formula": describe(data.entities[h], maps),
            "relation": data.relations[r],
            "target": describe(data.entities[t], maps),
            "formula_size": size,
            "rank_each_seed": ranks[:, edge_index].tolist(),
            "full_minus_zero_pair_target_score": delta,
            "pair_gate_mean": float(np.mean([values[3] for values in per_seed])),
            "linearized_pair_total_mean": float(np.mean([values[4] for values in per_seed])),
            "attribution_additivity_max_abs_error": float(max(
                abs(sum(item[2] for item in values[0]) - values[4]) for values in per_seed
            )),
            "top_positive_pairs": [item for item in pairs if item["attribution_mean"] > 0][:5],
            "top_negative_pairs": [item for item in reversed(pairs) if item["attribution_mean"] < 0][:5],
            "attribution_definition": "gradient of target score with respect to pair moment dotted with each unordered pair contribution; seed mean",
        })
    # Favor cases where the pair branch supports an exact top-1 prediction, while keeping formulas distinct.
    candidates.sort(key=lambda item: item["full_minus_zero_pair_target_score"], reverse=True)
    selected, seen = [], set()
    for item in candidates:
        formula_id = item["formula"]["id"]
        if formula_id in seen:
            continue
        selected.append(item)
        seen.add(formula_id)
        if len(selected) == 5:
            break
    report = {
        "scope": "gradient-based local pair sensitivities; no claim of clinical pair validity",
        "selection": "five distinct formulas, size 2-10, rank 1 for all three seeds, largest positive pair-branch target-score intervention",
        "eligible_edges": len(candidates),
        "cases": selected,
    }
    (OUT / "pair_attribution_cases.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
