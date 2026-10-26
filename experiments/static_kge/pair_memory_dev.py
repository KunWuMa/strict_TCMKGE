# -*- coding: utf-8 -*-
"""Train-only pair identity signal audit with dev-only score calibration.

This is a diagnostic statistical comparator, not a claim of clinical synergy.
Recipe duplicates are collapsed before counting pair/label occurrences.
"""
from __future__ import annotations

import argparse
import itertools
import json
from collections import Counter, defaultdict

import numpy as np
from scipy import sparse
import torch

from compositional_kge import OUT, CompositionalComplEx, make_strict_inductive, read_compositions
from inpetm_independent import InPETMDataset
from train_baselines import Dataset


def ingredients(data, head):
    return tuple(sorted(data.arrays[0][int(head)][data.arrays[1][int(head)]].tolist()))


class PairMemory:
    def __init__(self, data, relation, shrinkage=20, min_support=3):
        self.shrinkage = shrinkage
        self.candidates = data.range_candidates[relation]
        target_ids = {int(t): i for i, t in enumerate(self.candidates)}
        recipes = defaultdict(set)
        for h, r, t in data.train:
            if int(r) == relation and data.arrays[1][int(h)].any():
                recipes[ingredients(data, h)].add(target_ids[int(t)])
        self.n_recipes = len(recipes)
        herb_support, pair_support = Counter(), Counter()
        for herbs in recipes:
            herb_support.update(herbs)
            pair_support.update(itertools.combinations(herbs, 2))
        self.herb_vocab = {h: i for i, h in enumerate(sorted(herb_support))}
        self.pair_vocab = {p: i for i, p in enumerate(sorted(p for p, n in pair_support.items() if n >= min_support))}
        self.herb_n = np.asarray([herb_support[h] for h in self.herb_vocab], dtype=np.float32)
        self.pair_n = np.asarray([pair_support[p] for p in self.pair_vocab], dtype=np.float32)
        hr, hc, pr, pc, yr, yc = [], [], [], [], [], []
        for i, (herbs, labels) in enumerate(recipes.items()):
            for herb in herbs:
                hr.append(i); hc.append(self.herb_vocab[herb])
            for pair in itertools.combinations(herbs, 2):
                if pair in self.pair_vocab:
                    pr.append(i); pc.append(self.pair_vocab[pair])
            for label in labels:
                yr.append(i); yc.append(label)
        shape_y = (len(recipes), len(self.candidates))
        y = sparse.csr_matrix((np.ones(len(yr), dtype=np.int32), (yr, yc)), shape=shape_y)
        hx = sparse.csr_matrix((np.ones(len(hr), dtype=np.int32), (hr, hc)), shape=(len(recipes), len(self.herb_vocab)))
        px = sparse.csr_matrix((np.ones(len(pr), dtype=np.int32), (pr, pc)), shape=(len(recipes), len(self.pair_vocab)))
        self.herb_counts = (hx.T @ y).tocsr()
        self.pair_counts = (px.T @ y).tocsr()
        self.target_recipe_counts = np.asarray(y.sum(0)).ravel().astype(np.int64)
        self.prior = self.target_recipe_counts.astype(np.float32) / max(1, len(recipes))
        self.prior = np.clip(self.prior, 1e-6, 1-1e-6)

    def lift(self, counts, ids, totals):
        if not ids:
            return np.empty((0, len(self.candidates)), dtype=np.float32)
        n = totals[ids, None]
        probability = (counts[ids].toarray().astype(np.float32) + self.shrinkage * self.prior) / (n + self.shrinkage)
        probability = np.clip(probability, 1e-7, 1-1e-7)
        prior_logit = np.log(self.prior) - np.log1p(-self.prior)
        return (np.log(probability) - np.log1p(-probability) - prior_logit) * (n / (n + self.shrinkage))

    def scores(self, herbs, top_k=3):
        hids = [self.herb_vocab[h] for h in herbs if h in self.herb_vocab]
        hlift = self.lift(self.herb_counts, hids, self.herb_n)
        herb_map = {h: v for h, v in zip((h for h in herbs if h in self.herb_vocab), hlift)}
        pairs = [p for p in itertools.combinations(herbs, 2) if p in self.pair_vocab]
        plift = self.lift(self.pair_counts, [self.pair_vocab[p] for p in pairs], self.pair_n)
        zero = np.zeros(len(self.candidates), dtype=np.float32)
        result = {"herb_mean": hlift.mean(0) if len(hlift) else zero,
                  "pair_mean": plift.mean(0) if len(plift) else zero}
        def top(values):
            if not len(values): return zero
            k = min(top_k, len(values))
            return np.partition(values, -k, axis=0)[-k:].mean(0)
        result["herb_top"] = top(hlift)
        result["pair_top"] = top(plift)
        residual = np.asarray([value-herb_map[left]-herb_map[right]
                               for (left, right), value in zip(pairs, plift)])
        result["pair_residual_top"] = top(np.maximum(residual, 0))
        return result, len(pairs)


def standardized(x):
    return (x-x.mean()) / max(float(x.std()), 1e-8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["KTKP", "DRAGON", "PRIMARY"], required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    if args.source == "PRIMARY":
        data = Dataset("hybrid_holdout")
        data.arrays = read_compositions(data)
        make_strict_inductive(data)
        path = OUT / "expanded_mean_complex_seed20260915.pt"
    else:
        data = InPETMDataset(args.source, 20261021, 400)
        path = OUT / f"inpetm_{args.source.lower()}_g400_mean_k3_b0.25_seed20260915.pt"
    mean = CompositionalComplEx(data, "uniform_mean", data.arrays).cuda()
    mean.load_state_dict(torch.load(path, map_location="cuda", weights_only=False)["state_dict"])
    mean.eval()
    memories = {int(r): PairMemory(data, int(r)) for r in np.unique(data.dev[:, 1])}
    methods = ("herb_mean", "herb_top", "pair_mean", "pair_top", "pair_residual_top")
    weights = (0.0, 0.1, 0.25, 0.5, 0.75, 1.0)
    reciprocal = {(method, weight): [] for method in methods for weight in weights}
    cache = {}
    coverage = []
    with torch.inference_mode():
        for h, r, t in data.dev:
            key = int(h), int(r)
            candidates = data.range_candidates[int(r)]
            if key not in cache:
                base = mean.score_tails(torch.tensor(int(h), device="cuda"), torch.tensor(int(r), device="cuda"), torch.as_tensor(candidates, device="cuda")).cpu().numpy()
                scores, covered_pairs = memories[int(r)].scores(ingredients(data, h))
                cache[key] = standardized(base), {m: standardized(s) for m, s in scores.items()}
                coverage.append(covered_pairs)
            base, scores = cache[key]
            target_index = np.searchsorted(candidates, int(t))
            known = data.all_true[key] - {int(t)}
            allowed = ~np.isin(candidates, np.fromiter(known, dtype=np.int64))
            for method in methods:
                for weight in weights:
                    blend = (1-weight)*base + weight*scores[method]
                    target = blend[target_index]
                    rank = 1 + np.sum((blend > target) & allowed) + 0.5 * max(0, int(np.sum((blend == target) & allowed))-1)
                    reciprocal[method, weight].append(1/rank)
    report = {"source": args.source, "role": "development-only train-derived pair signal audit",
              "shrinkage": 20, "minimum_pair_support": 3, "top_k": 3,
              "n_dev": len(data.dev), "median_covered_pairs": float(np.median(coverage)),
              "memory": {str(r): {"unique_training_recipes": m.n_recipes, "herbs": len(m.herb_vocab), "pairs": len(m.pair_vocab)} for r, m in memories.items()},
              "mrr_by_blend": {method: {str(w): float(np.mean(reciprocal[method, w])) for w in weights} for method in methods}}
    (OUT / f"pair_memory_{args.source.lower()}_dev.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
