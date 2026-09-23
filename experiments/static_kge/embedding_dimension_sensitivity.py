# -*- coding: utf-8 -*-
"""Three-seed embedding-dimension sensitivity for the confirmatory hybrid model."""

from __future__ import annotations

import json
import random
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from compositional_kge import BATCH, EPOCHS, EVAL_EVERY, NEGATIVES, OUT, PATIENCE_EVALS, make_strict_inductive, read_compositions
from train_baselines import Dataset, evaluate, sample_negative_tails

SEEDS = (20260915, 20260916, 20260917)
DIMS_TO_TRAIN = (64, 256)


class DimHybridComplEx(nn.Module):
    def __init__(self, data, arrays, dim):
        super().__init__()
        self.dim = dim
        self.er = nn.Embedding(data.n_entities, dim)
        self.ei = nn.Embedding(data.n_entities, dim)
        self.rr = nn.Embedding(data.n_relations, dim)
        self.ri = nn.Embedding(data.n_relations, dim)
        self.unary_r = nn.Linear(dim, dim, bias=False)
        self.unary_i = nn.Linear(dim, dim, bias=False)
        self.branch_r = nn.Linear(dim, dim, bias=False)
        self.branch_i = nn.Linear(dim, dim, bias=False)
        for parameter in self.parameters():
            if parameter.ndim > 1:
                nn.init.xavier_uniform_(parameter)
        self.unary_gate = nn.Parameter(torch.tensor(-2.0))
        self.pair_gate_intercept = nn.Parameter(torch.tensor(-1.0))
        self.pair_gate_log_size = nn.Parameter(torch.tensor(-0.5))
        ingredients, mask, uniform, *_ = arrays
        self.register_buffer("ingredients", torch.from_numpy(ingredients), persistent=False)
        self.register_buffer("ingredient_mask", torch.from_numpy(mask), persistent=False)
        self.register_buffer("uniform_weight", torch.from_numpy(uniform), persistent=False)
        self.register_buffer("is_formula", torch.from_numpy(mask.any(1)), persistent=False)

    @staticmethod
    def _transform(xr, xi, wr, wi):
        return torch.tanh(wr(xr) - wi(xi)), torch.tanh(wr(xi) + wi(xr))

    @staticmethod
    def _pair_moment(ar, ai, valid):
        m = valid[..., None].to(ar.dtype)
        sr, si = (ar * m).sum(1), (ai * m).sum(1)
        rr, ii, ri = (ar.square() * m).sum(1), (ai.square() * m).sum(1), (ar * ai * m).sum(1)
        n = valid.sum(1).to(ar.dtype)
        denominator = (n * (n - 1)).clamp_min(1.0)[:, None]
        pr = (sr.square() - rr - si.square() + ii) / denominator
        pi = 2.0 * (sr * si - ri) / denominator
        keep = (n > 1)[:, None]
        return pr * keep, pi * keep, n

    def head_embedding(self, h, r):
        hr, hi = self.er(h), self.ei(h)
        rows = self.is_formula[h]
        if not bool(rows.any()):
            return hr, hi
        fh = h[rows]
        herbs, valid = self.ingredients[fh], self.ingredient_mask[fh]
        weights = self.uniform_weight[fh]
        ar, ai = self.er(herbs), self.ei(herbs)
        mu_r, mu_i = (weights[..., None] * ar).sum(1), (weights[..., None] * ai).sum(1)
        unary_r, unary_i = self._transform(mu_r, mu_i, self.unary_r, self.unary_i)
        pair_r, pair_i, n = self._pair_moment(ar, ai, valid)
        branch_r, branch_i = self._transform(pair_r, pair_i, self.branch_r, self.branch_i)
        pair_gate = torch.sigmoid(
            self.pair_gate_intercept + self.pair_gate_log_size * (torch.log(n.clamp_min(1.0)) - np.log(5.0))
        )[:, None]
        out_r = mu_r + torch.sigmoid(self.unary_gate) * unary_r + pair_gate * branch_r
        out_i = mu_i + torch.sigmoid(self.unary_gate) * unary_i + pair_gate * branch_i
        hr, hi = hr.clone(), hi.clone()
        hr[rows], hi[rows] = out_r, out_i
        return hr, hi

    def score(self, h, r, t):
        hr, hi = self.head_embedding(h, r)
        rr, ri, tr, ti = self.rr(r), self.ri(r), self.er(t), self.ei(t)
        return (hr * rr * tr + hi * rr * ti + hr * ri * ti - hi * ri * tr).sum(-1)

    def score_tails(self, h, r, tails):
        hr, hi = self.head_embedding(h.reshape(1), r.reshape(1))
        rr, ri, tr, ti = self.rr(r), self.ri(r), self.er(tails), self.ei(tails)
        return (hr * rr * tr + hi * rr * ti + hr * ri * ti - hi * ri * tr).sum(-1)


def train(data, arrays, dim, seed, device):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    model = DimHybridComplEx(data, arrays, dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003, weight_decay=1e-7)
    rng = np.random.default_rng(seed)
    best, stale, history, start = None, 0, [], time.time()
    checkpoint = OUT / f"sensitivity_dim{dim}_seed{seed}.pt"
    for epoch in range(1, EPOCHS + 1):
        model.train()
        losses = []
        order = rng.permutation(len(data.train))
        for offset in range(0, len(order), BATCH):
            raw = data.train[order[offset:offset + BATCH]]
            neg_np = sample_negative_tails(data, raw, rng)
            batch, neg = torch.as_tensor(raw, device=device), torch.as_tensor(neg_np, device=device)
            h, r, t = batch[:, 0], batch[:, 1], batch[:, 2]
            pos = model.score(h, r, t)
            ns = model.score(
                h[:, None].expand_as(neg).reshape(-1),
                r[:, None].expand_as(neg).reshape(-1),
                neg.reshape(-1),
            ).reshape(len(batch), NEGATIVES)
            loss = F.softplus(-pos).mean() + F.softplus(ns).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        if epoch == 1 or epoch % EVAL_EVERY == 0:
            dev = evaluate(model, data, data.dev, device, True)
            record = {"epoch": epoch, "loss": float(np.mean(losses)), "dev_typed": dev}
            history.append(record)
            print("dimension", dim, "seed", seed, record, flush=True)
            if best is None or dev["mrr"] > best["dev_typed"]["mrr"]:
                best, stale = record, 0
                torch.save({"state_dict": model.state_dict(), "dim": dim, "seed": seed}, checkpoint)
            else:
                stale += 1
                if stale >= PATIENCE_EVALS:
                    break
    return {
        "selected": best,
        "history": history,
        "checkpoint": checkpoint.name,
        "parameters": sum(p.numel() for p in model.parameters()),
        "seconds": time.time() - start,
    }


def summarize(seed_results):
    return {
        metric: {
            "mean": float(np.mean([value["test_typed"][metric] for value in seed_results.values()])),
            "sample_std": float(np.std([value["test_typed"][metric] for value in seed_results.values()], ddof=1)),
        }
        for metric in ("mrr", "hits1", "hits3", "hits10")
    }


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    device = torch.device("cuda")
    report = {"protocol": "development-selected epoch; test opened after all 64/256 runs", "dimensions": {}}
    selections = {}
    for dim in DIMS_TO_TRAIN:
        selections[str(dim)] = {}
        for seed in SEEDS:
            data = Dataset("hybrid_holdout")
            arrays = read_compositions(data)
            make_strict_inductive(data)
            selections[str(dim)][str(seed)] = train(data, arrays, dim, seed, device)
            (OUT / "dimension_sensitivity_dev.json").write_text(json.dumps(selections, indent=2), encoding="utf-8")
    for dim in DIMS_TO_TRAIN:
        seed_results = {}
        for seed in SEEDS:
            data = Dataset("hybrid_holdout")
            arrays = read_compositions(data)
            make_strict_inductive(data)
            model = DimHybridComplEx(data, arrays, dim).to(device)
            chosen = selections[str(dim)][str(seed)]
            model.load_state_dict(torch.load(OUT / chosen["checkpoint"], map_location=device, weights_only=True)["state_dict"])
            seed_results[str(seed)] = {
                "test_typed": evaluate(model, data, data.test, device, True),
                "selected": chosen["selected"],
                "parameters": chosen["parameters"],
                "seconds": chosen["seconds"],
            }
        report["dimensions"][str(dim)] = {"seeds": seed_results, "summary": summarize(seed_results)}
        print("DIMENSION TEST", dim, report["dimensions"][str(dim)]["summary"], flush=True)
    existing = json.loads((OUT / "hybrid_confirmatory_results.json").read_text(encoding="utf-8"))["hybrid_summary"]
    report["dimensions"]["128"] = {
        "source": "pre-registered confirmatory run",
        "summary": {
            metric: existing[metric]
            for metric in ("mrr", "hits1", "hits3", "hits10")
        },
    }
    (OUT / "dimension_sensitivity_results.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report["dimensions"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
