# -*- coding: utf-8 -*-
"""Hybrid PairComp-KGE: unary DeepSets backbone plus size-adaptive pair residual."""

from __future__ import annotations

import json
import random
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from compositional_kge import (BATCH, DIM, EPOCHS, EVAL_EVERY, NEGATIVES, OUT,
                               PATIENCE_EVALS, CompositionalComplEx,
                               make_strict_inductive, read_compositions)
from train_baselines import Dataset, evaluate, sample_negative_tails

VARIANTS = ("hybrid_pair", "dual_mean_control", "fixed_pair_gate", "ungated_pair")


class HybridPairComplEx(CompositionalComplEx):
    """Generate an unseen formula from first- and second-order herb set moments."""

    def __init__(self, data, variant, arrays):
        super().__init__(data, "uniform_mean", arrays)
        if variant not in VARIANTS:
            raise ValueError(variant)
        self.hybrid_variant = variant
        self.unary_r = nn.Linear(DIM, DIM, bias=False)
        self.unary_i = nn.Linear(DIM, DIM, bias=False)
        self.branch_r = nn.Linear(DIM, DIM, bias=False)
        self.branch_i = nn.Linear(DIM, DIM, bias=False)
        for layer in (self.unary_r, self.unary_i, self.branch_r, self.branch_i):
            nn.init.xavier_uniform_(layer.weight)
        self.unary_gate = nn.Parameter(torch.tensor(-2.0))
        # centered at five herbs; a learnable negative slope can suppress noisy
        # high-cardinality pair moments without discarding compact formula pairs.
        self.pair_gate_intercept = nn.Parameter(torch.tensor(-1.0))
        self.pair_gate_log_size = nn.Parameter(torch.tensor(-0.5))

    @staticmethod
    def _transform(xr, xi, wr, wi):
        return torch.tanh(wr(xr) - wi(xi)), torch.tanh(wr(xi) + wi(xr))

    @staticmethod
    def _pair_moment(ar, ai, valid):
        m = valid[..., None].to(ar.dtype)
        sr, si = (ar*m).sum(1), (ai*m).sum(1)
        rr, ii, ri = (ar.square()*m).sum(1), (ai.square()*m).sum(1), (ar*ai*m).sum(1)
        n = valid.sum(1).to(ar.dtype)
        denominator = (n*(n-1)).clamp_min(1.0)[:, None]
        pr = (sr.square()-rr-si.square()+ii)/denominator
        pi = 2.0*(sr*si-ri)/denominator
        keep = (n > 1)[:, None]
        return pr*keep, pi*keep, n

    def head_embedding(self, h, r):
        hr, hi = self.er(h), self.ei(h)
        rows = self.is_formula[h]
        if not bool(rows.any()):
            return hr, hi
        fh = h[rows]
        herbs, valid = self.ingredients[fh], self.ingredient_mask[fh]
        weights = self.uniform_weight[fh]
        ar, ai = self.er(herbs), self.ei(herbs)
        mu_r = (weights[..., None]*ar).sum(1)
        mu_i = (weights[..., None]*ai).sum(1)
        unary_r, unary_i = self._transform(mu_r, mu_i, self.unary_r, self.unary_i)
        pair_r, pair_i, n = self._pair_moment(ar, ai, valid)
        if self.hybrid_variant == "dual_mean_control":
            pair_r, pair_i = mu_r, mu_i
        branch_r, branch_i = self._transform(pair_r, pair_i, self.branch_r, self.branch_i)
        unary_gate = torch.sigmoid(self.unary_gate)
        centered_log_size = torch.log(n.clamp_min(1.0)) - np.log(5.0)
        if self.hybrid_variant == "fixed_pair_gate":
            pair_gate = torch.sigmoid(self.pair_gate_intercept).expand_as(n)[:, None]
        elif self.hybrid_variant == "ungated_pair":
            pair_gate = torch.ones_like(n)[:, None]
        else:
            pair_gate = torch.sigmoid(self.pair_gate_intercept + self.pair_gate_log_size*centered_log_size)[:, None]
        out_r = mu_r + unary_gate*unary_r + pair_gate*branch_r
        out_i = mu_i + unary_gate*unary_i + pair_gate*branch_i
        hr, hi = hr.clone(), hi.clone()
        hr[rows], hi[rows] = out_r, out_i
        return hr, hi


def train_hybrid(data, variant, arrays, device, seed=20260915, tag=None):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    model = HybridPairComplEx(data, variant, arrays).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003, weight_decay=1e-7)
    rng = np.random.default_rng(seed)
    best, stale, history, start = None, 0, [], time.time()
    checkpoint = OUT/(tag or f"hybrid_{variant}_seed{seed}.pt")
    for epoch in range(1, EPOCHS+1):
        model.train(); losses=[]
        order = rng.permutation(len(data.train))
        for offset in range(0, len(order), BATCH):
            raw = data.train[order[offset:offset+BATCH]]
            neg_np = sample_negative_tails(data, raw, rng)
            batch = torch.as_tensor(raw, device=device); neg = torch.as_tensor(neg_np, device=device)
            h, r, t = batch[:, 0], batch[:, 1], batch[:, 2]
            pos = model.score(h, r, t)
            nh = h[:, None].expand_as(neg).reshape(-1); nr = r[:, None].expand_as(neg).reshape(-1)
            ns = model.score(nh, nr, neg.reshape(-1)).reshape(len(batch), NEGATIVES)
            loss = F.softplus(-pos).mean()+F.softplus(ns).mean()
            optimizer.zero_grad(); loss.backward(); optimizer.step(); losses.append(float(loss.detach()))
        if epoch == 1 or epoch % EVAL_EVERY == 0:
            dev = evaluate(model, data, data.dev, device, True)
            record = {"epoch": epoch, "loss": float(np.mean(losses)), "dev_typed": dev,
                      "unary_gate": float(torch.sigmoid(model.unary_gate).detach()),
                      "pair_gate_at_5": float(torch.sigmoid(model.pair_gate_intercept).detach()),
                      "pair_gate_log_size": float(model.pair_gate_log_size.detach())}
            history.append(record); print(variant, record, flush=True)
            if best is None or dev["mrr"] > best["dev_typed"]["mrr"]:
                best, stale = record, 0
                torch.save({"state_dict": model.state_dict(), "variant": variant, "seed": seed}, checkpoint)
            else:
                stale += 1
                if stale >= PATIENCE_EVALS:
                    break
    return {"selected": best, "history": history, "checkpoint": checkpoint.name,
            "parameters": sum(p.numel() for p in model.parameters()), "seconds": time.time()-start}


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    device = torch.device("cuda")
    results = {}
    for split in ("composition_holdout", "pair_holdout"):
        results[split] = {}
        for variant in VARIANTS:
            data = Dataset(split); arrays = read_compositions(data); make_strict_inductive(data)
            results[split][variant] = train_hybrid(
                data, variant, arrays, device, seed=20260915,
                tag=f"hybrid_development_{split}_{variant}.pt")
            (OUT/"hybrid_development_selection.json").write_text(
                json.dumps(results, indent=2), encoding="utf-8")
    print("HYBRID DEVELOPMENT COMPLETE; NO NEW TEST EVALUATED", flush=True)


if __name__ == "__main__":
    main()
