# -*- coding: utf-8 -*-
"""PairComp-KGE: efficient complex-valued pair interaction set encoder."""

from __future__ import annotations

import json
import math
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

VARIANTS = ("deepsets_residual", "pair_residual", "relation_pair_residual")


class PairCompositionalComplEx(CompositionalComplEx):
    def __init__(self, data, variant, arrays):
        # Reuse embeddings and composition buffers; base encoder branch is overridden.
        super().__init__(data, "uniform_mean", arrays)
        if variant not in VARIANTS: raise ValueError(variant)
        self.pair_variant = variant
        self.transform_r = nn.Linear(DIM, DIM, bias=False)
        self.transform_i = nn.Linear(DIM, DIM, bias=False)
        nn.init.xavier_uniform_(self.transform_r.weight)
        nn.init.xavier_uniform_(self.transform_i.weight)
        gate_size = data.n_relations if variant == "relation_pair_residual" else 1
        # A small initial correction protects the reliable mean representation.
        self.interaction_gate = nn.Parameter(torch.full((gate_size,), -2.0))

    def _complex_transform(self, xr, xi):
        yr = self.transform_r(xr) - self.transform_i(xi)
        yi = self.transform_r(xi) + self.transform_i(xr)
        return torch.tanh(yr), torch.tanh(yi)

    def head_embedding(self, h, r):
        hr, hi = self.er(h), self.ei(h)
        rows = self.is_formula[h]
        if not bool(rows.any()): return hr, hi
        fh, fr = h[rows], r[rows]
        herbs = self.ingredients[fh]; valid = self.ingredient_mask[fh]
        weights = self.uniform_weight[fh]
        ar, ai = self.er(herbs), self.ei(herbs)
        mu_r = (weights[..., None] * ar).sum(1)
        mu_i = (weights[..., None] * ai).sum(1)

        if self.pair_variant == "deepsets_residual":
            corr_r, corr_i = self._complex_transform(mu_r, mu_i)
        else:
            m = valid[..., None].to(ar.dtype)
            sr = (ar*m).sum(1); si = (ai*m).sum(1)
            sum_rr = (ar.square()*m).sum(1); sum_ii = (ai.square()*m).sum(1)
            sum_ri = (ar*ai*m).sum(1)
            n = valid.sum(1).to(ar.dtype); denominator = (n*(n-1)).clamp_min(1.0)[:, None]
            pair_r = (sr.square()-sum_rr-si.square()+sum_ii) / denominator
            pair_i = 2.0*(sr*si-sum_ri) / denominator
            non_single = (n > 1)[:, None]
            pair_r = pair_r * non_single; pair_i = pair_i * non_single
            corr_r, corr_i = self._complex_transform(pair_r, pair_i)

        if self.pair_variant == "relation_pair_residual":
            gate = torch.sigmoid(self.interaction_gate[fr])[:, None]
        else:
            gate = torch.sigmoid(self.interaction_gate)[None, :]
        out_r = mu_r + gate*corr_r; out_i = mu_i + gate*corr_i
        hr = hr.clone(); hi = hi.clone(); hr[rows] = out_r; hi[rows] = out_i
        return hr, hi


def train_pair(data, variant, arrays, device, seed=20260915, tag=None):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    model = PairCompositionalComplEx(data, variant, arrays).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003, weight_decay=1e-7)
    rng = np.random.default_rng(seed); best = None; stale = 0; history = []; start = time.time()
    checkpoint = OUT/(tag or f"pair_{variant}_seed{seed}.pt")
    for epoch in range(1, EPOCHS+1):
        model.train(); order = rng.permutation(len(data.train)); losses = []
        for offset in range(0, len(order), BATCH):
            raw = data.train[order[offset:offset+BATCH]]; neg_np = sample_negative_tails(data, raw, rng)
            batch = torch.as_tensor(raw, device=device); neg = torch.as_tensor(neg_np, device=device)
            h,r,t = batch[:,0],batch[:,1],batch[:,2]
            pos = model.score(h,r,t)
            nh = h[:,None].expand_as(neg).reshape(-1); nr = r[:,None].expand_as(neg).reshape(-1)
            ns = model.score(nh,nr,neg.reshape(-1)).reshape(len(batch),NEGATIVES)
            loss = F.softplus(-pos).mean()+F.softplus(ns).mean()
            optimizer.zero_grad(); loss.backward(); optimizer.step(); losses.append(float(loss.detach()))
        if epoch == 1 or epoch%EVAL_EVERY == 0:
            dev = evaluate(model,data,data.dev,device,True)
            record={"epoch":epoch,"loss":float(np.mean(losses)),"dev_typed":dev}
            history.append(record); print(variant,record,flush=True)
            if best is None or dev["mrr"]>best["dev_typed"]["mrr"]:
                best=record; stale=0
                torch.save({"state_dict":model.state_dict(),"variant":variant,"seed":seed},checkpoint)
            else:
                stale+=1
                if stale>=PATIENCE_EVALS: break
    return {"selected":best,"history":history,"checkpoint":checkpoint.name,
            "parameters":sum(p.numel() for p in model.parameters()),"seconds":time.time()-start}


def main():
    if not torch.cuda.is_available(): raise RuntimeError("CUDA required")
    device=torch.device("cuda"); data=Dataset("composition_holdout")
    arrays=read_compositions(data); make_strict_inductive(data)
    selections={}
    for variant in VARIANTS:
        selections[variant]=train_pair(data,variant,arrays,device,tag=f"development_{variant}.pt")
        (OUT/"pair_development_selection.json").write_text(json.dumps(selections,indent=2),encoding="utf-8")
    # Deliberately no test evaluation in method-development stage.
    print("PAIR DEVELOPMENT COMPLETE; TEST NOT EVALUATED",flush=True)


if __name__ == "__main__": main()
