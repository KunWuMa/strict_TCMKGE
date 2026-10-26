# -*- coding: utf-8 -*-
"""Two-stage query-conditioned sparse pair scorer for exploratory experiments.

The old PairComp encoder provides the base score. A relation-conditioned
shortlist of distinct herb pairs is then rescored against each candidate tail.
This is a candidate-aware KGE scoring function, not a tail-independent entity
embedding. It is kept separate from the frozen manuscript implementation.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from compositional_kge import DIM
from hybrid_pair_kge import HybridPairComplEx


class CandidatePairComplEx(HybridPairComplEx):
    def __init__(self, data, arrays, shortlist: int = 12, top_k: int = 3,
                 gate_init: float = -3.0):
        super().__init__(data, "hybrid_pair", arrays)
        if shortlist < 1 or top_k < 1 or top_k > shortlist:
            raise ValueError("Require 1 <= top_k <= shortlist")
        self.shortlist = shortlist
        self.top_k = top_k
        self.candidate_gate = nn.Parameter(torch.full((data.n_relations,),
                                                       gate_init))
        row, col = torch.triu_indices(self.ingredients.shape[1],
                                      self.ingredients.shape[1], offset=1)
        self.register_buffer("pair_row", row, persistent=False)
        self.register_buffer("pair_col", col, persistent=False)

    def shortlist_pairs(self, h, r):
        batch = h.shape[0]
        length = self.ingredients.shape[1]
        herbs = self.ingredients[h]
        valid = self.ingredient_mask[h]
        ar, ai = self.er(herbs), self.ei(herbs)
        rr, ri = self.rr(r), self.ri(r)
        projected = ((ar * rr[:, None, :] + ai * ri[:, None, :]).sum(-1)
                     / math.sqrt(DIM))
        projected = projected.masked_fill(~valid, 0.0)
        row, col = self.pair_row, self.pair_col
        scores = projected[:, row] * projected[:, col]
        pair_valid = valid[:, row] & valid[:, col]
        scores = scores.masked_fill(~pair_valid, -1e9)
        keep = min(self.shortlist, scores.shape[1])
        _, chosen = scores.topk(keep, dim=1)
        chosen_valid = pair_valid.gather(1, chosen)
        left, right = row[chosen], col[chosen]
        left_r = ar.gather(1, left[..., None].expand(-1, -1, DIM))
        left_i = ai.gather(1, left[..., None].expand(-1, -1, DIM))
        right_r = ar.gather(1, right[..., None].expand(-1, -1, DIM))
        right_i = ai.gather(1, right[..., None].expand(-1, -1, DIM))
        pair_r = left_r * right_r - left_i * right_i
        pair_i = left_r * right_i + left_i * right_r
        return pair_r, pair_i, chosen_valid

    def score_bundle(self, h, r, tails):
        if tails.ndim != 2:
            raise ValueError("tails must be [batch, candidates]")
        hr, hi = self.head_embedding(h, r)
        rr, ri = self.rr(r), self.ri(r)
        tr, ti = self.er(tails), self.ei(tails)
        base = ((hr * rr)[:, None, :] * tr +
                (hi * rr)[:, None, :] * ti +
                (hr * ri)[:, None, :] * ti -
                (hi * ri)[:, None, :] * tr).sum(-1)
        formula = self.is_formula[h]
        if not bool(formula.any()):
            return base
        fr, fi, valid = self.shortlist_pairs(h[formula], r[formula])
        cr, ci = tr[formula], ti[formula]
        qr, qi = rr[formula], ri[formula]
        # Pair-specific ComplEx score for every candidate in the bundle.
        qreal = qr[:, None, :] * cr + qi[:, None, :] * ci
        qimag = qr[:, None, :] * ci - qi[:, None, :] * cr
        evidence = (torch.einsum("bpd,bcd->bcp", fr, qreal) +
                    torch.einsum("bpd,bcd->bcp", fi, qimag))
        masked = evidence.masked_fill(~valid[:, None, :], -1e9)
        values = masked.topk(min(self.top_k, masked.shape[-1]), dim=-1).values
        chosen = values > -1e8
        top = (values.masked_fill(~chosen, 0).sum(-1) /
               chosen.sum(-1).clamp_min(1))
        average = (evidence.masked_fill(~valid[:, None, :], 0).sum(-1) /
                   valid.sum(-1)[:, None].clamp_min(1))
        residual = top - average
        gate = torch.sigmoid(self.candidate_gate[r[formula]])[:, None]
        output = base.clone()
        output[formula] = base[formula] + gate * residual
        return output

    def score(self, h, r, t):
        return self.score_bundle(h, r, t[:, None])[:, 0]

    def score_tails(self, h, r, tails):
        return self.score_bundle(h.reshape(1), r.reshape(1),
                                 tails.reshape(1, -1))[0]
