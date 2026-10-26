# -*- coding: utf-8 -*-
"""Relation-conditioned sparse pair composition with no additional parameters.

This is an experimental successor to the frozen PairComp-KGE model.  The
selection is hard top-k; softmax weights within the selected pairs retain
gradients for the selected scores.  Test-set model selection is prohibited.
"""

from __future__ import annotations

import math

import torch

from compositional_kge import DIM
from hybrid_pair_kge import HybridPairComplEx


class RelationSparsePairComplEx(HybridPairComplEx):
    def __init__(self, data, arrays, top_k: int = 3, blend: float = 0.25,
                 selection: str = "relation"):
        super().__init__(data, "hybrid_pair", arrays)
        if top_k < 1 or not 0.0 <= blend <= 1.0:
            raise ValueError("top_k must be positive and blend must be in [0, 1]")
        if selection not in {"relation", "unconditioned"}:
            raise ValueError(selection)
        self.top_k = top_k
        self.blend = blend
        self.selection = selection
        row, col = torch.triu_indices(self.ingredients.shape[1],
                                      self.ingredients.shape[1], offset=1)
        self.register_buffer("pair_row", row, persistent=False)
        self.register_buffer("pair_col", col, persistent=False)

    def sparse_pair_moment(self, ar, ai, valid, relation):
        """Select top-k distinct pairs by relation-projected herb compatibility.

        The rank-one selector costs O(n d + n^2); only selected complex pair
        products are materialized, costing O(k d).  The uniform PairComp moment
        is retained separately as a stable residual.
        """
        batch, length, dim = ar.shape
        if length < 2:
            return torch.zeros((batch, dim), device=ar.device, dtype=ar.dtype), \
                   torch.zeros((batch, dim), device=ar.device, dtype=ar.dtype)
        if self.selection == "relation":
            query_r, query_i = self.rr(relation), self.ri(relation)
            projected = ((ar * query_r[:, None, :] +
                          ai * query_i[:, None, :]).sum(-1) / math.sqrt(DIM))
        else:
            # Parameter-free capacity control: a fixed, relation-independent
            # projection.  It receives the same embeddings and pair branch.
            projected = (ar + ai).sum(-1) / math.sqrt(DIM)
        projected = projected.masked_fill(~valid, 0.0)
        if length == self.ingredients.shape[1]:
            row, col = self.pair_row, self.pair_col
        else:
            row, col = torch.triu_indices(length, length, offset=1,
                                          device=ar.device)
        scores = projected[:, row] * projected[:, col]
        pair_valid = valid[:, row] & valid[:, col]
        scores = scores.masked_fill(~pair_valid, -1e9)
        k = min(self.top_k, scores.shape[1])
        chosen_scores, chosen = scores.topk(k, dim=1)
        chosen_valid = pair_valid.gather(1, chosen)
        # Per-formula scale normalization prevents the softmax from becoming
        # nearly uniform solely because embedding magnitudes are small.
        finite_scores = scores.masked_fill(~pair_valid, 0.0)
        count = pair_valid.sum(1, keepdim=True).clamp_min(1)
        mean = finite_scores.sum(1, keepdim=True) / count
        variance = (((finite_scores - mean).square() * pair_valid).sum(1,
                    keepdim=True) / count).clamp_min(1e-6)
        logits = (chosen_scores - mean) / variance.sqrt().detach()
        logits = logits.masked_fill(~chosen_valid, -1e9)
        weights = torch.softmax(logits, dim=1) * chosen_valid
        weights = weights / weights.sum(1, keepdim=True).clamp_min(1e-8)
        left, right = row[chosen], col[chosen]
        left_r = ar.gather(1, left[..., None].expand(-1, -1, dim))
        left_i = ai.gather(1, left[..., None].expand(-1, -1, dim))
        right_r = ar.gather(1, right[..., None].expand(-1, -1, dim))
        right_i = ai.gather(1, right[..., None].expand(-1, -1, dim))
        pair_r = left_r * right_r - left_i * right_i
        pair_i = left_r * right_i + left_i * right_r
        return (weights[..., None] * pair_r).sum(1), \
               (weights[..., None] * pair_i).sum(1)

    def head_embedding(self, h, r):
        hr, hi = self.er(h), self.ei(h)
        rows = self.is_formula[h]
        if not bool(rows.any()):
            return hr, hi
        fh, fr = h[rows], r[rows]
        herbs, valid = self.ingredients[fh], self.ingredient_mask[fh]
        ar, ai = self.er(herbs), self.ei(herbs)
        weights = self.uniform_weight[fh]
        mu_r = (weights[..., None] * ar).sum(1)
        mu_i = (weights[..., None] * ai).sum(1)
        unary_r, unary_i = self._transform(mu_r, mu_i,
                                          self.unary_r, self.unary_i)
        uniform_r, uniform_i, n = self._pair_moment(ar, ai, valid)
        sparse_r, sparse_i = self.sparse_pair_moment(ar, ai, valid, fr)
        pair_r = uniform_r + self.blend * (sparse_r - uniform_r)
        pair_i = uniform_i + self.blend * (sparse_i - uniform_i)
        branch_r, branch_i = self._transform(pair_r, pair_i,
                                            self.branch_r, self.branch_i)
        gate = torch.sigmoid(self.pair_gate_intercept +
                             self.pair_gate_log_size *
                             (torch.log(n.clamp_min(1.0)) - math.log(5.0)))[:, None]
        out_r = mu_r + torch.sigmoid(self.unary_gate) * unary_r + gate * branch_r
        out_i = mu_i + torch.sigmoid(self.unary_gate) * unary_i + gate * branch_i
        hr, hi = hr.clone(), hi.clone()
        hr[rows], hi[rows] = out_r, out_i
        return hr, hi

    def score_bundle(self, h, r, tails):
        """Score positive and sampled tails with one head encoding per query."""
        hr, hi = self.head_embedding(h, r)
        rr, ri = self.rr(r), self.ri(r)
        tr, ti = self.er(tails), self.ei(tails)
        return ((hr * rr)[:, None, :] * tr +
                (hi * rr)[:, None, :] * ti +
                (hr * ri)[:, None, :] * ti -
                (hi * ri)[:, None, :] * tr).sum(-1)
