# -*- coding: utf-8 -*-
"""Structural tests for the hybrid pair-aware inductive encoder."""

import numpy as np
import torch

from compositional_kge import make_strict_inductive, read_compositions
from hybrid_pair_kge import HybridPairComplEx
from train_baselines import Dataset


def main():
    data = Dataset("pair_holdout")
    arrays = read_compositions(data)
    make_strict_inductive(data)
    model = HybridPairComplEx(data, "hybrid_pair", arrays)
    h, r, t = map(int, data.test[0])
    ht = torch.tensor([h]); rt = torch.tensor([r]); tt = torch.tensor([t])
    before = model.score(ht, rt, tt).detach().clone()
    with torch.no_grad():
        model.er.weight[h].fill_(999.0); model.ei.weight[h].fill_(-999.0)
    after = model.score(ht, rt, tt).detach()
    assert torch.equal(before, after), "formula ID leaked into hybrid inductive score"

    # Set order cannot affect a prescription embedding.
    with torch.no_grad():
        original_ingredients = model.ingredients[h].clone()
        original_mask = model.ingredient_mask[h].clone()
        original_weights = model.uniform_weight[h].clone()
        valid_count = int(original_mask.sum())
        order = torch.arange(valid_count-1, -1, -1)
        model.ingredients[h, :valid_count] = original_ingredients[:valid_count][order]
        model.uniform_weight[h, :valid_count] = original_weights[:valid_count][order]
    permuted = model.score(ht, rt, tt).detach()
    assert torch.allclose(before, permuted, atol=1e-7), "encoder is not permutation invariant"
    with torch.no_grad():
        model.ingredients[h] = original_ingredients
        model.ingredient_mask[h] = original_mask
        model.uniform_weight[h] = original_weights

    ingredients, mask, *_ = arrays
    herbs = torch.as_tensor(ingredients[h][mask[h]], dtype=torch.long)
    assert len(herbs) >= 1
    ar, ai = model.er(herbs[None, :]), model.ei(herbs[None, :])
    valid = torch.ones((1, len(herbs)), dtype=torch.bool)
    pr, pi, n = model._pair_moment(ar, ai, valid)
    if len(herbs) == 1:
        assert torch.count_nonzero(pr) == 0 and torch.count_nonzero(pi) == 0
    assert np.isfinite(pr.detach().numpy()).all() and np.isfinite(pi.detach().numpy()).all()
    print("hybrid pair tests passed")


if __name__ == "__main__":
    main()
