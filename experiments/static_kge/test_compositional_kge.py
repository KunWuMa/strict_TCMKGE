# -*- coding: utf-8 -*-
"""Small invariance and split tests for the composition model."""

import numpy as np
import torch

from compositional_kge import CompositionalComplEx, make_strict_inductive, read_compositions
from train_baselines import Dataset


def main():
    data = Dataset("formula_holdout")
    arrays = read_compositions(data)
    held = set(map(int, data.dev[:, 0])) | set(map(int, data.test[:, 0]))
    stats = make_strict_inductive(data)
    assert not any(int(h) in held or int(t) in held for h, _, t in data.train)
    assert stats["held_formula_count"] == len(held)

    model = CompositionalComplEx(data, "uniform_mean", arrays)
    h = int(data.dev[0, 0]); r = int(data.dev[0, 1]); t = int(data.dev[0, 2])
    before = model.score(torch.tensor([h]), torch.tensor([r]), torch.tensor([t])).detach()
    with torch.no_grad():
        model.er.weight[h].fill_(999.0); model.ei.weight[h].fill_(-999.0)
    after = model.score(torch.tensor([h]), torch.tensor([r]), torch.tensor([t])).detach()
    assert torch.equal(before, after), "formula ID embedding leaked into formula score"

    ingredients, mask, uniform, dosage, complete = arrays
    assert np.allclose(uniform.sum(1)[mask.any(1)], 1.0)
    assert np.allclose(dosage.sum(1)[mask.any(1)], 1.0)
    assert int(complete.sum()) == 3636
    final_data = Dataset("composition_holdout")
    final_arrays = read_compositions(final_data)
    final_held = set(map(int, final_data.dev[:, 0])) | set(map(int, final_data.test[:, 0]))
    final_stats = make_strict_inductive(final_data)
    assert final_arrays[1][final_data.dev[:, 0]].any(1).all()
    assert final_arrays[1][final_data.test[:, 0]].any(1).all()
    assert not any(int(h) in final_held or int(t) in final_held for h, _, t in final_data.train)
    print("composition tests passed", stats, final_stats)


if __name__ == "__main__":
    main()
