# -*- coding: utf-8 -*-
import unittest

import numpy as np
import torch

import train_baselines as k


class TestKGE(unittest.TestCase):
    def test_transe_exact_translation_scores_higher(self):
        model = k.TransE(3, 1, 2)
        with torch.no_grad():
            model.entity.weight[:] = torch.tensor([[1., 0.], [0., 1.], [-1., 0.]])
            model.relation.weight[:] = torch.tensor([[-1., 1.]])
        positive = model.score(torch.tensor([0]), torch.tensor([0]), torch.tensor([1]))
        negative = model.score(torch.tensor([0]), torch.tensor([0]), torch.tensor([2]))
        self.assertGreater(float(positive), float(negative))

    def test_complex_can_be_asymmetric(self):
        model = k.ComplEx(2, 1, 1)
        with torch.no_grad():
            model.er.weight[:] = torch.tensor([[1.], [0.]])
            model.ei.weight[:] = torch.tensor([[0.], [1.]])
            model.rr.weight[:] = torch.tensor([[0.]])
            model.ri.weight[:] = torch.tensor([[1.]])
        forward = model.score(torch.tensor([0]), torch.tensor([0]), torch.tensor([1]))
        reverse = model.score(torch.tensor([1]), torch.tensor([0]), torch.tensor([0]))
        self.assertNotEqual(float(forward), float(reverse))

    def test_frozen_manifest_has_disjoint_target_splits(self):
        for split in ("random", "formula_holdout"):
            data = k.Dataset(split)
            train = {tuple(x) for x in data.train if data.target_relation(int(x[1]))}
            dev = {tuple(x) for x in data.dev}; test = {tuple(x) for x in data.test}
            self.assertFalse(train & dev); self.assertFalse(train & test); self.assertFalse(dev & test)
            for _, relation, tail in np.vstack([data.dev, data.test]):
                self.assertIn(int(tail), set(data.range_candidates[int(relation)].tolist()))


if __name__ == "__main__":
    unittest.main()
