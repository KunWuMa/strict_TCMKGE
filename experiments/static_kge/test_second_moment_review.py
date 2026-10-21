# -*- coding: utf-8 -*-
"""Mathematical and capacity checks for the second-moment review control."""

import numpy as np
import torch

from compositional_kge import make_strict_inductive, read_compositions
from hybrid_pair_kge import HybridPairComplEx
from train_baselines import Dataset


def main():
    rng = np.random.default_rng(814)
    for n in (1, 2, 5, 13):
        x = rng.normal(size=(2, n, 7)).astype(np.float32)
        ar, ai = torch.from_numpy(x[0:1]), torch.from_numpy(x[1:2])
        mask = torch.ones((1, n), dtype=torch.bool)
        qr, qi = HybridPairComplEx._second_moment(ar, ai, mask)
        pr, pi, _ = HybridPairComplEx._pair_moment(ar, ai, mask)
        mu = torch.complex(ar.mean(1), ai.mean(1))
        q = torch.complex(qr, qi)
        p = torch.complex(pr, pi)
        assert torch.allclose(q, torch.complex(ar, ai).square().mean(1), atol=1e-6)
        if n == 1:
            assert torch.count_nonzero(p) == 0
        else:
            assert torch.allclose(p, (n * mu.square() - q) / (n - 1), atol=1e-6)

    data = Dataset("hybrid_holdout")
    arrays = read_compositions(data)
    make_strict_inductive(data)
    counts = [sum(p.numel() for p in HybridPairComplEx(data, variant, arrays).parameters())
              for variant in ("hybrid_pair", "dual_mean_control", "second_moment_control")]
    assert len(set(counts)) == 1, counts
    print("second-moment identity and parameter matching verified:", counts[0])


if __name__ == "__main__":
    main()
