import torch

from compositional_kge import make_strict_inductive, read_compositions
from expanded_kge_baselines import COMPOSITIONAL_MODELS, CompositionalKGE
from train_baselines import Dataset


def main():
    data = Dataset("hybrid_holdout")
    arrays = read_compositions(data)
    integrity = make_strict_inductive(data)
    held = torch.tensor([int(data.test[0, 0]), int(data.test[1, 0])])
    relation = torch.tensor([int(data.test[0, 1]), int(data.test[1, 1])])
    tail = torch.tensor([int(data.test[0, 2]), int(data.test[1, 2])])
    assert integrity["held_formula_count"] > 0
    for name in COMPOSITIONAL_MODELS:
        model = CompositionalKGE(data, arrays, name)
        scores = model.score(held, relation, tail)
        assert scores.shape == (2,)
        assert torch.isfinite(scores).all()
        candidates = torch.as_tensor(data.range_candidates[int(relation[0])][:7])
        tail_scores = model.score_tails(held[0], relation[0], candidates)
        assert tail_scores.shape == (7,)
        assert torch.isfinite(tail_scores).all()
    print("expanded KGE baseline tests passed")


if __name__ == "__main__":
    main()
