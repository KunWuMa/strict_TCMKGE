"""Check that the dimension-general implementation is exactly the original model at d=128."""

import json

import torch

from compositional_kge import OUT, make_strict_inductive, read_compositions
from embedding_dimension_sensitivity import DimHybridComplEx
from hybrid_pair_kge import HybridPairComplEx
from train_baselines import Dataset


def main():
    data = Dataset("hybrid_holdout")
    arrays = read_compositions(data)
    make_strict_inductive(data)
    original = HybridPairComplEx(data, "hybrid_pair", arrays)
    generalized = DimHybridComplEx(data, arrays, 128)
    state = torch.load(OUT / "hybridconfirm_hybrid_seed20260915.pt", map_location="cpu", weights_only=True)["state_dict"]
    original.load_state_dict(state)
    generalized.load_state_dict(state)
    triples = torch.as_tensor(data.test[:64])
    with torch.inference_mode():
        a = original.score(triples[:, 0], triples[:, 1], triples[:, 2])
        b = generalized.score(triples[:, 0], triples[:, 1], triples[:, 2])
    report = {"queries": len(triples), "maximum_absolute_score_difference": float((a - b).abs().max()), "exact": bool(torch.equal(a, b))}
    (OUT / "dimension_implementation_equivalence.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    assert torch.allclose(a, b, atol=0.0, rtol=0.0)


if __name__ == "__main__":
    main()
