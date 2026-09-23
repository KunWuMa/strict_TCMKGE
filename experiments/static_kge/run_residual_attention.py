# -*- coding: utf-8 -*-
"""Run the two dev-motivated residual attention candidates, then open test once."""

import json
import torch

from compositional_kge import (HERE, OUT, SEED, CompositionalComplEx,
                               make_strict_inductive, read_compositions, train_one)
from train_baselines import Dataset, evaluate

VARIANTS = ("residual_attention", "dose_residual_attention")


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    data = Dataset("formula_holdout")
    arrays = read_compositions(data)
    strict = make_strict_inductive(data)
    device = torch.device("cuda")
    selections = {"config": {"seed": SEED, "strict": strict}, "models": {}}
    for variant in VARIANTS:
        selections["models"][variant] = train_one(data, variant, arrays, device)
        (OUT / "residual_attention_dev_selection.json").write_text(json.dumps(selections, indent=2), encoding="utf-8")
    results = {"config": selections["config"], "models": {}}
    for variant in VARIANTS:
        chosen = selections["models"][variant]
        model = CompositionalComplEx(data, variant, arrays).to(device)
        state = torch.load(OUT / chosen["checkpoint"], map_location=device, weights_only=True)
        model.load_state_dict(state["state_dict"])
        gates = torch.sigmoid(model.attention_gate).detach().cpu().tolist()
        result = {"typed": evaluate(model, data, data.test, device, True),
                  "untyped": evaluate(model, data, data.test, device, False),
                  "selected": chosen["selected"], "parameters": chosen["parameters"],
                  "relation_gates": dict(zip(data.relations, gates))}
        results["models"][variant] = result
        print("TEST", variant, result, flush=True)
    (OUT / "residual_attention_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("RESIDUAL ATTENTION SCREEN COMPLETE", flush=True)


if __name__ == "__main__":
    main()
