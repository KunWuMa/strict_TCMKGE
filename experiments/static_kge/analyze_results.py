# -*- coding: utf-8 -*-
"""Recompute typed test metrics by target relation from frozen checkpoints."""

import json
from collections import Counter

import torch

import train_baselines as k


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output = {}
    for split in ("random", "formula_holdout"):
        data = k.Dataset(split); output[split] = {}
        relation_ids = sorted({int(r) for _, r, _ in data.test})
        output[split]["frequency"] = {}
        for relation_id in relation_ids:
            subset = data.test[data.test[:, 1] == relation_id]
            output[split]["frequency"][data.relations[relation_id]] = k.frequency_baseline(data, subset, True)
        for model_name, cls in k.MODELS.items():
            checkpoint = torch.load(k.OUT / f"{split}_{model_name}.pt", map_location=device, weights_only=True)
            model = cls(data.n_entities, data.n_relations, checkpoint["dim"]).to(device)
            model.load_state_dict(checkpoint["state_dict"]); output[split][model_name] = {}
            for relation_id in relation_ids:
                subset = data.test[data.test[:, 1] == relation_id]
                output[split][model_name][data.relations[relation_id]] = k.evaluate(model, data, subset, device, True)
    (k.OUT / "per_relation.json").write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    for split, models in output.items():
        print(split)
        for model, relations in models.items():
            macro = sum(value["mrr"] for value in relations.values()) / len(relations)
            summary = {name.split("::")[-1]: {"n": value["n"], "mrr": round(value["mrr"], 4)} for name, value in relations.items()}
            print(model, "macro_relation_mrr", round(macro, 6), json.dumps(summary, ensure_ascii=True))


if __name__ == "__main__":
    main()
