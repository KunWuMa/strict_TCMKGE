# -*- coding: utf-8 -*-
"""Build two additional formula-disjoint splits for reviewer robustness tests."""
from __future__ import annotations
import csv, json
from collections import defaultdict
import numpy as np
from build_benchmark import OUT, collect, describe

SPLITS = {"review_split_2": 20261017, "review_split_3": 20261027}

def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        csv.writer(handle, delimiter="\t", lineterminator="\n").writerows(rows)

def main():
    target, context, groups = collect()
    by_head = defaultdict(list)
    for edge in target:
        by_head[edge[0]].append(edge)
    composition_heads = {h for h, r, _ in context if r == "cpm_contains_chp"}
    eligible = sorted(set(by_head) & composition_heads)
    manifests = {}
    for split, seed in SPLITS.items():
        rng = np.random.default_rng(seed)
        heads = np.asarray(eligible, dtype=object)[rng.permutation(len(eligible))].tolist()
        n_test = int(round(0.10 * len(heads)))
        n_dev = int(round(0.10 * len(heads)))
        test_heads = set(heads[:n_test])
        dev_heads = set(heads[n_test:n_test+n_dev])
        train_heads = set(by_head) - test_heads - dev_heads
        target_train = [e for h in train_heads for e in by_head[h]]
        dev = [e for h in dev_heads for e in by_head[h]]
        test = [e for h in test_heads for e in by_head[h]]
        train_relations = {r for _, r, _ in target_train}
        train_tails = {t for _, _, t in target_train}
        dev = [e for e in dev if e[1] in train_relations and e[2] in train_tails]
        test = [e for e in test if e[1] in train_relations and e[2] in train_tails]
        kept_dev = sorted({h for h, _, _ in dev})
        kept_test = sorted({h for h, _, _ in test})
        directory = OUT / split
        write_rows(directory / "train.tsv", target_train + context)
        write_rows(directory / "dev.tsv", dev)
        write_rows(directory / "test.tsv", test)
        manifests[split] = {
            "seed": seed, "eligible_formula_heads": len(eligible),
            "post_filter_dev_heads": len(kept_dev),
            "post_filter_test_heads": len(kept_test),
            "relation_labels": groups,
            "split": describe(split, target_train, dev, test, context, kept_dev, kept_test),
            "definition": "independent formula-disjoint split; identical construction and filtering rules"
        }
    (OUT / "reviewer_splits_manifest.json").write_text(
        json.dumps(manifests, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifests, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
