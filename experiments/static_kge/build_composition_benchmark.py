# -*- coding: utf-8 -*-
"""Build the final formula-disjoint benchmark restricted to formulas with compositions."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from build_benchmark import OUT, collect, describe

SEED = 20261001


def main():
    target, context, groups = collect()
    target_by_head = defaultdict(list)
    for edge in target:
        target_by_head[edge[0]].append(edge)
    composition_heads = {h for h, r, _ in context if r == "cpm_contains_chp"}
    eligible = sorted(set(target_by_head) & composition_heads)
    rng = np.random.default_rng(SEED)
    shuffled = np.asarray(eligible, dtype=object)[rng.permutation(len(eligible))].tolist()
    n_test = int(round(len(shuffled) * 0.10)); n_dev = int(round(len(shuffled) * 0.10))
    test_heads = set(shuffled[:n_test]); dev_heads = set(shuffled[n_test:n_test+n_dev])
    train_heads = set(target_by_head) - test_heads - dev_heads
    train = [edge for head in train_heads for edge in target_by_head[head]]
    dev = [edge for head in dev_heads for edge in target_by_head[head]]
    test = [edge for head in test_heads for edge in target_by_head[head]]
    train_relations = {r for _, r, _ in train}; train_tails = {t for _, _, t in train}
    dev = [e for e in dev if e[1] in train_relations and e[2] in train_tails]
    test = [e for e in test if e[1] in train_relations and e[2] in train_tails]
    kept_dev_heads = {h for h, _, _ in dev}; kept_test_heads = {h for h, _, _ in test}
    assert kept_dev_heads <= composition_heads and kept_test_heads <= composition_heads
    description = describe("composition_holdout", train, dev, test, context,
                           sorted(kept_dev_heads), sorted(kept_test_heads))
    manifest = {"seed": SEED, "eligible_formula_heads": len(eligible),
                "pre_filter_dev_heads": len(dev_heads), "pre_filter_test_heads": len(test_heads),
                "post_filter_dev_heads": len(kept_dev_heads), "post_filter_test_heads": len(kept_test_heads),
                "relation_labels": groups, "split": description,
                "definition": "formula-disjoint target split restricted to CPMs with D4 composition; D4 is side information only"}
    (OUT / "composition_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
