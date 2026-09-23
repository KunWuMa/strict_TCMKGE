# -*- coding: utf-8 -*-
"""Build the confirmatory resplit for the frozen Hybrid PairComp-KGE."""

from __future__ import annotations

import json
from collections import defaultdict

import numpy as np

from build_benchmark import OUT, collect, describe

SEED = 20261007
SPLIT = "hybrid_holdout"


def main():
    target, context, groups = collect()
    target_by_head = defaultdict(list)
    for edge in target:
        target_by_head[edge[0]].append(edge)
    composition_heads = {h for h, r, _ in context if r == "cpm_contains_chp"}
    eligible = sorted(set(target_by_head) & composition_heads)
    rng = np.random.default_rng(SEED)
    shuffled = np.asarray(eligible, dtype=object)[rng.permutation(len(eligible))].tolist()
    n_test = int(round(len(shuffled)*0.10)); n_dev = int(round(len(shuffled)*0.10))
    test_heads = set(shuffled[:n_test]); dev_heads = set(shuffled[n_test:n_test+n_dev])
    train_heads = set(target_by_head)-test_heads-dev_heads
    train = [e for h in train_heads for e in target_by_head[h]]
    dev = [e for h in dev_heads for e in target_by_head[h]]
    test = [e for h in test_heads for e in target_by_head[h]]
    train_relations = {r for _, r, _ in train}; train_tails = {t for _, _, t in train}
    dev = [e for e in dev if e[1] in train_relations and e[2] in train_tails]
    test = [e for e in test if e[1] in train_relations and e[2] in train_tails]
    kept_dev = {h for h, _, _ in dev}; kept_test = {h for h, _, _ in test}
    description = describe(SPLIT, train, dev, test, context, sorted(kept_dev), sorted(kept_test))
    manifest = {"seed": SEED, "frozen_model": "hybrid_pair", "role": "confirmatory resplit",
                "same_corpus_as_development": True, "eligible_formula_heads": len(eligible),
                "pre_filter_dev_heads": len(dev_heads), "pre_filter_test_heads": len(test_heads),
                "post_filter_dev_heads": len(kept_dev), "post_filter_test_heads": len(kept_test),
                "relation_labels": groups, "split": description,
                "definition": "formula-disjoint composition-only resplit created after hybrid architecture selection"}
    (OUT/"hybrid_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
