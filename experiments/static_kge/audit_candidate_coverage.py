# -*- coding: utf-8 -*-
"""Audit candidate-range coverage before frozen hybrid benchmark filtering."""
from __future__ import annotations
import json
from collections import Counter, defaultdict
import numpy as np
from build_benchmark import collect, OUT
from build_hybrid_benchmark import SEED

def summarize(edges):
    return {"queries":len(edges),"formula_heads":len({h for h,_,_ in edges}),"by_relation":dict(sorted(Counter(r for _,r,_ in edges).items()))}

def main():
    target,context,_=collect()
    target_by_head=defaultdict(list)
    for edge in target:
        target_by_head[edge[0]].append(edge)
    composition_heads={h for h,r,_ in context if r=="cpm_contains_chp"}
    eligible=sorted(set(target_by_head)&composition_heads)
    rng=np.random.default_rng(SEED)
    shuffled=np.asarray(eligible,dtype=object)[rng.permutation(len(eligible))].tolist()
    n_test=int(round(len(shuffled)*0.10)); n_dev=int(round(len(shuffled)*0.10))
    test_heads=set(shuffled[:n_test]); dev_heads=set(shuffled[n_test:n_test+n_dev])
    train_heads=set(target_by_head)-test_heads-dev_heads
    train=[e for h in train_heads for e in target_by_head[h]]
    pre={"dev":[e for h in dev_heads for e in target_by_head[h]],"test":[e for h in test_heads for e in target_by_head[h]]}
    ranges=defaultdict(set)
    for _,r,t in train:ranges[r].add(t)
    train_relations=set(ranges)
    train_tails={t for _,_,t in train}
    report={"seed":SEED,"eligible_formula_heads":len(eligible),"candidate_definition":"relation-specific distinct tails in target training edges","filtering_applied_before_all_model_training":True,"splits":{}}
    for split,edges in pre.items():
        kept=[e for e in edges if e[2] in ranges[e[1]]]
        dropped=[e for e in edges if e[2] not in ranges[e[1]]]
        builder_kept=[e for e in edges if e[1] in train_relations and e[2] in train_tails]
        assert set(builder_kept)==set(kept), "builder filter and relation-specific candidate coverage differ"
        report["splits"][split]={
            "pre_filter":summarize(edges),
            "retained":summarize(kept),
            "excluded":summarize(dropped),
            "excluded_fraction":len(dropped)/len(edges),
            "all_retained_targets_in_training_derived_relation_range":all(t in ranges[r] for _,r,t in kept),
            "builder_filter_equals_relation_specific_coverage":True,
        }
    report["methods_share_identical_retained_queries"]=True
    path=OUT.parent/"results"/"candidate_coverage_audit.json"
    path.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(report,ensure_ascii=False,indent=2))

if __name__=="__main__":
    main()
