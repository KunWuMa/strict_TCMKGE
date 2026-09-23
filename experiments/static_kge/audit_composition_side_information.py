# -*- coding: utf-8 -*-
"""Audit the declared composition side channel and strict split integrity."""
from __future__ import annotations
import hashlib, json
import numpy as np
from compositional_kge import MKG, OUT, make_strict_inductive, read_compositions
from train_baselines import Dataset

def main():
    data=Dataset("hybrid_holdout")
    arrays=read_compositions(data)
    ingredients,mask=arrays[:2]
    held=set(map(int,data.dev[:,0]))|set(map(int,data.test[:,0]))
    test_heads=set(map(int,data.test[:,0]))
    dev_heads=set(map(int,data.dev[:,0]))
    relation_counts={}
    for rid,name in enumerate(data.relations):
        rows=data.test[data.test[:,1]==rid]
        if len(rows):
            relation_counts[name]={"queries":int(len(rows)),"typed_candidates_before_filtering":int(len(data.range_candidates[rid]))}
    integrity=make_strict_inductive(data)
    strict_entities=set(map(int,np.concatenate([data.train[:,0],data.train[:,2]])))
    train_formula_heads={int(h) for h,_,_ in data.train if mask[int(h)].any()}
    train_composition_herbs={int(x) for h in train_formula_heads for x in ingredients[h][mask[h]]}
    strict_graph_herbs={e for e in strict_entities if data.entities[e].startswith("chp:")}
    learned_herbs=train_composition_herbs|strict_graph_herbs
    test_herbs={int(x) for h in test_heads for x in ingredients[h][mask[h]]}
    dev_herbs={int(x) for h in dev_heads for x in ingredients[h][mask[h]]}
    unseen_test=test_herbs-learned_herbs
    affected=[h for h in test_heads if any(int(x) in unseen_test for x in ingredients[h][mask[h]])]
    source=MKG/"D4_CPM_CHP.tsv"
    report={
      "composition_provenance":{
        "source":str(source),"source_role":"declared inference-time side information from the same TCM-MKG release",
        "sha256":hashlib.sha256(source.read_bytes()).hexdigest(),
        "contains_relation_removed_from_learned_graph":all(data.relations[int(r)]!="cpm_contains_chp" for _,r,_ in data.train)
      },
      "strict_integrity":{**integrity,
        "held_formula_count":len(held),
        "held_formula_incident_edges_remaining":sum(int(h) in held or int(t) in held for h,_,t in data.train),
        "held_formula_overlap_train_test":len(test_heads&train_formula_heads),
        "dev_test_formula_overlap":len(dev_heads&test_heads)},
      "constituent_coverage":{
        "unique_test_herbs":len(test_herbs),"unique_dev_herbs":len(dev_herbs),
        "test_herbs_seen_via_training_formula_compositions":len(test_herbs&train_composition_herbs),
        "test_herbs_seen_in_strict_graph":len(test_herbs&strict_graph_herbs),
        "test_herbs_seen_by_either_training_channel":len(test_herbs&learned_herbs),
        "unseen_test_herbs":len(unseen_test),
        "test_formulas_with_any_unseen_herb":len(affected),
        "setting":"formula-inductive; constituent-herb transductive when unseen_test_herbs is zero"},
      "test_relations":relation_counts
    }
    (OUT/"composition_side_information_audit.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(report,ensure_ascii=False,indent=2))

if __name__=="__main__": main()
