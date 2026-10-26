"""Rebuild recipe partitions from source tables and verify exported queries."""
from pathlib import Path
import argparse,hashlib,json,os,sys
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/"outputs/official_kge_dependencies"))
import numpy as np
from threadpoolctl import threadpool_limits
from pair_memory_dev import ingredients
from pair_prototype_full_suite import load_data
def main():
    ap=argparse.ArgumentParser();ap.add_argument("--workbook",type=Path,required=True);ap.add_argument("--output",type=Path,required=True)
    a=ap.parse_args();os.environ["PAIRPROTO_RESULTS_XLSX"]=str(a.workbook);os.environ.pop("PAIRPROTO_INPUT_MODE",None)
    from pairproto_excel_source import ExcelEvidence
    store=ExcelEvidence(a.workbook);records=[]
    protocol=store.json["budget500_full_protocol.json"]
    with threadpool_limits(limits=2):
        for source,spec in protocol["datasets"].items():
            for split in [spec["main"]]+spec["replications"]:
                data=load_data(source,split);expected=store.dataset(source,split)
                np.testing.assert_array_equal(data.test,expected.test)
                held=set(data.dev[:,0])|set(data.test[:,0])
                assert not (set(data.train[:,0])|set(data.train[:,2]))&held
                recipe_sets=[]
                for triples in [data.train,data.dev,data.test]:
                    recipe_sets.append({ingredients(data,int(h)) for h in np.unique(triples[:,0]) if data.arrays[1][int(h)].any()})
                assert not recipe_sets[0]&recipe_sets[1] and not recipe_sets[0]&recipe_sets[2] and not recipe_sets[1]&recipe_sets[2]
                for relation in np.unique(data.test[:,1]):
                    assert len(data.range_candidates[int(relation)])==len(expected.range_candidates[int(relation)])
                    assert set(data.test[data.test[:,1]==relation,2])<=set(data.range_candidates[int(relation)])
                records.append({"source":source,"split":split,"train_triples":len(data.train),"dev_queries":len(data.dev),
                    "test_queries":len(data.test),"test_order_matches_workbook":True,"incident_edge_leakage":False,
                    "exact_recipe_overlap":False,"candidate_extension_from_test":False,
                    "train_sha256":hashlib.sha256(np.asarray(data.train,dtype="<i8").tobytes()).hexdigest(),
                    "dev_sha256":hashlib.sha256(np.asarray(data.dev,dtype="<i8").tobytes()).hexdigest(),
                    "test_sha256":hashlib.sha256(np.asarray(data.test,dtype="<i8").tobytes()).hexdigest()})
                print("RECONSTRUCTION VERIFIED",source,split,flush=True)
    a.output.write_text(json.dumps({"passed":True,"workbook_sha256":store.sha256,"partitions":records,
        "method":"Fresh source-table reconstruction; no result cache used; source datasets are not exported."},ensure_ascii=False,indent=2),encoding="utf-8")
if __name__=="__main__":main()
