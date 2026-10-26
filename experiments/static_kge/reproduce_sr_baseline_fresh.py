"""Fresh, portable training of a development-locked follow-up baseline.

No author checkpoints, test choices, or recorded training reuse are required.
Outputs go to a separate directory and never replace the published run.
"""
import argparse,copy,hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--source",choices=["PRIMARY","KTKP"],required=True)
    ap.add_argument("--seed",type=int,required=True)
    a=ap.parse_args()
    base=ROOT/"outputs/scientific_reports_20261026"
    path=base/"protocol.json";p=json.loads(path.read_text(encoding="utf-8"))
    assert a.seed in p["seeds"]
    for name,digest in p["code_sha256"].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
    import budget500_training as bt
    from threadpoolctl import threadpool_limits
    cfg=p["selected"][a.source]
    bt.P=copy.deepcopy(p["training_protocol"])
    bt.P["neural_configs"][a.source][cfg["model"]]={"lr":cfg["lr"],"heads":cfg["heads"]}
    identity={"parent_protocol_sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
              "mode":"fresh-portable-reproduction","source":a.source,"seed":a.seed,
              "training":bt.P}
    encoded=json.dumps(identity,sort_keys=True,ensure_ascii=False).encode("utf-8")
    bt.SHA=hashlib.sha256(encoded).hexdigest()
    bt.RESULTS=base/"fresh_reproduction"/bt.SHA[:12]
    bt.RESULTS.mkdir(parents=True,exist_ok=True)
    (bt.RESULTS/"protocol.json").write_bytes(encoded)
    with threadpool_limits(limits=2):
        bt.run_simple(a.source,"neural",cfg["model"],a.seed)
    print("Fresh reproduction complete:",bt.RESULTS)
if __name__=="__main__":main()
