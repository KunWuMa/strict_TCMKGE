"""Development-locked baseline follow-up for Scientific Reports."""
from pathlib import Path
import argparse,copy,hashlib,json,sys,shutil
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/"outputs/official_kge_dependencies"))
import torch
from threadpoolctl import threadpool_limits
import budget500_training as bt
BASE=ROOT/"outputs/scientific_reports_20261026"
P=json.loads((BASE/"protocol.json").read_text(encoding="utf-8"))
SHA=hashlib.sha256((BASE/"protocol.json").read_bytes()).hexdigest()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--source",required=True);ap.add_argument("--seed",type=int,required=True)
    a=ap.parse_args();cfg=P["selected"][a.source];name=cfg["model"]
    assert a.seed in P["seeds"]
    for filename,digest in P["code_sha256"].items():
        assert hashlib.sha256((ROOT/filename).read_bytes()).hexdigest()==digest,filename
    bt.P=copy.deepcopy(P["training_protocol"])
    bt.P["neural_configs"][a.source][name]={"lr":cfg["lr"],"heads":cfg["heads"]}
    bt.SHA=SHA;bt.RESULTS=BASE/"results"
    target=bt.stem(a.source,"neural",name,a.seed)
    with threadpool_limits(limits=2):
        if a.seed==P["reuse_seed"] and not target.with_suffix(".train.json").exists():
            original=json.loads(Path(cfg["reuse_training_record"]).read_text(encoding="utf-8"))
            assert original["protocol_sha256"]==P["parent_protocol_sha256"]
            assert original["config"]["lr"]==cfg["lr"]
            assert original["config"].get("heads",4)==cfg["heads"]
            assert original["seed"]==a.seed and original["split"]==P["training_protocol"]["datasets"][a.source]
            alias=copy.deepcopy(original)
            ckpt=torch.load(original["checkpoint"],map_location="cpu",weights_only=False)
            assert ckpt["protocol_sha256"]==P["parent_protocol_sha256"]
            ckpt.update(protocol_sha256=SHA,parent_protocol_sha256=P["parent_protocol_sha256"])
            destination=target.with_suffix(".pt")
            torch.save(ckpt,destination)
            alias.update(protocol_sha256=SHA,parent_protocol_sha256=P["parent_protocol_sha256"],
                checkpoint=str(destination),training_budget=P["training_protocol"]["training"],
                reuse_reason="Identical 500/1/15 development-audit training; no test-based configuration choice",
                original_training_record=cfg["reuse_training_record"])
            bt.check(alias);bt.save(target.with_suffix(".train.json"),alias)
        bt.run_simple(a.source,"neural",name,a.seed)
    print("FOLLOWUP COMPLETE",a.source,name,a.seed,flush=True)
if __name__=="__main__":main()
