"""Run two follow-up streams with bounded concurrency and durable status."""
import os,sys,json,time,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parent
BASE=ROOT/"outputs/scientific_reports_20261026"
P=json.loads((BASE/"protocol.json").read_text(encoding="utf-8"))
jobs=[(source,seed) for seed in P["seeds"] for source in P["selected"]]
env=os.environ.copy()
env.update(PYTHONIOENCODING="utf-8",OMP_NUM_THREADS="2",MKL_NUM_THREADS="2",
    PYTHONPATH=str(ROOT/"outputs/official_kge_dependencies"))
env.pop("PAIRPROTO_INPUT_MODE",None)
active=[];done=[];failed=[]
def status():
    obj={"updated_local":time.strftime("%Y-%m-%d %H:%M:%S"),"total":len(jobs),
         "complete":len(done)==len(jobs),"done":done,"failed":failed,
         "running":[{"source":s,"seed":seed,"pid":p.pid} for s,seed,p,h in active]}
    path=BASE/"status.json";tmp=path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding="utf-8");tmp.replace(path)
waiting=list(jobs)
while waiting or active:
    for entry in list(active):
        source,seed,process,handle=entry
        rc=process.poll()
        if rc is not None:
            handle.close();active.remove(entry)
            (done if rc==0 else failed).append({"source":source,"seed":seed,"exit_code":rc})
    while waiting and len(active)<2 and not failed:
        source,seed=waiting.pop(0)
        handle=(BASE/"logs"/f"{source.lower()}_{seed}.log").open("w",encoding="utf-8")
        process=subprocess.Popen([sys.executable,str(ROOT/"scientific_reports_baseline_worker.py"),
             "--source",source,"--seed",str(seed)],cwd=str(ROOT),env=env,
             stdout=handle,stderr=subprocess.STDOUT,creationflags=0x08000000)
        active.append((source,seed,process,handle))
    status()
    if failed and not active:break
    time.sleep(3)
status()
if failed:raise SystemExit(1)
print("FOLLOWUP MATRIX COMPLETE",len(done),flush=True)
