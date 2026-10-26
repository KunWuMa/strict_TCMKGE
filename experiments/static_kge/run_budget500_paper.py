"""Restartable dependency queue for every remaining 500/1/15 paper experiment."""
from __future__ import annotations
import argparse,hashlib,json,os,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'outputs/official_kge_dependencies'))
from run_budget500_comparisons import resources
from pairproto_execution_lock import ExclusiveLock
BASE=ROOT/'outputs/paper_budget500_20261025'
P=json.loads((BASE/'protocol.json').read_text(encoding='utf-8'))
SHA=hashlib.sha256((BASE/'protocol.json').read_bytes()).hexdigest()
ENV=os.environ.copy();ENV.update(PYTHONIOENCODING='utf-8',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',
    PYTHONPATH=str(ROOT/'outputs/official_kge_dependencies'))
def save(path,value):
    tmp=path.with_name(path.name+f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False),encoding='utf-8');tmp.replace(path)
def completed(job):
    path=BASE/'completed'/(job['id']+'.json')
    if not path.exists():return False
    value=json.loads(path.read_text(encoding='utf-8'))
    assert value['protocol_sha256']==SHA and value['job']==job
    return True
def priority(j):
    k=j['kind']
    if k=='train':
        tag=j['id'].split('__',1)[1]
        if tag in P['reuse']:return 0
        main=P['datasets'][j['source']]['main']
        cfg=P['selected'][j['source']].get(j['model'],{})
        if j['split']==main and j['lr']==cfg.get('lr') and j['heads']==cfg.get('heads',4):return 2
        return 4 if j['split']==main else 3
    return {'mechanism_lock':1,'herb_lock':1,'prototype':2,'mechanism':2,'herb':2,
        'dev_audit':5,'sensitivity':5,'boundary':6,'measurements':9}[k]
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--max-workers',type=int,default=3)
    args=ap.parse_args();assert 1<=args.max_workers<=3
    gate=json.loads((BASE/'preflight.json').read_text(encoding='utf-8'))
    assert gate['passed'] and gate['protocol_sha256']==SHA
    jobs=P['jobs'];pending=sorted([j for j in jobs if not completed(j)],key=lambda j:(priority(j),j.get('seed',0),j['id']))
    active=[];failed=[];limit=args.max_workers;last=0
    def status(phase):
        done={j['id'] for j in jobs if completed(j)}
        counts={}
        for kind in sorted({j['kind'] for j in jobs}):
            selected=[j for j in jobs if j['kind']==kind]
            counts[kind]={'completed':sum(j['id'] in done for j in selected),'total':len(selected)}
        shown=[]
        for item in active:
            j=item['job'];row={**j,'pid':item['p'].pid,'log':str(item['path'])}
            progress=list((BASE/'training'/j['id'].split('__',1)[-1]).glob('*.progress.json')) if j['kind'] in ('train','dev_audit') else []
            if progress:
                try:row['progress']=json.loads(progress[0].read_text(encoding='utf-8'))
                except (OSError,json.JSONDecodeError):pass
            shown.append(row)
        state={'phase':phase,'complete':phase=='complete','protocol_sha256':SHA,
            'supervisor_pid':os.getpid(),'updated_local':time.strftime('%Y-%m-%d %H:%M:%S'),
            'completed_jobs':len(done),'total_jobs':len(jobs),'counts':counts,'pending_jobs':len(pending),
            'active':shown,'failed':failed,'parallel_limit':limit,'resources':resources()}
        save(BASE/'status.json',state)
        return done
    with ExclusiveLock(BASE/'supervisor.lock'):
        while pending or active:
            for item in active[:]:
                code=item['p'].poll()
                if code is None:continue
                item['handle'].close();active.remove(item);j=item['job']
                if code==0 and completed(j):
                    print('COMPLETE',j['id'],flush=True);continue
                tail=item['path'].read_text(encoding='utf-8',errors='replace')[-7000:]
                native=code in [3221226505,3221225477,-1073740791,-1073741819]
                transient=native or any(t in tail.lower() for t in ['out of memory','illegal memory access','alloc_failed'])
                if transient and item['attempt']<3:
                    limit=1;pending.append(j)
                    print('RETRY SERIAL',j['id'],code,flush=True)
                else:
                    failed.append({'job':j,'exit_code':code,'log':str(item['path']),'tail':tail[-2500:]})
                    print('FAILED',j['id'],code,flush=True)
            done={j['id'] for j in jobs if completed(j)}
            while len(active)<limit:
                r=resources()
                if r['ram_available_gb']<8 or r.get('gpu_free_gb',100)<2.5:break
                choice=None
                memory_kinds={'prototype','mechanism','mechanism_lock','herb_lock','sensitivity'}
                for i,j in enumerate(pending):
                    if not set(j['deps'])<=done:continue
                    if j['kind']=='measurements' and active:continue
                    if j.get('model')=='set_transformer' and any(x['job'].get('model')=='set_transformer' for x in active):continue
                    if j['kind'] in memory_kinds and any(x['job']['kind'] in memory_kinds for x in active):continue
                    choice=i;break
                if choice is None:break
                j=pending.pop(choice)
                numbers=[int(p.stem.rsplit('attempt',1)[1]) for p in (BASE/'logs').glob(j['id']+'.attempt*.log')]
                attempt=max(numbers,default=0)+1
                path=BASE/'logs'/f"{j['id']}.attempt{attempt}.log";handle=path.open('w',encoding='utf-8')
                p=subprocess.Popen([sys.executable,'-u',str(ROOT/'budget500_paper_worker.py'),'--job',j['id']],
                    cwd=ROOT,env=ENV,stdout=handle,stderr=subprocess.STDOUT,creationflags=0x08000000)
                active.append({'job':j,'p':p,'path':path,'handle':handle,'attempt':attempt})
                print('START',j['id'],p.pid,flush=True);time.sleep(2)
            if time.time()-last>=15:status('running');last=time.time()
            if pending and not active and failed:
                status('blocked_by_failed_dependencies')
                raise RuntimeError('Independent work completed; failed dependencies require repair')
            time.sleep(3)
        if failed:status('incomplete');raise RuntimeError('Inspect failed jobs')
        assert all(completed(j) for j in jobs)
        status('exporting')
        subprocess.run([sys.executable,'-u',str(ROOT/'budget500_paper_export.py')],cwd=ROOT,env=ENV,check=True)
        status('complete')
        print('ALL PAPER EXPERIMENTS AND EXCEL-DERIVED ARTIFACTS COMPLETE',flush=True)
if __name__=='__main__':
    try:main()
    except BaseException:
        traceback.print_exc()
        raise
