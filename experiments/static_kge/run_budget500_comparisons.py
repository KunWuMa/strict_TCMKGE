"""Resource-aware, restartable three-process queue for the 500/1/15 comparison rerun."""
from __future__ import annotations
import argparse,hashlib,json,os,subprocess,sys,time,shutil,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'outputs/official_kge_dependencies'))
import psutil
from pairproto_execution_lock import ExclusiveLock
RUN=ROOT/'outputs/comparison_budget500_20261024'
P=json.loads((RUN/'protocol.json').read_text(encoding='utf-8'))
SHA=hashlib.sha256((RUN/'protocol.json').read_bytes()).hexdigest()
ENV=os.environ.copy();ENV.update(PYTHONIOENCODING='utf-8',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
ENV['PYTHONPATH']=str(ROOT/'outputs/official_kge_dependencies')
SI=0x08000000

def save(path,value):
    temporary=path.with_name(path.name+f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value,indent=2,ensure_ascii=False),encoding='utf-8')
    temporary.replace(path)

def key(job):
    return f"{job['source'].lower()}_{job['family']}_{job['model']}_seed{job['seed']}"

def result_path(job):
    return RUN/'results'/(key(job)+'.evaluation.json')

def completed(job):
    path=result_path(job)
    if not path.exists():return False
    record=json.loads(path.read_text(encoding='utf-8'));assert record['protocol_sha256']==SHA
    if job['family']=='neural' and job['model']=='mean':
        proposed=RUN/'results'/f"{job['source'].lower()}_proposed_pairproto_seed{job['seed']}.evaluation.json"
        return proposed.exists() and json.loads(proposed.read_text(encoding='utf-8'))['protocol_sha256']==SHA
    return True

def jobs():
    order=[('neural','set_transformer'),('neural','mean'),('extended','mean_distmult'),
      ('extended','mean_transe'),('neural','rotate'),('neural','pairre'),
      ('neural','full_deepsets'),('extended','nodepiece'),('extended','ultra'),
      ('extended','trix'),('extended','ingram'),('neural','pair')]
    return [{'source':source,'family':family,'model':name,'seed':seed}
       for seed in P['seeds'] for family,name in order for source in P['datasets']]

def resources():
    memory=psutil.virtual_memory()
    result={'ram_available_gb':memory.available/2**30,'ram_total_gb':memory.total/2**30}
    tool=shutil.which('nvidia-smi')
    if tool:
        try:
            value=subprocess.check_output([tool,'--query-gpu=memory.free,memory.total,utilization.gpu',
              '--format=csv,noheader,nounits'],text=True,timeout=10,creationflags=SI).strip().splitlines()[0]
            free,total,util=[float(v.strip()) for v in value.split(',')]
            result.update(gpu_free_gb=free/1024,gpu_total_gb=total/1024,gpu_utilization_percent=util)
        except (OSError,subprocess.SubprocessError,ValueError):pass
    return result

def status(tasks,pending,active,failures,limit,phase='training'):
    records=[]
    for item in active:
        record={**item['job'],'pid':item['process'].pid,'log':str(item['log_path'])}
        progress=RUN/'results'/(key(item['job'])+'.progress.json')
        if progress.exists():
            try:record['progress']=json.loads(progress.read_text(encoding='utf-8'))
            except (PermissionError,json.JSONDecodeError):pass
        records.append(record)
    evaluations=list((RUN/'results').glob('*.evaluation.json'))
    state={'phase':phase,'complete':phase=='complete','protocol_sha256':SHA,
      'updated_local':time.strftime('%Y-%m-%d %H:%M:%S'),'supervisor_pid':os.getpid(),
      'training_tasks':len(tasks),'completed_tasks':sum(completed(job) for job in tasks),
      'evaluations_present':len(evaluations),'expected_evaluations':130,
      'pending_tasks':len(pending),'active':records,'failed':failures,
      'current_parallel_limit':limit,'resources':resources()}
    save(RUN/'status.json',state)
    return state

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--max-workers',type=int,default=3)
    args=parser.parse_args();assert 1<=args.max_workers<=3
    gate=json.loads((RUN/'preflight.json').read_text(encoding='utf-8'))
    assert gate['passed'] and gate['protocol_sha256']==SHA
    assert hashlib.sha256(Path(P['prior_workbook']['path']).read_bytes()).hexdigest()==P['prior_workbook']['sha256']
    tasks=jobs();assert len(tasks)==120
    pending=[job for job in tasks if not completed(job)]
    if len(pending)<len(tasks):
        pending.sort(key=lambda job:job['model']=='set_transformer')
    active=[];failures=[];attempts={};limit=args.max_workers;last_status=0
    with ExclusiveLock(RUN/'supervisor.lock'):
        while pending or active:
            for item in active[:]:
                exitcode=item['process'].poll()
                if exitcode is None:continue
                item['handle'].close();active.remove(item)
                job=item['job'];identity=key(job)
                if exitcode==0 and completed(job):
                    print('COMPLETE',identity,flush=True)
                    continue
                tail=item['log_path'].read_text(encoding='utf-8',errors='replace')[-7000:]
                oom=any(word in tail.lower() for word in ['out of memory','alloc_failed','memoryallocation'])
                if oom and attempts[identity]<3:
                    limit=max(1,limit-1);pending.append(job)
                    print('RETRY AT LOWER CONCURRENCY',identity,limit,flush=True)
                else:
                    failures.append({**job,'exit_code':exitcode,'log':str(item['log_path']),'tail':tail[-1800:]})
                    print('FAILED',identity,exitcode,flush=True)
            while pending and len(active)<limit:
                available=resources()
                if available['ram_available_gb']<P['scheduling']['reserve_host_ram_gb']+1.:break
                if available.get('gpu_free_gb',100)<P['scheduling']['minimum_free_gpu_gb_to_launch']:break
                heavy={'set_transformer','nodepiece','ingram','trix'}
                choice=next((i for i,job in enumerate(pending)
                    if job['model'] not in heavy or sum(x['job']['model'] in heavy for x in active)<2),None)
                if choice is None:break
                job=pending.pop(choice);identity=key(job)
                previous=list((RUN/'logs').glob(identity+'.attempt*.log'))
                previous_numbers=[int(p.stem.rsplit('attempt',1)[1]) for p in previous]
                attempts[identity]=max([attempts.get(identity,0),*previous_numbers])+1
                log_path=RUN/'logs'/f"{identity}.attempt{attempts[identity]}.log"
                handle=log_path.open('w',encoding='utf-8')
                command=[sys.executable,'-u',str(ROOT/'budget500_training.py'),
                  '--source',job['source'],'--family',job['family'],'--model',job['model'],'--seed',str(job['seed'])]
                process=subprocess.Popen(command,cwd=ROOT,env=ENV,stdout=handle,stderr=subprocess.STDOUT,creationflags=SI)
                active.append({'job':job,'process':process,'handle':handle,'log_path':log_path})
                print('START',identity,process.pid,flush=True);time.sleep(3)
            if time.time()-last_status>=15:
                status(tasks,pending,active,failures,limit);last_status=time.time()
            time.sleep(3)
        if failures:
            status(tasks,pending,active,failures,limit,'incomplete')
            raise RuntimeError('Some runs failed; other completed runs remain available. Inspect status.json.')
        assert all(completed(job) for job in tasks)
        status(tasks,[],[],[],limit,'exporting_excel')
        subprocess.run([sys.executable,'-u',str(ROOT/'budget500_results.py')],cwd=ROOT,env=ENV,check=True)
        status(tasks,[],[],[],limit,'complete')
        print('ALL 500/1/15 COMPARISONS COMPLETE',flush=True)

if __name__=='__main__':
    try:main()
    except Exception:
        traceback.print_exc()
        raise
