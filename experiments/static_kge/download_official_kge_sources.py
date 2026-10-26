import urllib.request,json,hashlib,time,zipfile,io,concurrent.futures
from pathlib import Path
ROOT=Path(__file__).resolve().parent
repos={'NodePiece':('migalkin/NodePiece','main'),'ULTRA':('DeepGraphLearning/ULTRA','main'),'TRIX':('yuchengz99/TRIX','main')}
def fetch(url):
 for attempt in range(4):
  try:return urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'TCMKGE-research'}),timeout=120).read()
  except Exception:
   if attempt==3:raise
   time.sleep(2*(attempt+1))
def job(item):
 name,(repo,branch)=item;dest=ROOT/'third_party'/name;dest.mkdir(parents=True,exist_ok=True)
 blob=fetch(f'https://codeload.github.com/{repo}/zip/refs/heads/{branch}')
 z=zipfile.ZipFile(io.BytesIO(blob));manifest=[]
 for entry in z.infolist():
  if entry.is_dir():continue
  p='/'.join(entry.filename.split('/')[1:])
  if not(p.endswith(('.py','.yaml','.yml','.toml')) or p.split('/')[-1].lower() in ['requirements.txt','license','license.txt','readme.md','setup.cfg']):continue
  if p.startswith(('data/','datasets/','llm/')):continue
  target=(dest/p).resolve()
  if not target.is_relative_to(dest.resolve()):raise ValueError(p)
  target.parent.mkdir(parents=True,exist_ok=True);b=z.read(entry);target.write_bytes(b)
  manifest.append({'path':p,'sha256':hashlib.sha256(b).hexdigest()})
 record={'repository':'https://github.com/'+repo,'branch':branch,'zip_sha256':hashlib.sha256(blob).hexdigest(),'zip_comment':z.comment.decode('ascii',errors='replace'),'files':manifest,'downloaded_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
 (dest/'TCMKGE_SOURCE_MANIFEST.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
 print(name,len(manifest),record['zip_comment'],flush=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
 for x in pool.map(job,repos.items()):pass
