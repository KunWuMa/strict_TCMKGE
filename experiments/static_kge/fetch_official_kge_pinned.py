"""Download pinned upstream code only; never download their benchmark data."""
import argparse,hashlib,io,json,time,urllib.request,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parent
REPOS={"NodePiece":("migalkin/NodePiece","9adc57efe302919d017d74fc648f853308cf75fd"),
"ULTRA":("DeepGraphLearning/ULTRA","427966ad8ed60420eef034063d44f3153addff90"),
"TRIX":("yuchengz99/TRIX","7596e14eefefe89e61396205a0550172cadeddb0"),
"InGram":("bdi-lab/InGram","f2554dd0c684bad71d3e01d534f2f7814cf28644")}
def main():
    ap=argparse.ArgumentParser();ap.add_argument("--models",nargs="+",choices=list(REPOS),default=list(REPOS));a=ap.parse_args()
    for name in a.models:
        repo,commit=REPOS[name];url=f"https://codeload.github.com/{repo}/zip/{commit}"
        req=urllib.request.Request(url,headers={"User-Agent":"PairProto-reproduction"})
        with urllib.request.urlopen(req,timeout=90) as response:blob=response.read()
        archive=zipfile.ZipFile(io.BytesIO(blob));dest=ROOT/"third_party"/name
        rows=[]
        for info in archive.infolist():
            if info.is_dir():continue
            rel=Path(*Path(info.filename).parts[1:])
            if rel.parts[0] in {"data","datasets","llm",".git"}:continue
            if not (rel.suffix in {".py",".yaml",".yml",".toml"} or rel.name.lower() in {"readme.md","requirements.txt","setup.cfg","license","license.txt","license.md"}):continue
            target=(dest/rel).resolve()
            assert target.is_relative_to(dest.resolve())
            target.parent.mkdir(parents=True,exist_ok=True);body=archive.read(info)
            target.write_bytes(body);rows.append({"path":rel.as_posix(),"sha256":hashlib.sha256(body).hexdigest()})
        dest.mkdir(parents=True,exist_ok=True)
        (dest/"PINNED_SOURCE_MANIFEST.json").write_text(json.dumps({"repository":"https://github.com/"+repo,"commit":commit,"files":rows},indent=2),encoding="utf-8")
        print(name,commit,len(rows))
if __name__=="__main__":main()
