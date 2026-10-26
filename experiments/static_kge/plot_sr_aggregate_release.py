"""Render summary-only plots; never reads or distributes query records."""
import argparse,json
from pathlib import Path

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--summary",type=Path,default=Path(__file__).parent/"published_results/scientific_reports_20261026/aggregate_summary.json")
    ap.add_argument("--outdir",type=Path,default=Path(__file__).parent/"figures/scientific_reports_20261026_recreated")
    a=ap.parse_args();r=json.loads(a.summary.read_text(encoding="utf-8"))
    import plot_pairproto_paper as p
    p.FIG=a.outdir;p.FIG.mkdir(parents=True,exist_ok=True)
    p.PAPER=a.outdir/"no-manuscript-target"
    p.main_comparison(r);p.mechanism(r);p.uncertainty(r);p.robustness(r);p.relation_results(r)
    print("Aggregate plots:",a.outdir)
if __name__=="__main__":main()
