from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
R=ROOT/"results"
O=ROOT/"figures"/"paper"
O.mkdir(parents=True,exist_ok=True)
gate=json.loads((R/"gate_ablation_results.json").read_text(encoding="utf-8"))
trans=json.loads((R/"evaluation_transparency.json").read_text(encoding="utf-8"))
COL={"PairComp":"#0072B2","Fixed gate":"#CC79A7","No gate":"#999999"}
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9,"axes.titlesize":10,"axes.labelsize":9,"legend.fontsize":8,"axes.spines.top":False,"axes.spines.right":False})
fig,ax=plt.subplots(1,3,figsize=(14.2,4.15),constrained_layout=True)

names=["Learned size gate","Fixed gate","No gate"]
keys=["learned_size_gate","fixed_pair_gate","ungated_pair"]
colors=[COL["PairComp"],COL["Fixed gate"],COL["No gate"]]
means=[gate["models"][k]["summary"]["mrr"]["mean"] for k in keys]
std=[gate["models"][k]["summary"]["mrr"]["sample_std"] for k in keys]
bars=ax[0].bar(np.arange(3),means,yerr=std,color=colors,capsize=3,width=.62)
ax[0].set(xticks=np.arange(3),xticklabels=names,ylim=(.61,.66),ylabel="Typed filtered MRR",title="a  Gate ablation")
ax[0].tick_params(axis="x",rotation=12)
ax[0].grid(axis="y",color="#E6E6E6",lw=.7)
for b,v in zip(bars,means): ax[0].text(b.get_x()+b.get_width()/2,v+.004,f"{v:.3f}",ha="center",va="bottom",fontsize=8)

bins=["1-5","6-10","11+"]
g=gate["branch_magnitudes"]["learned_size_gate"]
mu=[g[b]["mu_norm"] for b in bins]; unary=[g[b]["unary_norm"] for b in bins]; pair=[g[b]["pair_norm"] for b in bins]; beta=[g[b]["pair_gate"] for b in bins]
xx=np.arange(3);w=.23
ax[1].bar(xx-w,mu,w,label="Mean residual",color="#56B4E9")
ax[1].bar(xx,unary,w,label="Unary branch",color="#E69F00")
ax[1].bar(xx+w,pair,w,label="Gated pair branch",color="#0072B2")
ax[1].set(xticks=xx,xticklabels=bins,xlabel="Number of herbs",ylabel="Mean complex-vector norm",title="b  Learned branch magnitudes")
ax[1].grid(axis="y",color="#E6E6E6",lw=.7)
ax[1].legend(frameon=False,loc="upper left")
a2=ax[1].twinx();a2.plot(xx,beta,color="#CC79A7",marker="D",lw=1.8,label="Pair gate")
a2.set(ylabel="Pair gate",ylim=(0,1.08));a2.spines["top"].set_visible(False)

rows=[v for v in trans["relations"].values() if v["queries"]>=5]
rows.sort(key=lambda z:z["queries"],reverse=True)
label_map={(460,258):"Disease",(429,142):"Treatment method",(195,175):"Syndrome",(5,3):"Treatment principle"}
labels=[label_map.get((z["queries"],z["typed_candidates"]),f'n={z["queries"]}') for z in rows]
delta=[z["paircomp"]["mrr"]["mean"]-z["matched"]["mrr"]["mean"] for z in rows]
sizes=[35+0.35*z["queries"] for z in rows]
y=np.arange(len(rows))[::-1]
ax[2].axvline(0,color="#555555",ls="--",lw=1)
for d,yy,z,ss in zip(delta,y,rows,sizes):
    descriptive=z["queries"]<=5
    ax[2].scatter(d,yy,s=ss,facecolor="white" if descriptive else "#0072B2",edgecolor="#0072B2",alpha=.9,lw=1.4 if descriptive else .8)
    suffix="; descriptive" if descriptive else ""
    ax[2].text(d+(.0015 if d>=0 else -.0015),yy,f'{d:+.3f}  ({z["typed_candidates"]} cand.{suffix})',va="center",ha="left" if d>=0 else "right",fontsize=8)
ax[2].set(yticks=y,yticklabels=labels,xlabel="PairComp - matched-control MRR",title="c  Relation-wise effects")
ax[2].grid(axis="x",color="#E6E6E6",lw=.7)
fig.suptitle("Robustness and evaluation transparency",fontsize=13,fontweight="bold")
fig.savefig(O/"figure_8_review_evidence.png",dpi=300,bbox_inches="tight")
fig.savefig(O/"figure_8_review_evidence_abc.pdf",dpi=300,bbox_inches="tight")
print(O/"figure_8_review_evidence.png")
print(O/"figure_8_review_evidence_abc.pdf")
