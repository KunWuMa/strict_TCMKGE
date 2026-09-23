# -*- coding: utf-8 -*-
"""Publication figures from frozen PairComp-KGE JSON artifacts."""
import json,math,textwrap
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parent; O=R/"results"; F=R/"figures"/"paper"; F.mkdir(parents=True,exist_ok=True)
N="#16324F"; T="#2A9D8F"; C="#E76F51"; G="#D99A00"; P="#6C5CE7"; B="#457B9D"; S="#A8DADC"; Y="#7A8491"; L="#E9EEF3"; D="#25313C"
COL={"PairComp-KGE":C,"Mean-ComplEx":T,"Matched control":B,"DeepSets":P,"Pair-only":G,"NodePiece-style":N,"InGram":Y,"Set Transformer":"#CC79A7","Frequency":D,"Random":"#C7CDD4"}
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9.2,"axes.titlesize":10.5,"axes.spines.top":False,"axes.spines.right":False,"legend.frameon":False,"figure.facecolor":"white","axes.facecolor":"white","savefig.facecolor":"white","pdf.fonttype":42,"ps.fonttype":42})
def rd(n):return json.loads((O/n).read_text(encoding="utf8"))
def sv(fig,n):
 fig.savefig(F/f"{n}.pdf",bbox_inches="tight");fig.savefig(F/f"{n}.png",dpi=450,bbox_inches="tight");plt.close(fig)
def pn(a,l,t):
 a.text(-.10,1.06,l,transform=a.transAxes,fontweight="bold",fontsize=11);a.set_title(t,loc="left",fontweight="bold",pad=8)
def main_evidence():
 e=rd("expanded_kge_baselines_detailed_analysis.json");h=rd("hybrid_confirmatory_detailed_analysis.json");r=rd("expert_systems_robustness.json");x=rd("expanded_kge_baselines_results.json");st=rd("set_transformer_clustered_comparison.json")
 ms=["PairComp-KGE","Mean-ComplEx","Matched control","DeepSets","Set Transformer"]
 q={"PairComp-KGE":e["overall"]["paircomp_kge"],"Mean-ComplEx":e["overall"]["mean_complex"],"Matched control":h["overall"]["parameter_control"],"DeepSets":h["overall"]["deepsets"],"Set Transformer":{"mrr":x["models"]["set_transformer"]["summary"]["mrr"]["mean"],"hits1":x["models"]["set_transformer"]["summary"]["hits1"]["mean"]}}
 f,a=plt.subplots(1,2,figsize=(11.7,4.25),gridspec_kw={"width_ratios":[1.15,1]},constrained_layout=True);y=np.arange(len(ms))[::-1]
 for i,m in enumerate(ms):
  a[0].plot([q[m]["hits1"],q[m]["mrr"]],[y[i],y[i]],color=L,lw=4)
  a[0].scatter(q[m]["mrr"],y[i],s=62,color=COL[m],edgecolor="white",label="MRR" if i==0 else None,zorder=3)
  a[0].scatter(q[m]["hits1"],y[i],s=47,marker="s",facecolor="white",edgecolor=COL[m],lw=1.6,label="Hits@1" if i==0 else None,zorder=3)
  a[0].text(q[m]["mrr"]+.009,y[i],f'{q[m]["mrr"]:.3f}',va="center",fontsize=8,color=COL[m])
 a[0].set(yticks=y,yticklabels=ms,xlim=(0,.72),xlabel="Typed filtered score");a[0].grid(axis="x",color=L);a[0].legend(loc="lower left",ncol=2);pn(a[0],"a","Head-of-ranking performance")
 cs=[("Mean-ComplEx",e["clustered_bootstrap"]["paircomp_vs_mean_complex"]),("Matched control",r["clustered_bootstrap"]["hybrid_vs_parameter_control"]),("DeepSets",r["clustered_bootstrap"]["hybrid_vs_deepsets"]),("Set Transformer",st["paircomp_vs_set_transformer"])]
 y=np.arange(len(cs))[::-1];a[1].axvline(0,color=D,ls="--")
 for yy,(n,z) in zip(y,cs):
  v=z["mrr_difference"];ci=z["cluster_bootstrap_95ci"];a[1].hlines(yy,*ci,color=COL[n],lw=3);a[1].scatter(v,yy,s=70,color=COL[n],edgecolor="white",zorder=3);a[1].text(ci[1]+.003,yy,f"{v:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}]",va="center",fontsize=8)
 a[1].set(yticks=y,yticklabels=[f"vs {x[0]}" for x in cs],xlim=(-.012,.060),xlabel="PairComp MRR difference (formula-clustered 95% CI)");a[1].grid(axis="x",color=L);pn(a[1],"b","Effect size and uncertainty");sv(f,"figure_2_main_evidence")
def mechanism():
 d=rd("hybrid_confirmatory_detailed_analysis.json");f,a=plt.subplots(1,3,figsize=(12.5,3.75),constrained_layout=True)
 sz=[("1–5","1_to_5"),("6–10","6_to_10"),(">10","over_10")];ms=[("PairComp-KGE","hybrid"),("Matched control","parameter_control"),("DeepSets","deepsets"),("Pair-only","pair_only")];x=np.arange(3)
 for n,k in ms:a[0].plot(x,[d["by_formula_size"][s][k+"_mrr"] for _,s in sz],marker="o",lw=2,color=COL[n],label=n)
 a[0].set(xticks=x,xticklabels=[s for s,_ in sz],xlabel="Formula size (herbs)",ylabel="MRR",ylim=(.52,.71));a[0].grid(axis="y",color=L);a[0].legend(fontsize=7.5,ncol=2,loc="lower left");pn(a[0],"a","Performance by formula size")
 gs=d["learned_gates"]["hybrid"];n=np.arange(2,21);cur=[]
 for g in gs:
  b=math.log(g["pair_gate_at_5"]/(1-g["pair_gate_at_5"]));cur.append(1/(1+np.exp(-(b+g["pair_gate_log_size"]*(np.log(n)-math.log(5))))))
 cur=np.array(cur);a[1].fill_between(n,cur.min(0),cur.max(0),color=C,alpha=.18,label="seed range");a[1].plot(n,cur.mean(0),color=C,lw=2.5,label=r"$\beta(n_f)$");a[1].axhline(np.mean([g["unary_gate"] for g in gs]),color=T,ls="--",lw=1.8,label=r"$\alpha$ (mean)")
 a[1].set(xlabel="Formula size $n_f$",ylabel="Learned gate",ylim=(.15,1),xticks=[2,5,10,15,20]);a[1].grid(axis="y",color=L);a[1].legend(fontsize=8);pn(a[1],"b","Size-adaptive pair gate")
 ns=["PairComp-KGE","Pair-only","DeepSets","Matched control"];ks=["hybrid","pair_only","deepsets","parameter_control"];v=[d["overall"][k]["mrr"] for k in ks];y=np.arange(4)[::-1];a[2].barh(y,v,color=[COL[n] for n in ns],height=.58)
 for yy,z in zip(y,v):a[2].text(z+.002,yy,f"{z:.3f}",va="center",fontsize=8)
 a[2].set(yticks=y,yticklabels=ns,xlim=(.60,.66),xlabel="MRR");a[2].grid(axis="x",color=L);pn(a[2],"c","Pair-specific controls");sv(f,"figure_3_mechanism")
def robustness():
 r=rd("expert_systems_robustness.json");p=rd("composition_permutation_extended.json");f,a=plt.subplots(1,2,figsize=(10.8,3.8),constrained_layout=True)
 x=np.array([0,.1,.2,.3,.5]);m=np.array([p["observed_mrr"]]+[r["ingredient_dropout"][str(z)]["mrr"]["mean"] for z in x[1:]]);s=np.array([0]+[r["ingredient_dropout"][str(z)]["mrr"]["std_across_masks"] for z in x[1:]])
 a[0].fill_between(x*100,m-s,m+s,color=T,alpha=.18);a[0].plot(x*100,m,color=T,lw=2.5,marker="o")
 for xx,z in zip(x*100,m):a[0].text(xx,z+.012,f"{z:.3f}",ha="center",fontsize=8)
 a[0].set(xlabel="Randomly removed ingredients (%)",ylabel="MRR",ylim=(.43,.69));a[0].grid(axis="y",color=L);pn(a[0],"a","Missing-ingredient robustness")
 n=np.array(p["null_mrr_values"]);v=a[1].violinplot(n,positions=[0],widths=.6,showmedians=True)
 for z in v["bodies"]:z.set_facecolor(S);z.set_edgecolor(B);z.set_alpha(.8)
 for k in ["cmedians","cbars","cmins","cmaxes"]:v[k].set_color(B)
 rng=np.random.default_rng(20260916);a[1].scatter(rng.normal(0,.045,len(n)),n,s=18,color=B,alpha=.55);a[1].scatter(0,p["observed_mrr"],s=95,marker="D",color=C,edgecolor="white",zorder=4);a[1].axhline(p["null_mrr_95percentile"],color=G,ls="--",lw=1.5,label="null 95th percentile");a[1].text(.12,p["observed_mrr"],f'correct composition = {p["observed_mrr"]:.3f}',va="center",color=C,fontweight="bold");a[1].text(.12,n.mean(),f'{p["permutations"]} size-matched permutations\nmean = {n.mean():.4f}; p = {p["empirical_p_upper"]:.3f}\n95th percentile = {p["null_mrr_95percentile"]:.4f}',va="center",fontsize=8);a[1].legend(loc="center right",fontsize=8)
 a[1].set(xlim=(-.45,1.3),ylim=(.09,.70),xticks=[],ylabel="MRR");a[1].grid(axis="y",color=L);pn(a[1],"b","Wrong-composition falsification");sv(f,"figure_4_robustness")
def transfer():
 e=rd("expanded_kge_baselines_detailed_analysis.json");t=rd("decoder_transfer_analysis.json");d=rd("dimension_sensitivity_results.json");q=rd("efficiency_results.json");f,a=plt.subplots(1,3,figsize=(12.8,3.8),constrained_layout=True)
 rs=[("ComplEx",e["clustered_bootstrap"]["paircomp_vs_mean_complex"]["mrr_difference"],e["clustered_bootstrap"]["paircomp_vs_mean_complex"]["cluster_bootstrap_95ci"])]
 for k in ["distmult","transe"]:
  z=t["decoders"][k]["pair_vs_mean_formula_cluster_bootstrap"];rs.append((k.title(),z["mrr_difference"],z["cluster_bootstrap_95ci"]))
 y=np.arange(3)[::-1];a[0].axvline(0,color=D,ls="--")
 for yy,(n,z,ci) in zip(y,rs):
  co=T if ci[0]>0 else Y;a[0].hlines(yy,*ci,color=co,lw=3);a[0].scatter(z,yy,s=70,color=co,edgecolor="white");a[0].text(ci[1]+.001,yy,f"{z:+.3f}",va="center",fontsize=8)
 a[0].set(yticks=y,yticklabels=[z[0] for z in rs],xlim=(-.012,.045),xlabel="Pair − mean MRR (95% CI)");a[0].grid(axis="x",color=L);pn(a[0],"a","Decoder transfer")
 ds=[64,128,256];m=[d["dimensions"][str(x)]["summary"]["mrr"]["mean"] for x in ds];me=[d["dimensions"][str(x)]["summary"]["mrr"]["sample_std"] for x in ds];h=[d["dimensions"][str(x)]["summary"]["hits1"]["mean"] for x in ds];he=[d["dimensions"][str(x)]["summary"]["hits1"]["sample_std"] for x in ds]
 a[1].errorbar(ds,m,yerr=me,color=C,marker="o",lw=2,capsize=3,label="MRR");a[1].errorbar(ds,h,yerr=he,color=P,marker="s",lw=2,capsize=3,label="Hits@1");a[1].set(xticks=ds,xlabel="Embedding dimension",ylabel="Score",ylim=(.50,.68));a[1].grid(axis="y",color=L);a[1].legend();pn(a[1],"b","Dimension sensitivity")
 ms=[("PairComp-KGE","hybrid"),("Matched control","parameter_control"),("DeepSets","deepsets"),("Pair-only","pair_only")];y=np.arange(4)[::-1];v=[q["models"][k]["batch_milliseconds_mean"] for _,k in ms];er=[q["models"][k]["batch_milliseconds_std"] for _,k in ms];a[2].barh(y,v,xerr=er,color=[COL[n] for n,_ in ms],height=.58,capsize=2)
 for yy,(n,k),z in zip(y,ms,v):a[2].text(z+.06,yy,f'{z:.2f} ms | {q["models"][k]["parameters"]/1e6:.2f} M',va="center",fontsize=7.6)
 a[2].set(yticks=y,yticklabels=[n for n,_ in ms],xlim=(0,4.25),xlabel="Batch time for 610 formulas");a[2].grid(axis="x",color=L);pn(a[2],"c","Encoding cost");sv(f,"figure_5_transfer_efficiency")
def external():
 d=rd("external_zero_shot_results.json");f,a=plt.subplots(1,3,figsize=(12.8,3.8),constrained_layout=True);ms=[("PairComp-KGE","hybrid"),("Matched control","parameter_control"),("DeepSets","deepsets"),("Pair-only","pair_only")];v=[d["overall"][k]["mrr"]["mean"] for _,k in ms]+[d["frequency"]["mrr"],d["uniform_random_expectation"]["metrics"]["mrr"]];er=[d["overall"][k]["mrr"]["sample_std"] for _,k in ms]+[0,0];lb=[n for n,_ in ms]+["Frequency","Random"];a[0].bar(np.arange(6),v,yerr=er,color=[COL[x] for x in lb],capsize=2);a[0].set(xticks=np.arange(6),xticklabels=lb,ylabel="MRR",ylim=(0,.31));a[0].tick_params(axis="x",rotation=32);a[0].grid(axis="y",color=L);pn(a[0],"a","Frozen external zero-shot ranking")
 cs=[("Matched control","hybrid_vs_parameter_control"),("DeepSets","hybrid_vs_deepsets"),("Pair-only","hybrid_vs_pair_only")];y=np.arange(3)[::-1];a[1].axvline(0,color=D,ls="--")
 for yy,(n,k) in zip(y,cs):
  z=d["patient_cluster_bootstrap"][k]["mrr"];v=z["difference"];ci=z["patient_cluster_bootstrap_95ci"];a[1].hlines(yy,*ci,color=Y,lw=3);a[1].scatter(v,yy,s=70,color=Y,edgecolor="white");a[1].text(ci[1]+.004,yy,f"{v:+.3f}",va="center",fontsize=8)
 a[1].set(yticks=y,yticklabels=[f"vs {n}" for n,_ in cs],xlim=(-.05,.115),xlabel="PairComp MRR difference\n(patient-clustered 95% CI)");a[1].grid(axis="x",color=L);pn(a[1],"b","No confirmed pair-specific transfer")
 q=d["coverage"];lb=["Herb\ntypes","Herb\nrecords","Syndrome\nstrings","Included\nprescriptions"];v=[q["herb_unique_exact_coverage"],q["herb_record_exact_coverage"],q["syndrome_unique_exact_coverage"],q["included_prescriptions"]/q["total_prescriptions"]];a[2].bar(np.arange(4),np.array(v)*100,color=[T,B,G,C])
 for x,z in enumerate(v):a[2].text(x,z*100+2,f"{z*100:.1f}%",ha="center",fontsize=8)
 a[2].set(xticks=np.arange(4),xticklabels=lb,ylim=(0,100),ylabel="Coverage (%)");a[2].grid(axis="y",color=L);pn(a[2],"c","Exact-mapping bottleneck");sv(f,"figure_6_external_stress_test")
def attribution():
 cs=rd("pair_attribution_cases.json")["cases"][:3];f,a=plt.subplots(1,3,figsize=(13.2,4.5),constrained_layout=True)
 for i,(ax,c) in enumerate(zip(a,cs)):
  rs=c["top_positive_pairs"][:3]+c["top_negative_pairs"][:3];lb=[];v=[];er=[]
  for r in rs:
   x=r["herb_a"].get("English_term") or r["herb_a"]["id"];y=r["herb_b"].get("English_term") or r["herb_b"]["id"];lb.append(textwrap.fill(x+" + "+y,24));v.append(r["attribution_mean"]);er.append(r["attribution_std"])
  o=np.argsort(v);lb=np.array(lb)[o];v=np.array(v)[o];er=np.array(er)[o];y=np.arange(len(v));ax.barh(y,v,xerr=er,color=[C if z>0 else B for z in v],capsize=2);ax.axvline(0,color=D,lw=.8);ax.set(yticks=y,yticklabels=lb,xlabel="Local pair sensitivity to target score");ax.tick_params(axis="y",labelsize=7.3);tt=(c["formula"].get("Pinyin_term") or c["formula"]["id"])+" → "+(c["target"].get("English_term") or c["target"]["id"]);pn(ax,chr(97+i),textwrap.fill(tt,31));ax.grid(axis="x",color=L)
 f.suptitle("Gradient-based local pair sensitivities (descriptive, not clinical validation)",fontweight="bold",fontsize=11.5,y=1.03);sv(f,"figure_7_pair_attributions")
def main():
 main_evidence();mechanism();robustness();transfer();external();attribution();print(F.resolve())
if __name__=="__main__":main()
