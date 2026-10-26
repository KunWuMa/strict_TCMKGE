"""English, vector-first figures generated exclusively from saved evidence."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from pairproto_evidence import (OUT, PLAN, SOURCES, collect, prototype_tag, config_for, npz, build_report)

HERE=Path(__file__).resolve().parent
PAPER=HERE/'manuscript'/'_ExpertSystem_PairComp_KGE__Pair_Interaction_Aware_Compositional_Inductive_Knowledge_Graph_Embedding_for_Traditional_Chinese_Medicine'/'Optimal-Design-layout'
FIG=HERE/'figures'/'pairproto_paper';FIG.mkdir(parents=True,exist_ok=True)
BLUE='#287DAD';PURPLE='#7951A1';ORANGE='#C56D32';GRAY='#82909A'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.titlesize':12,
    'axes.labelsize':11,'xtick.labelsize':10,'ytick.labelsize':10,'legend.fontsize':10,
    'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,
    'axes.spines.right':False,'savefig.dpi':240})


def save(fig,name):
    fig.savefig(FIG/(name+'.pdf'),bbox_inches='tight')
    fig.savefig(FIG/(name+'.svg'),bbox_inches='tight')
    fig.savefig(FIG/(name+'.png'),bbox_inches='tight')
    if PAPER.exists():shutil.copy2(FIG/(name+'.pdf'),PAPER/(name+'.pdf'))
    plt.close(fig)


def pending(ax,message='Declared results pending'):
    ax.text(.5,.5,message,ha='center',va='center',transform=ax.transAxes,color=GRAY)
    ax.set_xticks([]);ax.set_yticks([])


def main_comparison(report):
    methods=[('mean','Mean-ComplEx'),('neural_full_deepsets','Deep Sets'),
        ('neural_set_transformer','Set Transformer adapter'),('pairproto','PairProto-KGE')]
    fig,axes=plt.subplots(1,2,figsize=(10,2.9))
    for panel,(ax,(source,resource)) in enumerate(zip(axes,report['sources'].items())):
        models=resource['main']['models'];y=np.arange(len(methods))
        for i,(key,label) in enumerate(methods):
            if key not in models:
                ax.text(.015,i,'Pending',va='center',color=GRAY,fontsize=8)
                continue
            value=models[key]['summary']['mrr'];sd=value['sample_std'] or 0
            ax.barh(i,value['mean'],height=.62,color=PURPLE if key=='pairproto' else BLUE if key=='mean' else GRAY,alpha=.95)
            ax.errorbar(value['mean'],i,xerr=sd,fmt='none',ecolor='#263644',capsize=2,lw=1)
            ax.annotate(f"{value['mean']:.3f} (n={models[key]['runs']})",(value['mean']+sd,i),xytext=(5,0),textcoords='offset points',va='center',fontsize=7.5)
        ax.set_yticks(y,[label for _,label in methods]);ax.invert_yaxis()
        highest=max((m['summary']['mrr']['mean'] for m in models.values()),default=.5)
        ax.set_xlim(0,highest*1.32);ax.set_xlabel('Filtered MRR (mean ± seed SD)')
        ax.set_title(f'({chr(97+panel)}) '+SOURCES[source],loc='left');ax.grid(axis='x',alpha=.14);ax.set_axisbelow(True)
    if not report['complete']:
        fig.suptitle('Working results: development selection or declared runs pending',fontsize=10,color=GRAY)
    fig.tight_layout(w_pad=2);save(fig,'pairproto_main_comparison')


def mechanism(report):
    controls=[('mean','Mean-ComplEx'),('herb_prototype','Herb prototype'),
      ('herb_prototype_dev_selected','Herb prototype (dev-selected)'),
      ('neural_pair','Uniform pair moment'),('neural_second_moment','Second moment'),
      ('neural_matched','Dual mean'),('pair_identity_shuffle','Pair identity shuffled'),
      ('relation_identity_shuffle','Relation identity shuffled'),('shared_pair_prototypes','Shared pair prototypes'),
      ('posterior_smoothing_only','Posterior smoothing only'),('no_shrinkage','No support shrinkage'),
      ('support_one','Minimum support = 1'),('raw_pair_dev_selected','Raw scores (dev-selected)')]
    fig,axes=plt.subplots(1,2,figsize=(10,5.5))
    for ax,(source,resource) in zip(axes,report['sources'].items()):
        comparisons=resource['main']['comparisons'];drawn=False
        for i,(key,label) in enumerate(controls):
            if key not in comparisons:
                ax.text(.75,i,'Pending',transform=ax.get_yaxis_transform(),color=GRAY,fontsize=7,va='center')
                continue
            row=comparisons[key]['recipe_mrr'];lo,hi=row['recipe_cluster_ci95'];x=row['difference']
            ax.errorbar(x,i,xerr=[[max(0,x-lo)],[max(0,hi-x)]],fmt='o',color=PURPLE,capsize=2,ms=4);drawn=True
        ax.axvline(0,color=GRAY,lw=1,ls='--');ax.set_yticks(range(len(controls)),[v for _,v in controls]);ax.invert_yaxis()
        ax.set_title(SOURCES[source]);ax.set_xlabel('PairProto − control: ΔMRR\n95% recipe-cluster CI')
        ax.grid(axis='x',alpha=.14)
        if not drawn:ax.text(.5,.02,'Controls pending',transform=ax.transAxes,color=GRAY,ha='center')
    fig.tight_layout(w_pad=2);save(fig,'pairproto_mechanism_ablation')


def uncertainty(report):
    fig,axes=plt.subplots(1,2,figsize=(9,3.4),sharey=True)
    names=[('mean','Mean-ComplEx'),('herb_prototype_dev_selected','Dev-selected herb prototype'),('raw_pair_dev_selected','Raw pair scores')]
    for ax,metric,label in zip(axes,['recipe_mrr','recipe_hits1'],['MRR','Hits@1']):
        labels=[];i=0
        for source,resource in report['sources'].items():
            for name,title in names:
                comparison=resource['main']['comparisons'].get(name)
                if not comparison:continue
                row=comparison[metric];lo,hi=row['recipe_cluster_ci95'];x=row['difference']
                ax.errorbar(x,i,xerr=[[max(0,x-lo)],[max(0,hi-x)]],fmt='o',capsize=3,color=BLUE if source=='PRIMARY' else ORANGE)
                labels.append(SOURCES[source]+' / '+title);i+=1
        ax.axvline(0,color=GRAY,ls='--',lw=1);ax.set_yticks(range(i),labels);ax.set_ylim(i-.5,-.5)
        ax.set_title(label)
    axes[1].tick_params(axis='y',labelleft=False)
    fig.supxlabel('PairProto − control (95% recipe-cluster CI)',fontsize=11)
    fig.tight_layout(w_pad=2);save(fig,'pairproto_uncertainty')


def robustness(report):
    fig,axes=plt.subplots(1,2,figsize=(9,3.5))
    for ax,(source,resource) in zip(axes,report['sources'].items()):
        partitions=[resource['main']]+list(resource['replications'].values())
        for name,label,color in [('mean','Mean-ComplEx',BLUE),('herb_prototype','Herb prototype',ORANGE),('pairproto','PairProto-KGE',PURPLE)]:
            xs=[];ys=[];errors=[]
            for i,row in enumerate(partitions):
                m=row['models'].get(name)
                if not m:continue
                xs.append(i);ys.append(m['summary']['mrr']['mean']);errors.append(m['summary']['mrr']['sample_std'] or 0)
            ax.errorbar(xs,ys,yerr=errors,fmt='o-',capsize=3,lw=1.2,color=color,label=label)
        ax.set_xticks(range(len(partitions)),['Main']+[f'Replication {i}' for i in range(1,len(partitions))]);ax.set_title(SOURCES[source]);ax.set_ylabel('MRR (mean ± seed SD)')
        ax.grid(axis='y',alpha=.15);ax.legend(frameon=False)
    fig.tight_layout();save(fig,'pairproto_robustness')


def sensitivity(report):
    fig,axes=plt.subplots(2,2,figsize=(9,6));any_data=False
    for column,(source,resource) in enumerate(report['sources'].items()):
        lr=resource['main']['mean_lr'];split=PLAN['datasets'][source]['main']
        path=OUT/(prototype_tag(source,split,PLAN['main_seeds'][0],lr)+'_sensitivity.json')
        if not path.exists():
            for ax in axes[:,column]:pending(ax)
            continue
        records=json.loads(path.read_text(encoding='utf-8'))['records'];any_data=True
        standard={'ridge_relative':.01,'shrinkage':20,'minimum_pair_support':3}
        base=next(r for r in records if r['config']==standard)
        ax=axes[0,column];weights=sorted(map(float,base['dev_weights']))
        ax.plot(weights,[base['dev_weights'][str(w)]['mrr'] for w in weights],'o-',color=PURPLE)
        gamma=PLAN['prototype'][source+'_pair_weight'];ax.axvline(gamma,color=GRAY,ls='--',lw=1)
        ax.set_xlabel('Pair mixture weight γ');ax.set_ylabel('Development MRR');ax.set_title(SOURCES[source]+' / fixed backbone')
        ax=axes[1,column]
        for key,color in [('minimum_pair_support',BLUE),('shrinkage',ORANGE),('ridge_relative',PURPLE)]:
            rows=[r for r in records if all(r['config'][k]==standard[k] for k in standard if k!=key)]
            rows.sort(key=lambda r:r['config'][key]);x=np.arange(len(rows))
            y=[r['dev_weights'][str(float(gamma))]['mrr'] for r in rows]
            # Separate categorical points make unlike hyperparameter scales explicit.
            offset={'minimum_pair_support':0,'shrinkage':4,'ridge_relative':8}[key]
            ax.plot(x+offset,y,'o-',color=color,label=key.replace('_',' '))
            for xx,yy,r in zip(x+offset,y,rows):ax.annotate(str(r['config'][key]),(xx,yy),xytext=(0,6),textcoords='offset points',ha='center',fontsize=7)
        ax.set_xticks([]);ax.set_ylabel('Development MRR');ax.set_xlabel('One-factor controls (values annotated)');ax.legend(frameon=False,fontsize=7)
    if any_data:fig.tight_layout();save(fig,'pairproto_dev_sensitivity')
    else:plt.close(fig)


def engineering():
    path=OUT/'pairproto_paper_cases_costs.json'
    if not path.exists():return
    record=json.loads(path.read_text(encoding='utf-8'));sources=list(record['sources'])
    fig,grid=plt.subplots(2,2,figsize=(9,6));axes=grid.ravel();x=np.arange(len(sources))
    for off,key,label,color in [(-.25,'base_parameter_bytes','Base parameters',BLUE),(0,'pair_prototype_bytes','Pair prototype array',PURPLE),(.25,'sparse_count_bytes','Sparse count arrays',ORANGE)]:
        axes[0].bar(x+off,[record['sources'][s][key]/2**20 for s in sources],width=.22,label=label,color=color)
    axes[0].set_ylabel('Payload (MiB)');axes[0].set_xticks(x,[SOURCES[s] for s in sources]);axes[0].legend(frameon=False,fontsize=9)
    axes[1].bar(x,[record['sources'][s]['prototype_build_seconds'] for s in sources],width=.5,color=PURPLE)
    axes[1].set_xticks(x,[SOURCES[s] for s in sources]);axes[1].set_ylabel('Prototype construction time (s)')
    for i,(key,label,color) in enumerate([('mean','Mean',BLUE),('prototype_cold','Prototype cold',ORANGE),('prototype_warm','Prototype cached',PURPLE),('raw_pair_cold','Raw pair scores',GRAY)]):
        axes[2].bar(x+(i-1.5)*.18,[record['sources'][s]['latency'][key]['median_ms'] for s in sources],width=.17,label=label,color=color)
    axes[2].set_xticks(x,[SOURCES[s] for s in sources]);axes[2].set_ylabel('Median query latency (ms)');axes[2].legend(frameon=False,fontsize=9)
    for i,(key,label,color) in enumerate([('mean','Mean',BLUE),('pairproto','PairProto',PURPLE)]):
        axes[3].bar(x+(i-.5)*.24,[record['sources'][s]['all_entity_boundary'][key]['mrr'] for s in sources],width=.23,label=label,color=color)
    axes[3].set_xticks(x,[SOURCES[s] for s in sources]);axes[3].set_ylabel('All-entity MRR (one seed)');axes[3].legend(frameon=False,fontsize=9)
    for ax,title in zip(axes,['(a) Array payloads','(b) Offline construction','(c) Catalogue scoring','(d) Unrestricted boundary']):ax.set_title(title,loc='left',fontsize=10)
    for ax in axes:ax.grid(axis='y',alpha=.13);ax.set_axisbelow(True)
    fig.tight_layout();save(fig,'pairproto_cost_boundary')
    cases=[(s,c) for s in sources for c in record['sources'][s]['cases']]
    if not cases:return
    rows=(len(cases)+1)//2
    fig,axes=plt.subplots(rows,2,figsize=(9,3.3*rows),squeeze=False)
    for ax,(source,case) in zip(axes.flat,cases):
        entries=case['top_pairs'][:5];labels=['Herb mean']+[' + '.join(p['herbs']) for p in entries]+['Other pairs']
        # Stable opaque IDs avoid untranslated or guessed herb names.
        labels=['Herb mean']+[f"Pair {p['herb_ids'][0]}–{p['herb_ids'][1]}" for p in entries]+['Other pairs']
        values=[case['mean_term']]+[p['contribution'] for p in entries]+[case['pair_term_total']-sum(p['contribution'] for p in entries)]
        ax.barh(range(len(values)),values,color=[BLUE]+[PURPLE if v>=0 else ORANGE for v in values[1:]])
        ax.set_yticks(range(len(values)),labels);ax.invert_yaxis();ax.axvline(0,lw=.8,color=GRAY)
        ax.set_title(SOURCES[source]+' / '+case['selection'].replace('_',' ')+'\n'+f"RR {case['mean_rr']:.3f} → {case['prototype_rr']:.3f}",fontsize=11)
        ax.set_xlabel('Signed score contribution')
    for ax in axes.ravel()[len(cases):]:ax.axis('off')
    fig.tight_layout(w_pad=2);save(fig,'pairproto_case_decomposition')


def strata():
    fig,axes=plt.subplots(2,3,figsize=(11,6));any_data=False
    specifications=[('coverage',[-1e-8,0,.25,.5,.75,1.00001],['0','(0,.25]','(.25,.5]','(.5,.75]','(.75,1]']),
        ('lengths',[0,5,10,15,np.inf],['1–5','6–10','11–15','>15']),
        ('nearest_jaccard',[-1e-8,.25,.5,.75,1.00001],['≤.25','(.25,.5]','(.5,.75]','>.75'])]
    for row,source in enumerate(PLAN['datasets']):
        path=OUT/f'pairproto_paper_{source.lower()}_strata.npz'
        if not path.exists():
            for ax in axes[row]:pending(ax)
            continue
        arrays=npz(path)
        _,pred=collect(source,PLAN['datasets'][source]['main'],PLAN['main_seeds'])
        seeds=sorted(set(pred.get('mean',{}))&set(pred.get('pairproto',{})))
        if not seeds:continue
        any_data=True
        for ax,(key,bins,labels) in zip(axes[row],specifications):
            for name,color,label in [('mean',BLUE,'Mean'),('pairproto',PURPLE,'PairProto')]:
                ys=[]
                for lo,hi in zip(bins[:-1],bins[1:]):
                    mask=(arrays[key]>lo)&(arrays[key]<=hi)
                    ys.append(np.mean([pred[name][s]['rr'][mask].mean() for s in seeds]) if mask.any() else np.nan)
                ax.plot(range(len(labels)),ys,'o-',color=color,label=label)
            counts=[int(((arrays[key]>lo)&(arrays[key]<=hi)).sum()) for lo,hi in zip(bins[:-1],bins[1:])]
            ax.set_xticks(range(len(labels)),[f'{label}\nn={n}' for label,n in zip(labels,counts)])
            ax.set_title(SOURCES[source]+' / '+key.replace('_',' '),fontsize=10);ax.set_ylabel('MRR (seed average)');ax.grid(axis='y',alpha=.13)
            ax.legend(frameon=False,fontsize=7)
    if any_data:fig.tight_layout();save(fig,'pairproto_stratified_results')
    else:plt.close(fig)


def relation_results(report):
    translations={'传统医学疾病':'TCM disorder','传统医学证候':'TCM pattern',
                  '治则':'Treatment principle','治法':'Treatment method','病因':'Etiology','病机':'Pathogenesis'}
    fig,axes=plt.subplots(1,2,figsize=(10,3.5))
    for panel,(ax,(source,resource)) in enumerate(zip(axes,report['sources'].items())):
        models=resource['main']['models']
        baseline={row['relation']:row for row in models.get('mean',{}).get('relation_results',[])}
        labels=[];points=[];deviations=[]
        for row in models.get('pairproto',{}).get('relation_results',[]):
            reference=baseline.get(row['relation'])
            if not reference:continue
            common=sorted(set(row['per_seed'])&set(reference['per_seed']))
            if not common:continue
            values=[row['per_seed'][seed]['mrr']-reference['per_seed'][seed]['mrr'] for seed in common]
            label=row['relation'].split('::')[-1];label=translations.get(label,label)
            if not label.isascii():raise ValueError('Missing English relation label: '+label)
            labels.append(f"{label} (q={row['queries']})")
            points.append(np.mean(values));deviations.append(np.std(values,ddof=1) if len(values)>1 else 0.)
        if points:
            ax.errorbar(points,range(len(points)),xerr=deviations,fmt='o',capsize=3,color=PURPLE)
            ax.set_yticks(range(len(points)),labels);ax.invert_yaxis();ax.axvline(0,color=GRAY,ls='--',lw=1)
            ax.grid(axis='x',alpha=.13)
        else:pending(ax)
        ax.set_title(f"({chr(97+panel)}) {SOURCES[source]}")
        ax.set_xlabel('PairProto − Mean: relation MRR\n(mean ± paired-seed SD)')
    fig.tight_layout(w_pad=2);save(fig,'pairproto_relation_results')


def main():
    report=build_report()
    main_comparison(report);mechanism(report);uncertainty(report);robustness(report)
    sensitivity(report);engineering();strata();relation_results(report)
    framework=FIG/'pairproto_framework.pdf'
    if framework.exists() and PAPER.exists():shutil.copy2(framework,PAPER/framework.name)
    print('EVIDENCE FIGURES UPDATED',flush=True)


if __name__=='__main__':main()
