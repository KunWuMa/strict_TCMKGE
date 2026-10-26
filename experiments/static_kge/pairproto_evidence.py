"""Pure experiment aggregation; contains no manuscript text or dataset payload."""
from pathlib import Path
import json
import numpy as np
from threadpoolctl import threadpool_limits
from pair_memory_dev import ingredients
from pair_prototype_full_suite import neural_tag, bootstrap, save
from pairproto_excel_source import source_root, read_arrays, load_dataset, input_provenance
OUT = source_root()
HERE=Path(__file__).resolve().parent
from pairproto_paper_plan import load_plan
PLAN=load_plan()

NAMES={"mean":"Mean-ComplEx","neural_mean":"Mean-ComplEx","herb_prototype":"Herb prototype","pairproto":"PairProto-KGE",
       "neural_pair":"Uniform pair-moment composer","neural_matched":"Dual-mean control",
       "neural_second_moment":"Second-moment control","neural_deepsets":"Elementwise residual control",
       "neural_full_deepsets":"Deep Sets",
       "herb_prototype_dev_selected":"Herb prototype (dev-selected)",
       "neural_set_transformer":"Set Transformer adapter","neural_rotate":"Mean-RotatE","neural_pairre":"Mean-PairRE",
       "neural_transe":"ID-TransE","neural_distmult":"ID-DistMult","neural_complex":"ID-ComplEx",
       "raw_pair_dev_selected":"Raw pair scores (dev-selected)","raw_herb_dev_selected":"Raw herb scores (dev-selected)",
       "raw_pair_only":"Raw pair scores only","raw_herb_only":"Raw herb scores only","raw_fixed_weight":"Raw pair scores (same weight)",
       "pair_identity_shuffle":"Pair-identity shuffle","relation_identity_shuffle":"Relation-identity shuffle",
       "shared_pair_prototypes":"Shared pair prototypes","posterior_smoothing_only":"Posterior smoothing only",
       "no_shrinkage":"No support shrinkage","support_one":"Minimum support one",
       "frequency_prior":"Training target frequency"}

NAMES.update({'baseline_mean_transe':'Mean-TransE','baseline_mean_distmult':'Mean-DistMult',
 'baseline_nodepiece':'NodePiece (OOS adapter)','baseline_ingram':'InGram (graph adapter)',
 'baseline_ultra':'ULTRA (from scratch)','baseline_trix':'TRIX (from scratch)'})

SOURCES={"PRIMARY":"TCM-MKG","KTKP":"KTKP"}


def npz(path):
    return read_arrays(path)


def add_prediction(store,name,seed,rr,extra=None):
    rr=np.asarray(rr)
    if rr.ndim==1:rr=rr[None,:]
    values={"rr":rr.mean(0),"mr":(1./rr).mean(0),"hits1":(rr==1).mean(0),
            "hits3":(rr>=1/3).mean(0),"hits5":(rr>=.2).mean(0),"hits10":(rr>=.1).mean(0)}
    store.setdefault(name,{})[seed]={**values,"metadata":extra or {}}


def config_for(source,name,default=False):
    full_path=OUT/f'pairproto_full_deepsets_{source.lower()}_selected_config.json'
    if name=='full_deepsets' and not default and full_path.exists():
        return json.loads(full_path.read_text(encoding='utf-8'))['chosen']
    path=OUT/f"pairproto_paper_{source.lower()}_selected_configs.json"
    if not default and path.exists():
        report=json.loads(path.read_text(encoding="utf-8"))
        if name in report["models"]:return report["models"][name]["chosen"]
    return {"lr":.003,"heads":4}


def prototype_tag(source,split,seed,lr):
    result=f"prototype_suite_{source.lower()}_{split}_prototype_seed{seed}"
    return result+(f"_backbone_lr{lr:g}" if lr!=.003 else "")


def collect(source,split,seeds,default=False,mechanisms=False):
    data=load_dataset(source,split);pred={};builds=[]
    mean_lr=config_for(source,"mean",default)["lr"]
    for seed in seeds:
        tag=prototype_tag(source,split,seed,mean_lr);path=OUT/(tag+"_rr.npz")
        record=OUT/(tag+'.json')
        if path.exists() and record.exists():
            z=npz(path)
            if not np.array_equal(z["heads"],data.test[:,0]) or not np.array_equal(z['relations'],data.test[:,1]):raise AssertionError("Prototype query order changed")
            for name,key in (("mean","mean"),("herb_prototype","herb_prototype"),("pairproto","pair_prototype")):
                add_prediction(pred,name,seed,z[key])
            if 'frequency_prior' in z:add_prediction(pred,'frequency_prior',seed,z['frequency_prior'])
            record=OUT/(tag+".json")
            if record.exists():builds.append(json.loads(record.read_text(encoding="utf-8")))
        for name in PLAN["core_models"]+PLAN["ancillary_models"]+['full_deepsets']:
            cfg=config_for(source,name,default)
            tag=neural_tag(source,split,name,seed,cfg["lr"],cfg["heads"])
            path=OUT/(tag+"_rr.npz")
            metadata_path=OUT/(tag+'_test.json')
            if path.exists() and metadata_path.exists():
                z=npz(path)
                if not np.array_equal(z["heads"],data.test[:,0]) or not np.array_equal(z['relations'],data.test[:,1]):raise AssertionError("Neural query order changed")
                metadata=json.loads(metadata_path.read_text(encoding="utf-8"))
                add_prediction(pred,"neural_"+name,seed,z["rr"],metadata)
        if not default and str(split)==str(PLAN['datasets'][source]['main']):
            manifest_path=OUT/'pairproto_extended_baselines_selection.json'
            if manifest_path.exists():
                extension=json.loads(manifest_path.read_text(encoding='utf-8'))
                for extra_name in extension['models'].get(source,{}):
                    extra_stem=f'extended_baseline_{source.lower()}_{split}_{extra_name}_seed{seed}'
                    extra_path=OUT/(extra_stem+'_rr.npz')
                    extra_record=OUT/(extra_stem+'_test.json')
                    if extra_path.exists() and extra_record.exists():
                        z=npz(extra_path)
                        if not np.array_equal(z['heads'],data.test[:,0]) or not np.array_equal(z['relations'],data.test[:,1]):
                            raise AssertionError('Extra baseline query order changed')
                        extra_info=json.loads(extra_record.read_text(encoding='utf-8'))
                        if extra_info['protocol_sha256']!=extension['protocol_sha256']:
                            raise AssertionError('Extra baseline protocol changed')
                        add_prediction(pred,'baseline_'+extra_name,seed,z['rr'],extra_info)
        if mechanisms:
            stem=OUT/f'pairproto_herb_control_{source.lower()}_seed{seed}'
            path=stem.with_name(stem.name+'_rr.npz');record=stem.with_suffix('.json')
            if path.exists() and record.exists():
                z=npz(path)
                if not np.array_equal(z['heads'],data.test[:,0]) or not np.array_equal(z['relations'],data.test[:,1]):raise AssertionError('Herb control query order changed')
                add_prediction(pred,'herb_prototype_dev_selected',seed,z['rr'],json.loads(record.read_text(encoding='utf-8')))
        if mechanisms:
            path=OUT/f"pairproto_paper_{source.lower()}_mechanisms_seed{seed}_rr.npz"
            complete_record=OUT/f"pairproto_paper_{source.lower()}_mechanisms_seed{seed}.json"
            if path.exists() and complete_record.exists():
                z=npz(path)
                if not np.array_equal(z["heads"],data.test[:,0]) or not np.array_equal(z['relations'],data.test[:,1]):raise AssertionError("Mechanism query order changed")
                for name in ("raw_pair_dev_selected","raw_herb_dev_selected","raw_pair_only","raw_herb_only","raw_fixed_weight",
                             "shared_pair_prototypes","posterior_smoothing_only","no_shrinkage","support_one"):
                    add_prediction(pred,name,seed,z[name])
                for name in ("pair_identity_shuffle","relation_identity_shuffle"):
                    arrays=[z[f"{name}_{s}"] for s in PLAN["shuffle_seeds"]]
                    add_prediction(pred,name,seed,np.stack(arrays),{"permutations":len(arrays)})
                if seed in pred.get("pairproto",{}) and not np.allclose(pred["pairproto"][seed]["rr"],z["full"],atol=1e-7):
                    raise AssertionError("Mechanism full model differs from saved main model")
    for seed in set(pred.get('mean',{}))&set(pred.get('neural_mean',{})):
        if not np.allclose(pred['mean'][seed]['rr'],pred['neural_mean'][seed]['rr'],atol=1e-8):
            raise AssertionError('Mean API ranks differ from the shared prototype-cache Mean ranks')
    models={}
    for name,values in pred.items():
        order=sorted(values)
        summary={}
        for key,score in (("mr","mr"),("mrr","rr"),("hits1","hits1"),("hits3","hits3"),("hits5","hits5"),("hits10","hits10")):
            numbers=[float(values[s][score].mean()) for s in order]
            summary[key]={"mean":float(np.mean(numbers)),"sample_std":float(np.std(numbers,ddof=1)) if len(numbers)>1 else None}
        macro=[np.mean([values[s]["rr"][data.test[:,1]==r].mean() for r in np.unique(data.test[:,1])]) for s in order]
        per_seed={str(s):{key:float(values[s][score].mean()) for key,score in (("mr","mr"),("mrr","rr"),("hits1","hits1"),("hits3","hits3"),("hits5","hits5"),("hits10","hits10"))} for s in order}
        models[name]={"seeds":order,"runs":len(order),"summary":summary,"per_seed":per_seed,"relation_macro_mrr":{"mean":float(np.mean(macro)),
                                   "sample_std":float(np.std(macro,ddof=1)) if len(macro)>1 else None}}
        relations=[]
        for r in np.unique(data.test[:,1]):
            mask=data.test[:,1]==r
            records={str(seed):{key:float(values[seed][score][mask].mean())
                     for key,score in (("mr","mr"),("mrr","rr"),("hits1","hits1"),("hits3","hits3"),("hits5","hits5"),("hits10","hits10"))}
                     for seed in order}
            relations.append({'relation':str(data.relations[int(r)]),'queries':int(mask.sum()),
                              'per_seed':records})
        models[name]['relation_results']=relations
    group_ids={};recipe=[]
    if hasattr(data, 'recipe_group_ids'):
        recipe=data.recipe_group_ids
        group_ids={int(group):int(group) for group in np.unique(recipe)}
    else:
        for h in data.test[:,0]:
            key=ingredients(data,int(h));recipe.append(group_ids.setdefault(key,len(group_ids)))
        recipe=np.asarray(recipe)
    comparisons={}
    for name,values in pred.items():
        if name=="pairproto":continue
        common=sorted(set(values)&set(pred.get("pairproto",{})))
        if not common:continue
        rr=np.mean([pred["pairproto"][s]["rr"]-values[s]["rr"] for s in common],axis=0)
        hit=np.mean([pred["pairproto"][s]["hits1"]-values[s]["hits1"] for s in common],axis=0)
        comparisons[name]={"seeds":common,"mrr":bootstrap(rr,data.test[:,0]),"hits1":bootstrap(hit,data.test[:,0]),
                           "recipe_mrr":bootstrap(rr,recipe,"recipe"),"recipe_hits1":bootstrap(hit,recipe,"recipe")}
    report={"source":source,"split":split,"mean_lr":mean_lr,"audit":data.audit,"models":models,"comparisons":comparisons,
            "test_queries":len(data.test),"test_formula_ids":len(np.unique(data.test[:,0])),"test_recipe_groups":len(group_ids),
            "candidate_ranges":{data.relations[int(r)]:len(data.range_candidates[int(r)]) for r in np.unique(data.test[:,1])},
            "builds":builds}
    return report,pred


def complete_check(report):
    missing=[]
    for source,spec in PLAN["datasets"].items():
        row=report["sources"][source]["main"]
        if not (OUT/f"pairproto_paper_{source.lower()}_selected_configs.json").exists():missing.append(source+": dev selection")
        if not (OUT/f'pairproto_full_deepsets_{source.lower()}_selected_config.json').exists():missing.append(source+': full Deep Sets selection')
        if row['models'].get('neural_full_deepsets',{}).get('runs',0)!=len(PLAN['ancillary_seeds']):missing.append(source+': full Deep Sets five seeds')
        if row['models'].get('herb_prototype_dev_selected',{}).get('runs',0)!=5:missing.append(source+': calibrated herb control five seeds')
        if not (OUT/f'pairproto_herb_control_{source.lower()}_selected.json').exists():missing.append(source+': calibrated herb control dev lock')
        for name in ("mean","herb_prototype","pairproto"):
            if row["models"].get(name,{}).get("runs",0)!=5:missing.append(source+": "+name)
        for name in PLAN["core_models"]+PLAN["ancillary_models"]:
            expected=len(PLAN['main_seeds']) if name in PLAN["core_models"] else len(PLAN['ancillary_seeds'])
            if row["models"].get("neural_"+name,{}).get("runs",0)!=expected:missing.append(source+": "+name)
        for name in ("raw_pair_dev_selected","pair_identity_shuffle","relation_identity_shuffle","shared_pair_prototypes",
                     "posterior_smoothing_only","no_shrinkage","support_one"):
            if row["models"].get(name,{}).get("runs",0)!=5:missing.append(source+": "+name)
        for split,rep in report["sources"][source]["replications"].items():
            for name in ["pairproto"]+["neural_"+m for m in PLAN["replication_models"]]:
                if rep["models"].get(name,{}).get("runs",0)!=len(PLAN['ancillary_seeds']):missing.append(source+"/"+split+": "+name)
        for name in PLAN['core_models']+PLAN['ancillary_models']+['full_deepsets']:
            if report['sources'][source]['fixed']['models'].get('neural_'+name,{}).get('runs',0)!=len(PLAN['main_seeds']):
                missing.append(source+': fixed '+name+' five seeds')
        lr=row["mean_lr"];stem=prototype_tag(source,spec["main"],20260915,lr)
        if not (OUT/(stem+"_sensitivity.json")).exists():missing.append(source+": sensitivity")
    for filename in ("pairproto_paper_cases_costs.json","pairproto_moment_collision.json"):
        if not (OUT/filename).exists():missing.append(filename)
    costpath=OUT/'pairproto_paper_cases_costs.json'
    if costpath.exists():
        costs=json.loads(costpath.read_text(encoding='utf-8'))
        if not costs.get('complete'):missing.append('Cases/costs: incomplete source matrix')
        for source in PLAN['datasets']:
            for kind in ('boundary_rr','strata'):
                if not (OUT/f'pairproto_paper_{source.lower()}_{kind}.npz').exists():missing.append(source+': '+kind)
    extension_path=OUT/'pairproto_extended_baselines_selection.json'
    if extension_path.exists():
        extension=json.loads(extension_path.read_text(encoding='utf-8'))
        for source,methods in extension['models'].items():
            for name in methods:
                count=report['sources'][source]['main']['models'].get('baseline_'+name,{}).get('runs',0)
                if count!=len(extension['evaluation_seeds']):
                    missing.append(source+': extended '+name+' five seeds')
    return missing


def build_report():
    report={"protocol":PLAN,"sources":{},"input_provenance":input_provenance()}
    amendment=HERE/'pairproto_full_deepsets_amendment.json'
    if amendment.exists():report['architecture_fidelity_amendment']=json.loads(amendment.read_text(encoding='utf-8'))
    herb_amendment=HERE/'pairproto_herb_control_amendment.json'
    if herb_amendment.exists():report['herb_control_amendment']=json.loads(herb_amendment.read_text(encoding='utf-8'))
    with threadpool_limits(limits=4):
        for source,spec in PLAN["datasets"].items():
            main,_=collect(source,spec["main"],PLAN["main_seeds"],mechanisms=True)
            fixed,_=collect(source,spec["main"],PLAN["main_seeds"],default=True)
            rep={split:collect(source,split,PLAN["ancillary_seeds"])[0] for split in spec["replications"]}
            report["sources"][source]={"main":main,"fixed":fixed,"replications":rep}
    extension_path=OUT/'pairproto_extended_baselines_selection.json'
    if extension_path.exists():
        report['extended_baseline_amendment']=json.loads(extension_path.read_text(encoding='utf-8'))
    report["missing"]=complete_check(report)
    report["complete"]=not report["missing"]
    return report
