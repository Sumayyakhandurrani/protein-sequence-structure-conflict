#!/usr/bin/env python3
"""Descriptive post-hoc TEST audit. No fitting, feature selection or bootstrap."""
from pathlib import Path
import hashlib, json
import numpy as np

SEQ=Path('data/final_evaluation/sequence_expert_v1_test')
STRUCT=Path('data/final_evaluation/structure_expert_v1_test')
SPLIT=Path('data/splits/mmseqs30_graph_v2/labels')
OUT=Path('data/final_evaluation/posthoc_failure_distributions_v1')
EPS=1e-7

def read_ids(p): return [s.strip() for s in p.read_text().splitlines() if s.strip()]
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()
def js(p,q):
    p=np.clip(p,EPS,1-EPS); q=np.clip(q,EPS,1-EPS)
    m=(p+q)/2
    return .5*(p*np.log(p/m)+(1-p)*np.log((1-p)/(1-m))) + .5*(q*np.log(q/m)+(1-q)*np.log((1-q)/(1-m)))
def stats(x):
    v=np.asarray(x,dtype=np.float64)
    return {'n':int(v.size),'mean':float(v.mean()),'median':float(np.median(v)),
            'q05':float(np.quantile(v,.05)),'q25':float(np.quantile(v,.25)),
            'q75':float(np.quantile(v,.75)),'q95':float(np.quantile(v,.95))}
def main():
    all_ids=read_ids(SPLIT/'test_uniprot_ids.txt')
    seq_ids=read_ids(SEQ/'test_uniprot_ids.txt')
    struct_ids=read_ids(STRUCT/'test_structure_eligible_ids.txt')
    assert len(all_ids)==4754 and seq_ids==all_ids
    assert len(struct_ids)==4731 and len(set(struct_ids))==4731
    idx={s:i for i,s in enumerate(all_ids)}
    rows=np.array([idx[s] for s in struct_ids])
    ps_path=SEQ/'test_calibrated_probabilities.npy'
    pt_path=STRUCT/'test_probabilities.npy'
    y_path=SPLIT/'test_labels.npy'
    ps_all=np.load(ps_path); pt=np.load(pt_path); y_all=np.load(y_path)
    assert ps_all.shape==y_all.shape==(4754,494) and pt.shape==(4731,494)
    ps=ps_all[rows].astype(np.float64); y=y_all[rows]; pt=pt.astype(np.float64)
    assert all(np.isfinite(a).all() for a in [ps,pt,y])
    assert np.isin(y,[0,1]).all()
    assert np.all((ps>=0)&(ps<=1)) and np.all((pt>=0)&(pt<=1))
    groups={'correct_negative':(y==0)&(pt<.5),
            'correct_positive':(y==1)&(pt>=.5),
            'false_positive_type':(y==0)&(pt>=.5),
            'missed_positive_type':(y==1)&(pt<.5)}
    assert sum(int(m.sum()) for m in groups.values())==4731*494
    assert int(groups['false_positive_type'].sum())==3899
    assert int(groups['missed_positive_type'].sum())==31207
    divergence=js(ps,pt)
    report={'analysis_classification':'post_hoc_descriptive','test_used_for_model_fitting':False,
            'n_test_proteins':4731,'n_labels':494,'n_pairs':4731*494,
            'definition':'Structure predicted positive when p_structure >= 0.5; four disjoint groups',
            'source_sha256':{str(p):sha(p) for p in [ps_path,pt_path,y_path]},'groups':{}}
    for name,mask in groups.items():
        report['groups'][name]={'count':int(mask.sum()),'fraction_all_pairs':float(mask.mean()),
           'p_sequence':stats(ps[mask]),'p_structure':stats(pt[mask]),
           'bernoulli_js_natural_log':stats(divergence[mask])}
    report['overall_failure_prevalence']=float((groups['false_positive_type']|groups['missed_positive_type']).mean())
    OUT.mkdir(parents=True,exist_ok=True)
    output=OUT/'results.json'
    if output.exists():raise FileExistsError(f'Will not overwrite existing audit: {output}')
    report['source_script_sha256']=sha(Path(__file__))
    output.write_text(json.dumps(report,indent=2)+'\n')
    print('Descriptive audit — no model fitting')
    for name,entry in report['groups'].items():
        print(f"{name}: n={entry['count']:,}, seq median={entry['p_sequence']['median']:.5f}, "
              f"struct median={entry['p_structure']['median']:.5f}, JS median={entry['bernoulli_js_natural_log']['median']:.6f}")
    print('Saved:',output)
if __name__=='__main__':main()
