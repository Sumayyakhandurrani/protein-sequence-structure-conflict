#!/usr/bin/env python3
"""Post-hoc structure-only reliability and error-subtype audit.
Run from the protein_conflict repository root. Never overwrites earlier experiments.
"""
from pathlib import Path
import hashlib, json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

CAL = Path('data/conflict/go_frequency_v1_inputs')
SEQ = Path('data/final_evaluation/sequence_expert_v1_test')
STRUCT = Path('data/final_evaluation/structure_expert_v1_test')
SPLIT = Path('data/splits/mmseqs30_graph_v2/labels')
RAW = Path('data/final_evaluation/posthoc_raw_probability_baseline_v1/test_reliability_predictions.npz')
MEFILE = Path('data/final_evaluation/posthoc_sequence_only_probability_v1/test_reliability_predictions.npz')
OUT = Path('data/final_evaluation/posthoc_structure_only_subtypes_v1')
N=494
SEED=20261023
B=1000

def ids(path): return [s.strip() for s in path.read_text().splitlines() if s.strip()]
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def ap(y,p):
    if np.unique(y).size<2:return float('nan')
    return float(average_precision_score(y,p))

def main():
    ps_cal=np.load(CAL/'p_seq_cal.npy').astype(np.float64)
    pt_cal=np.load(CAL/'p_struct_cal.npy').astype(np.float64)
    y_cal=np.load(CAL/'y_cal.npy').astype(np.float64)
    assert ps_cal.shape==pt_cal.shape==y_cal.shape==(2369,N)
    test_ids=ids(SPLIT/'test_uniprot_ids.txt')
    assert ids(SEQ/'test_uniprot_ids.txt')==test_ids and len(test_ids)==4754
    struct_ids=ids(STRUCT/'test_structure_eligible_ids.txt')
    assert len(struct_ids)==4731 and len(set(struct_ids))==4731
    lookup={v:i for i,v in enumerate(test_ids)}
    idx=np.array([lookup[v] for v in struct_ids])
    ps_all=np.load(SEQ/'test_calibrated_probabilities.npy').astype(np.float64)
    pt=np.load(STRUCT/'test_probabilities.npy').astype(np.float64)
    y_all=np.load(SPLIT/'test_labels.npy').astype(np.float64)
    assert ps_all.shape==y_all.shape==(4754,N) and pt.shape==(4731,N)
    ps,y=ps_all[idx],y_all[idx]
    for arr in [ps_cal,pt_cal,y_cal,ps,pt,y]:assert np.isfinite(arr).all()
    for arr in [y_cal,y]:assert np.isin(arr,[0,1]).all()
    target_cal=(np.abs(pt_cal-y_cal)>=0.5).astype(np.uint8).ravel()
    failure=(np.abs(pt-y)>=0.5)
    fp=(pt>=0.5)&(y==0)
    fn=(pt<0.5)&(y==1)
    assert np.array_equal(failure,fp|fn) and not np.any(fp&fn)
    model=LogisticRegression(max_iter=2000,solver='lbfgs')
    model.fit(pt_cal.reshape(-1,1),target_cal)
    mf=model.predict_proba(pt.reshape(-1,1))[:,1].reshape(4731,N)
    with np.load(RAW,allow_pickle=False) as z:
        preds={k:z[k].copy() for k in ['MA','MB','MC','MD']}
    with np.load(MEFILE,allow_pickle=False) as z:
        print('ME NPZ keys:',z.files)
        key='ME' if 'ME' in z.files else None
        if key is None:raise KeyError('ME key missing from saved predictions; inspect the NPZ')
        preds['ME']=z[key].copy()
    preds['MF']=mf
    for name,p in preds.items():
        assert p.shape==(4731,N) and np.isfinite(p).all(),name
        assert np.all((p>=0)&(p<=1)),name
    # Verify published scores before proceeding.
    expected={'MA':0.33168476214437315,'MB':0.4432987897763585,
              'MC':0.4024362713139442,'MD':0.4452442784481795,
              'ME':0.36322475861156106}
    for name,val in expected.items():
        actual=ap(failure.ravel(),preds[name].ravel())
        if abs(actual-val)>1e-9:raise RuntimeError(f'{name} AP mismatch: {actual} vs {val}')
    metrics={}
    for name,p in preds.items():
        metrics[name]={'overall_ap':ap(failure.ravel(),p.ravel()),
                       'overall_roc_auc':float(roc_auc_score(failure.ravel(),p.ravel())),
                       'fp_type_ap_all_pairs':ap(fp.ravel(),p.ravel()),
                       'missed_positive_type_ap_all_pairs':ap(fn.ravel(),p.ravel())}
    # Each subtype is a one-vs-rest endpoint over all pairs; other failure type is negative.
    pairs=[('MC','ME'),('MC','MF'),('MB','ME'),('MD','ME'),('MD','MC'),('ME','MF')]
    point={f'{a}_minus_{b}':metrics[a]['overall_ap']-metrics[b]['overall_ap'] for a,b in pairs}
    rng=np.random.default_rng(SEED)
    boot={f'{a}_minus_{b}':[] for a,b in pairs}
    for i in range(B):
        rows=rng.integers(0,len(struct_ids),len(struct_ids))
        target=failure[rows].ravel()
        vals={name:ap(target,p[rows].ravel()) for name,p in preds.items()}
        for a,b in pairs:boot[f'{a}_minus_{b}'].append(vals[a]-vals[b])
        if (i+1)%200==0:print('Bootstrap',i+1,'/',B,flush=True)
    ci={k:{'delta_ap':point[k], 'ci95_percentile':[float(x) for x in np.percentile(v,[2.5,97.5])],
           'n_resamples':B} for k,v in boot.items()}
    payload={'classification':'post_hoc_exploratory','cal_proteins':2369,
             'test_proteins':4731,'labels':N,'pairs':4731*N,
             'failure_prevalence':float(failure.mean()),
             'counts':{'structure_fp_type':int(fp.sum()),'structure_missed_positive_type':int(fn.sum()),
                       'total_failures':int(failure.sum())},
             'subtype_definition':'one-vs-rest over all protein-label pairs; the other failure type counts as negative',
             'MF_model':'LogisticRegression(max_iter=2000,solver=lbfgs,sklearn defaults)',
             'MF_coefficient':float(model.coef_[0,0]),'MF_intercept':float(model.intercept_[0]),
             'MF_iterations':int(model.n_iter_[0]),'metrics':metrics,'bootstrap':ci,
             'bootstrap_note':'1000 paired whole-protein resamples, frozen scores, no model refitting; exploratory repeated TEST use',
             'input_sha256':{'MA_MD_npz':sha(RAW),'ME_npz':sha(MEFILE)}}
    OUT.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(OUT/'test_reliability_predictions.npz',MF=mf)
    (OUT/'results.json').write_text(json.dumps(payload,indent=2)+'\n')
    print('\nFAILURES:',payload['counts'],'prevalence:',payload['failure_prevalence'])
    for name,v in metrics.items():print(name, 'AP',round(v['overall_ap'],9),'FP-type AP',round(v['fp_type_ap_all_pairs'],9),'missed-positive AP',round(v['missed_positive_type_ap_all_pairs'],9))
    for name,v in ci.items():print(name,round(v['delta_ap'],9),'95% CI',v['ci95_percentile'])
    print('Saved:',OUT/'results.json')
if __name__=='__main__':main()
