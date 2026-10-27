from pathlib import Path
import hashlib
import json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

CAL = Path('data/conflict/go_frequency_v1_inputs')
SEQ = Path('data/final_evaluation/sequence_expert_v1_test')
STRUCT = Path('data/final_evaluation/structure_expert_v1_test')
SPLIT = Path('data/splits/mmseqs30_graph_v2/labels')
EXISTING = Path('data/final_evaluation/posthoc_raw_probability_baseline_v1')
OUT = Path('data/final_evaluation/posthoc_sequence_only_probability_v1')
N_LABELS = 494

def ids(path):
    return [x.strip() for x in path.read_text().splitlines() if x.strip()]

def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

ps_cal = np.load(CAL / 'p_seq_cal.npy').astype(np.float64)
pt_cal = np.load(CAL / 'p_struct_cal.npy').astype(np.float64)
y_cal = np.load(CAL / 'y_cal.npy').astype(np.float64)
assert ps_cal.shape == pt_cal.shape == y_cal.shape == (2369, N_LABELS)

test_ids = ids(SPLIT / 'test_uniprot_ids.txt')
seq_ids = ids(SEQ / 'test_uniprot_ids.txt')
struct_ids = ids(STRUCT / 'test_structure_eligible_ids.txt')
assert len(test_ids) == 4754 and seq_ids == test_ids
assert len(struct_ids) == len(set(struct_ids)) == 4731
mapping = {uid: i for i, uid in enumerate(test_ids)}
assert all(uid in mapping for uid in struct_ids)
rows = np.array([mapping[uid] for uid in struct_ids])
ps_all = np.load(SEQ / 'test_calibrated_probabilities.npy').astype(np.float64)
pt = np.load(STRUCT / 'test_probabilities.npy').astype(np.float64)
y_all = np.load(SPLIT / 'test_labels.npy').astype(np.float64)
assert ps_all.shape == y_all.shape == (4754, N_LABELS)
assert pt.shape == (4731, N_LABELS)
ps, y = ps_all[rows], y_all[rows]
for a in (ps_cal, pt_cal, y_cal, ps, pt, y):
    assert np.isfinite(a).all()
for a in (y_cal, y):
    assert np.all((a == 0) | (a == 1))
for a in (ps_cal, ps, pt_cal, pt):
    assert np.all((a >= 0) & (a <= 1))

target_cal = (np.abs(pt_cal - y_cal) >= 0.5).astype(np.uint8).ravel()
target_test_matrix = (np.abs(pt - y) >= 0.5).astype(np.uint8)
target_test = target_test_matrix.ravel()

# ME: p_seq alone, exactly one logistic feature; default regularization matches MA-MD.
model = LogisticRegression(max_iter=2000, solver='lbfgs')
model.fit(ps_cal.reshape(-1, 1), target_cal)
me = model.predict_proba(ps.reshape(-1, 1))[:, 1].reshape(4731, N_LABELS)
assert model.n_iter_[0] < 2000, 'ME logistic regression did not converge'

with np.load(EXISTING / 'test_reliability_predictions.npz', allow_pickle=False) as old:
    assert all(old[k].shape == me.shape for k in ('MA','MB','MC','MD'))
    preds = {k: old[k].copy() for k in ('MA','MB','MC','MD')}
preds['ME'] = me

metrics = {}
for name, pred in preds.items():
    scores = pred.ravel()
    metrics[name] = {
        'ap': float(average_precision_score(target_test, scores)),
        'roc_auc': float(roc_auc_score(target_test, scores)),
    }

# Error subtype descriptive analysis; not a separately trained/tuned classifier.
fp = (pt >= 0.5) & (y == 0)
fn = (pt < 0.5) & (y == 1)
assert np.array_equal(target_test_matrix.astype(bool), fp | fn)
subtypes = {
    'structure_false_positive_type': int(fp.sum()),
    'structure_missed_positive_type': int(fn.sum()),
    'total_failures': int(target_test.sum()),
    'failure_prevalence': float(target_test.mean()),
    'positive_annotation_prevalence': float(y.mean()),
}

OUT.mkdir(parents=True, exist_ok=True)
np.savez_compressed(OUT / 'test_reliability_predictions.npz', **preds)
payload = {
    'analysis_classification': 'post_hoc_exploratory',
    'test_used_for_model_fitting': False,
    'cal_proteins': 2369,
    'paired_test_proteins': 4731,
    'labels': N_LABELS,
    'endpoint': 'abs(p_structure-y)>=0.5',
    'ME_features': ['p_sequence'],
    'ME_model': 'LogisticRegression(max_iter=2000, solver=lbfgs, sklearn defaults)',
    'ME_coef': float(model.coef_[0, 0]),
    'ME_intercept': float(model.intercept_[0]),
    'ME_iterations': int(model.n_iter_[0]),
    'metrics': metrics,
    'subtypes': subtypes,
    'comparisons_ap': {
        'MB_minus_ME': metrics['MB']['ap'] - metrics['ME']['ap'],
        'MC_minus_ME': metrics['MC']['ap'] - metrics['ME']['ap'],
        'MD_minus_ME': metrics['MD']['ap'] - metrics['ME']['ap'],
    },
    'existing_prediction_file_sha256': sha256(EXISTING / 'test_reliability_predictions.npz'),
    'source_script_sha256': sha256(Path(__file__)),
}
(OUT / 'results.json').write_text(json.dumps(payload, indent=2) + '\n')
print(json.dumps(payload, indent=2))
print('\nSaved:', OUT / 'results.json')
print('Saved:', OUT / 'test_reliability_predictions.npz')
