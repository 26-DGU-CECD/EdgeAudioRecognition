import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

PILOT = Path(r'C:\Users\PLASS\Desktop\audio-policy-review-20261015T164106Z-1-001\audio-policy-review\policy-pilot')
PROJECT = Path(r'C:\Users\PLASS\Desktop\audio_experiments_DEF')
RESULT = PROJECT / 'policy_results' / 'S01_20261016_022757'
sys.path.insert(0, str(PILOT))
from project_io import ProjectFiles, load_aligned_predictions
from policy_core import MODEL_NAMES
from run_selection_pilot import measure
from selection_core import BACKGROUND_MODES, owned_candidates, subset, fit_thresholds


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


config = json.loads((RESULT / 'selection_config.json').read_text(encoding='utf-8-sig'))
reference = read_csv(PILOT / 'selection_reference_results' / 'selection_comparison.csv')
actual = read_csv(RESULT / 'selection_comparison.csv')
assert actual == reference
for item in config['prediction_provenance'].values():
    assert hashlib.sha256(Path(item['path']).read_bytes()).hexdigest() == item['sha256']
assert hashlib.sha256((PROJECT / 'metadata.csv').read_bytes()).hexdigest() == config['metadata_sha256']

files = ProjectFiles(PROJECT)
paths, y, logits = load_aligned_predictions(files, 'runs/E_predefined_specialists/nonexpert_070')
metadata = {r['path']: r for r in read_csv(PROJECT / 'metadata.csv') if r['split'] == 'val'}
assert all(files.class_names[int(target)] == metadata[path]['label'] for path, target in zip(paths, y))
groups = np.array([metadata[path]['group_id'] for path in paths])
cal = np.array([int(hashlib.sha256(f"{config['seed']}|{g}".encode()).hexdigest()[:16], 16) % 2 == 0 for g in groups])
evaluation = ~cal
assert int(cal.sum()) == config['calibration_clips']
assert int(evaluation.sum()) == config['evaluation_clips']
assert not set(groups[cal]) & set(groups[evaluation])

diagnostics = []
sensitivity = []
maximum_difference = 0.
for mode in BACKGROUND_MODES:
    candidates = owned_candidates(
        {m: logits[i] for i, m in enumerate(MODEL_NAMES)}, files.class_names,
        config['selection_spec']['expert_classes'], config['selection_spec']['model_class_names'],
        mode, config['selection_spec']['score_type'],
    )
    for target in (.90, .925, .95):
        common, thresholds = fit_thresholds(subset(candidates, cal), y[cal], target, 20)
        for policy in ('S0', 'S1', 'S2'):
            row, per_class = measure(subset(candidates, evaluation), y[evaluation], policy, common, thresholds)
            row.update(background_mode=mode, target_precision=target, common_threshold=common,
                       withheld_labels=[k for k, v in thresholds.items() if v is None],
                       fire_alarm_recall=next(r['recall'] for r in per_class if r['label'] == 'fire_alarm'),
                       knock_recall=next(r['recall'] for r in per_class if r['label'] == 'knock'))
            sensitivity.append(row)
            if target == .95:
                saved = next(r for r in actual if r['background_mode'] == mode and r['policy'] == policy)
                for key in ('accepted', 'accepted_precision', 'foreground_correct_accept_rate',
                            'background_false_accept_rate', 'foreground_macro_f1', 'unconfirmed', 'background_decisions'):
                    maximum_difference = max(maximum_difference, abs(float(saved[key]) - row[key]))
    if mode != 'global_max':
        continue
    top, score = candidates.top()
    for label in ('fire_alarm', 'knock', 'cry', 'baby_cry'):
        cid = files.class_names.index(label)
        predicted_cal = cal & (top == cid)
        predicted_eval = evaluation & (top == cid)
        true_eval = evaluation & (y == cid)
        possible = []
        for tau in np.unique(np.r_[np.arange(0., 1., .005), .999, 1.]):
            chosen = predicted_cal & (score >= tau)
            n = int(chosen.sum())
            if n >= 20:
                correct = int((chosen & (y == cid)).sum())
                possible.append((correct/n, correct, -float(tau), n))
        best = max(possible) if possible else None
        diagnostics.append(dict(
            label=label, calibration_true_clips=int((cal & (y == cid)).sum()),
            evaluation_true_clips=int(true_eval.sum()),
            calibration_true_recorded_groups=len(set(groups[cal & (y == cid)])),
            evaluation_true_recorded_groups=len(set(groups[true_eval])),
            S0_evaluation_output_count=int(predicted_eval.sum()),
            S0_evaluation_true_positives=int((predicted_eval & (y == cid)).sum()),
            S0_evaluation_false_positive_sources=dict(Counter(files.class_names[int(t)] for t in y[predicted_eval & (y != cid)])),
            S0_evaluation_max_output_score=float(score[predicted_eval].max()) if predicted_eval.any() else None,
            calibration_best_precision_with_minimum_20=None if best is None else best[0],
            calibration_best_precision_output_count=None if best is None else best[3],
            calibration_best_precision_tau=None if best is None else -best[2],
            evaluation_true_owner_score_quantiles=np.quantile(candidates.scores[true_eval,cid],[0,.25,.5,.75,1]).tolist(),
        ))

report = dict(reference_csv_identical=True, provenance_hashes_match=True,
              evaluation_foreground_clips=int((evaluation & (y != files.class_names.index('background'))).sum()),
              evaluation_background_clips=int((evaluation & (y == files.class_names.index('background'))).sum()),
              maximum_recomputed_S01_difference=maximum_difference,
              diagnostics=diagnostics, exploratory_sensitivity=sensitivity,
              scope='Exploratory reuse of the existing validation split; thresholds fit on calibration only. No independent model selection, timing or test evaluation.')
(Path(__file__).parent / 'selection_result_analysis.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps({k:v for k,v in report.items() if k != 'exploratory_sensitivity'}, ensure_ascii=False, indent=2))
print('GLOBAL_MAX_EXPLORATORY_SENSITIVITY')
for row in sensitivity:
    if row['background_mode'] == 'global_max' and row['policy'] != 'S0':
        print(json.dumps(row, ensure_ascii=False))
