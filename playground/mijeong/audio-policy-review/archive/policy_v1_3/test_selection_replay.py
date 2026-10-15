import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from evaluate_selection_replay import evaluate
from check_selection_reference import check
from policy_core import MODEL_NAMES
from prepare_selection_replay import prepare
from selection_replay_io import (SOURCE_NAMES, expected_keys, is_calibration_group, load_bundle,
                                 map_audio_path, sha_file, verify_checkpoint)


class ReplayTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='selection-replay-test-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.project, self.audio = self.root / 'project', self.root / 'audio'
        self.project.mkdir()
        self.audio.mkdir()
        self.names = ['background', 'a', 'b', 'c']
        for relative in SOURCE_NAMES:
            file = self.project / relative
            file.parent.mkdir(parents=True, exist_ok=True)
            file.write_text('# synthetic fixture only\n', encoding='utf-8')
        (self.project / 'class_names.txt').write_text('\n'.join(self.names), encoding='utf-8')
        self.rows = []
        for label in self.names:
            for cal in (True, False):
                number = 0
                while is_calibration_group(42, f'{label}-{cal}-{number}') != cal:
                    number += 1
                group = f'{label}-{cal}-{number}'
                relative = f'{label}/{group}.wav'
                (self.audio / label).mkdir(exist_ok=True)
                (self.audio / relative).write_bytes(b'synthetic path fixture; not decoded')
                self.rows.append(dict(path='/old/' + relative, label=label, split='val', group_id=group))
        with (self.project / 'metadata.csv').open('w', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(self.rows[0]))
            writer.writeheader()
            writer.writerows(self.rows)
        provenance = {}
        for model in MODEL_NAMES:
            folder = self.project / model
            folder.mkdir()
            file = folder / 'val_predictions.npz'
            np.savez_compressed(file, paths=np.array([r['path'] for r in self.rows]),
                                y_true=np.array([self.names.index(r['label']) for r in self.rows]),
                                logits=np.zeros((len(self.rows), len(self.names))))
            (folder / ('best.keras' if model == 'yamnet' else 'best.pt')).write_bytes(b'synthetic checkpoint')
            provenance[model] = dict(path=str(file), sha256=sha_file(file))
        self.config = dict(schema='owned_max_selection_v1', voting=False, score_renormalization=False,
                           seed=42, class_names=self.names,
                           metadata_sha256=sha_file(self.project / 'metadata.csv'),
                           calibration_clips=4, evaluation_clips=4, calibration_groups=4, evaluation_groups=4,
                           prediction_provenance=provenance,
                           selection_spec=dict(experiment_id='synthetic-test-only', score_type='softmax_logits',
                                               expert_classes=dict(efficientat=['a'], yamnet=['b'], mobilenetv4=['c']),
                                               model_class_names={m: self.names for m in MODEL_NAMES}),
                           policies={mode: dict(global_threshold=.6, class_thresholds=dict(a=.1, b=.2, c=.8))
                                     for mode in ('global_max', 'exclude', 'per_model_gate')})
        self.config_file = self.root / 'source.json'
        self.config_file.write_text(json.dumps(self.config), encoding='utf-8')
        self.out = self.root / 'replay'

    def prepared(self):
        return prepare(self.project, self.config_file, self.audio, self.out, old_dataset_root='/old')

    def caches(self):
        cfg, manifest, fingerprint = load_bundle(self.out)
        keys = sorted(expected_keys(manifest))
        # Owner max is c=.70, whereas averaging all model vectors picks a.
        vectors = dict(efficientat=[.05, .60, .30, .05],
                       yamnet=[.05, .80, .10, .05], mobilenetv4=[.05, .20, .05, .70])
        for model in MODEL_NAMES:
            values = []
            for key in keys:
                episode, end = map(int, key.split(':'))
                relevant = end > 32000 and end - 32000 < 64000
                false_preinsert = manifest['episodes'][episode]['label'] == 'background' and end == 32000
                values.append(np.log(vectors[model] if relevant or false_preinsert else [.94, .02, .02, .02]))
            order = np.arange(len(keys))[::-1] if model == 'yamnet' else np.arange(len(keys))
            np.savez_compressed(self.out / f'selection_replay_{model}.npz',
                                keys=np.array(keys)[order], logits=np.array(values)[order],
                                preprocessing_s=np.full(len(keys), .001), inference_s=np.full(len(keys), .002),
                                fingerprint=np.array(fingerprint), model_name=np.array(model),
                                score_type=np.array('softmax_logits'),
                                checkpoint_sha256=np.array(manifest['checkpoint_files'][model]['sha256']))
        return manifest

    def test_posix_to_local_mapping_and_traversal(self):
        self.assertEqual(map_audio_path('/old/a/x.wav', '/old', self.audio), self.audio / 'a' / 'x.wav')
        with self.assertRaises(ValueError):
            map_audio_path('/other/x.wav', '/old', self.audio)
        with self.assertRaises(ValueError):
            map_audio_path('/old/../x.wav', '/old', self.audio)

    def test_prepare_preserves_metadata_and_uses_only_evaluation_groups(self):
        before = sha_file(self.project / 'metadata.csv')
        manifest = self.prepared()
        self.assertEqual(len(manifest['episodes']), 4)
        self.assertTrue(all(not is_calibration_group(42, e['group_id']) for e in manifest['episodes']))
        self.assertEqual(before, sha_file(self.project / 'metadata.csv'))
        self.assertEqual(self.config_file.read_bytes(), (self.out / 'selection_config.json').read_bytes())
        with self.assertRaises(FileExistsError):
            self.prepared()

    def test_no_voting_no_fallback_hops_and_whole_episode_false_alerts(self):
        self.prepared()
        self.caches()
        summary, details = evaluate(self.out)
        self.assertEqual(len(summary), 27)
        row = next(r for r in details if r['background_mode'] == 'global_max' and r['label'] == 'c'
                   and r['output_policy'] == 'S0' and r['input_policy'] == 'I2')
        self.assertEqual(row['first_correct_audio_time_from_insert_s'], .5)
        slower = next(r for r in details if r['background_mode'] == 'global_max' and r['label'] == 'c'
                      and r['output_policy'] == 'S0' and r['input_policy'] == 'I0')
        self.assertEqual(slower['first_correct_audio_time_from_insert_s'], 2.)
        rejected = next(r for r in details if r['background_mode'] == 'global_max' and r['label'] == 'c'
                        and r['output_policy'] == 'S2' and r['input_policy'] == 'I2')
        self.assertIsNone(rejected['first_correct_audio_time_from_insert_s'])
        background = next(r for r in details if r['background_mode'] == 'global_max' and r['label'] == 'background'
                          and r['output_policy'] == 'S0' and r['input_policy'] == 'I2')
        self.assertTrue(background['background_episode_any_alert'])
        self.assertTrue(background['preinsert_any_alert'])
        with (self.out / 'selection_replay_window_decisions.csv').open(encoding='utf-8-sig', newline='') as stream:
            trace = list(csv.DictReader(stream))
        selected = next(r for r in trace if r['background_mode'] == 'global_max' and r['output_policy'] == 'S0' and r['key'] == '3:40000')
        self.assertEqual(selected['label'], 'c')
        rejected_trace = next(r for r in trace if r['background_mode'] == 'global_max' and r['output_policy'] == 'S2' and r['key'] == '3:40000')
        self.assertEqual(rejected_trace['status'], 'unconfirmed')

    def test_changed_config_and_checkpoint_are_rejected(self):
        manifest = self.prepared()
        file = self.out / 'selection_config.json'
        changed = json.loads(file.read_text(encoding='utf-8'))
        changed['policies']['global_max']['global_threshold'] = .9
        file.write_text(json.dumps(changed), encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'configuration changed'):
            load_bundle(self.out)
        checkpoint = Path(manifest['checkpoint_files']['efficientat']['path'])
        checkpoint.write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'Checkpoint changed'):
            verify_checkpoint(manifest, 'efficientat')

    def test_incomplete_or_stale_model_cache_is_rejected(self):
        self.prepared()
        self.caches()
        file = self.out / 'selection_replay_yamnet.npz'
        with np.load(file, allow_pickle=False) as data:
            arrays = {k: data[k] for k in data.files}
        arrays['fingerprint'] = np.array('stale')
        np.savez_compressed(file, **arrays)
        with self.assertRaisesRegex(ValueError, 'stale'):
            evaluate(self.out)
        self.caches()
        with np.load(file, allow_pickle=False) as data:
            arrays = {k: data[k] for k in data.files}
        for key in ('keys', 'logits', 'preprocessing_s', 'inference_s'):
            arrays[key] = arrays[key][:-1]
        np.savez_compressed(file, **arrays)
        with self.assertRaisesRegex(ValueError, 'Missing or extra'):
            evaluate(self.out)

    def test_full_source_window_reference_reports_mismatch_and_match(self):
        self.prepared()
        self.caches()
        self.assertFalse(check(self.out)['passed'])
        for model in MODEL_NAMES:
            file = self.out / f'selection_replay_{model}.npz'
            with np.load(file, allow_pickle=False) as data:
                arrays = {key: data[key] for key in data.files}
            full_windows = np.array([key.endswith(':64000') for key in arrays['keys']])
            arrays['logits'][full_windows] = 0.
            np.savez_compressed(file, **arrays)
        self.assertTrue(check(self.out)['passed'])


if __name__ == '__main__':
    unittest.main()
