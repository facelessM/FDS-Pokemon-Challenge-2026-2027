"""Regression checks for stale state-model caches after an experiment edit."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('state_cache_guard',ROOT/'experiments/state_reconstruction/cache_guard.py')
guard=importlib.util.module_from_spec(spec);spec.loader.exec_module(guard)


class CacheGuardTests(unittest.TestCase):
    def setUp(self):
        self.directory=tempfile.TemporaryDirectory();self.addCleanup(self.directory.cleanup)
        self.root=Path(self.directory.name);self.study=self.root/'study';self.data=self.root/'data'
        self.study.mkdir();self.data.mkdir()
        for name in ['scratchboard','deep_features','merge_check']:
            (self.study/name/'results').mkdir(parents=True)
        for name in ['scratchboard','deep_features']:(self.study/name/'cache').mkdir()
        (self.study/'deep_features/features.py').write_text('POWER = 1\n')
        (self.data/'train.jsonl').write_text('{"battle_id":1,"hp":1,"player_won":true}\n')
        (self.data/'type_chart_gen2.csv').write_text('attacking_type,defending_type,multiplier\n')
        (self.data/'test.jsonl').write_text('{"battle_id":2,"hp":1}\n')

    def stamp_and_cache(self,action='cv'):
        guard.validate_cache(self.study,self.data,action)
        path=self.study/'scratchboard/cache/20261026_0.pkl.gz';path.write_bytes(b'evaluated features')
        prediction=self.study/'deep_features/results/20261026_deep_entry_sets_lgb.npz';prediction.write_bytes(b'evaluated predictions')
        return path,prediction

    def test_same_inputs_reuse_existing_files(self):
        paths=self.stamp_and_cache();before=[p.read_bytes() for p in paths]
        self.assertIsNone(guard.validate_cache(self.study,self.data,'cv'))
        self.assertEqual(before,[p.read_bytes() for p in paths])

    def test_observation_edit_is_detected_even_with_same_ids_and_labels(self):
        paths=self.stamp_and_cache()
        (self.data/'train.jsonl').write_text('{"battle_id":1,"hp":0.5,"player_won":true}\n')
        with self.assertRaisesRegex(RuntimeError,'--refresh-cache'):guard.validate_cache(self.study,self.data,'cv')
        self.assertTrue(all(p.exists() for p in paths))

    def test_feature_source_edit_is_detected(self):
        self.stamp_and_cache();(self.study/'deep_features/features.py').write_text('POWER = 2\n')
        with self.assertRaises(RuntimeError):guard.validate_cache(self.study,self.data,'cv')

    def test_package_change_is_detected(self):
        with patch.object(guard.importlib.metadata,'version',return_value='1'):self.stamp_and_cache()
        with patch.object(guard.importlib.metadata,'version',return_value='2'):
            with self.assertRaises(RuntimeError):guard.validate_cache(self.study,self.data,'cv')

    def test_refresh_preserves_old_files_and_invalidates_prediction_shortcut(self):
        cached,prediction=self.stamp_and_cache();(self.study/'deep_features/features.py').write_text('POWER = 2\n')
        selection=self.study/'merge_check/selection.json';selection.write_text('{"weights":{"lgb":1}}')
        archive=guard.validate_cache(self.study,self.data,'cv',refresh=True)
        self.assertEqual((archive/cached.relative_to(self.study)).read_bytes(),b'evaluated features')
        self.assertEqual((archive/prediction.relative_to(self.study)).read_bytes(),b'evaluated predictions')
        self.assertFalse(cached.exists());self.assertFalse(prediction.exists());self.assertTrue(selection.exists())
        self.assertIsNone(guard.validate_cache(self.study,self.data,'cv'))

    def test_legacy_cache_needs_explicit_refresh(self):
        cached=self.study/'scratchboard/cache/full_fit.pkl.gz';cached.write_bytes(b'old')
        with self.assertRaisesRegex(RuntimeError,'no input manifest'):guard.validate_cache(self.study,self.data,'predict')
        self.assertFalse((self.study/'cache_manifest.json').exists());self.assertEqual(cached.read_bytes(),b'old')

    def test_training_validation_does_not_require_or_read_test(self):
        (self.data/'test.jsonl').unlink();self.stamp_and_cache()
        guard.validate_cache(self.study,self.data,'cv')

    def test_changed_test_is_rejected_for_prediction_only(self):
        self.stamp_and_cache('predict')
        (self.study/'scratchboard/cache/full_fit.pkl.gz').write_bytes(b'old test features')
        (self.data/'test.jsonl').write_text('{"battle_id":2,"hp":0.5}\n')
        guard.validate_cache(self.study,self.data,'cv')
        with self.assertRaisesRegex(RuntimeError,'test input'):guard.validate_cache(self.study,self.data,'predict')

    def test_first_prediction_after_cv_can_add_its_test_fingerprint(self):
        self.stamp_and_cache();guard.validate_cache(self.study,self.data,'predict')
        manifest=json.loads((self.study/'cache_manifest.json').read_text())
        self.assertEqual(manifest['test_sha256'],guard.file_hash(self.data/'test.jsonl'))

    def test_review_selection_does_not_invalidate_independent_feature_caches(self):
        paths=self.stamp_and_cache()
        (self.study/'merge_check/selection.json').write_text('{"weights":{"lgb":0.5,"xgb":0.5}}')
        self.assertIsNone(guard.validate_cache(self.study,self.data,'cv'))
        self.assertTrue(all(p.exists() for p in paths))

    def test_malformed_manifest_does_not_bless_existing_files(self):
        paths=self.stamp_and_cache();(self.study/'cache_manifest.json').write_text('[]')
        with self.assertRaisesRegex(RuntimeError,'cannot be read'):guard.validate_cache(self.study,self.data,'cv')
        self.assertTrue(all(p.exists() for p in paths))


if __name__=='__main__':unittest.main()
