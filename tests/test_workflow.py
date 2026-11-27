"""Selection separation, information cutoffs and fixed-hyperparameter follow-up."""
from pathlib import Path
import json
from types import SimpleNamespace
import sys
from contextlib import contextmanager
import shutil
from uuid import uuid4
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from sparse_wq.workflow import choices,slots,model_path
from sparse_wq.analysis import Analysis
from sparse_wq.evaluate.forecasts import prior_values
from sparse_wq.evaluate.block_permutation import rearrange_records
from sparse_wq.report.publication import r2_interval
from sparse_wq.evaluate.permutations import refresh_report


@contextmanager
def temporary_tree():
    parent=Path(__file__).resolve().parents[1]/'outputs/tests'
    root=parent/('test_'+uuid4().hex);root.mkdir(parents=True)
    try:yield root
    finally:
        assert root.resolve().parent==parent.resolve() and root.name.startswith('test_')
        shutil.rmtree(root)


class ScientificRules(unittest.TestCase):
    def test_locked_derived_report_does_not_propagate_into_search_control(self):
        with patch('sparse_wq.evaluate.permutations.report',side_effect=PermissionError('File open in reader')):
            self.assertIn('File open',refresh_report(Path('unused'),{}))

    def test_shifted_prior_cannot_use_a_value_inside_the_window(self):
        values=pd.Series([1.,2.,3.,4.],index=pd.to_datetime(['2020-01-01','2020-01-03','2020-01-05','2020-01-07']))
        target=pd.to_datetime(['2020-01-07'])
        self.assertEqual(prior_values(values,target,72,0)[0],2.)
        self.assertEqual(prior_values(values,target,72,48)[0],1.)

    def test_qualified_payload_moves_but_dates_and_other_targets_do_not(self):
        original=pd.DataFrame(dict(TIMESTAMP=pd.date_range('2020-01-01',periods=10),target=['A']*9+['B'],
            parameter=['Assay']*10,reported_number=np.arange(10),is_left_censored=[True,False]*5,limit=np.arange(10)+.1))
        changed,mapping,_=rearrange_records(original,'A',3,4101,1)
        pd.testing.assert_series_equal(original.TIMESTAMP,changed.TIMESTAMP)
        pd.testing.assert_series_equal(original.iloc[-1],changed.iloc[-1])
        for name in ['reported_number','is_left_censored','limit']:
            np.testing.assert_array_equal(changed.loc[:8,name],original.iloc[mapping.source_position][name])

    def test_no_undefined_floating_constant_bootstrap_r2(self):
        result=r2_interval(np.repeat(.07,8),np.repeat(.08,8))
        self.assertEqual(result['n_boot_valid'],0)
        self.assertIsNone(result['ci05'])

    def test_horizons_require_and_preserve_final_hyperparameters(self):
        model=Analysis.__new__(Analysis);model.uncertainty=False
        model.variant=lambda variant:('gp','mean')
        model.gp_budget=lambda *args: (_ for _ in ()).throw(AssertionError('Must not select another budget'))
        with self.assertRaisesRegex(ValueError,'saved final hyperparameters'):
            model.predict([],[],[],'gp_04',horizon=48)
        saved={'revision_fixed_steps':17}
        with patch('sparse_wq.fitting.Analysis.predict',return_value=('predictions','metadata')) as base:
            self.assertEqual(model.predict([],[],[],'gp_04',horizon=48,hyper_override=saved),('predictions','metadata'))
        self.assertEqual(base.call_args.kwargs['horizon'],48)
        self.assertEqual(base.call_args.kwargs['hyper_override'],saved)

    def test_prepared_fingerprints_accept_line_endings_but_reject_changed_values(self):
        import hashlib
        from sparse_wq.workflow import verify_supplied_results
        with temporary_tree() as root:
            prepared=root/'prepared';results=root/'results';prepared.mkdir();results.mkdir()
            data=b'value\n1.25\n'
            (prepared/'values.csv').write_bytes(data.replace(b'\n',b'\r\n'))
            (results/'reference_input_hashes.json').write_text(json.dumps({'values.csv':hashlib.sha256(data).hexdigest()}))
            verify_supplied_results(results,prepared)
            (prepared/'values.csv').write_bytes(b'value\n2.25\n')
            with self.assertRaisesRegex(ValueError,'do not match'):
                verify_supplied_results(results,prepared)

    def test_primary_predictor_set_is_fixed_without_reading_evaluation(self):
        from sparse_wq import workflow
        args=SimpleNamespace(target=['ph'],run=workflow.RUNS)
        store=SimpleNamespace(manifest={'study':{'targets':{'pH':'ph'}}})
        dates=pd.date_range('2020-01-01',periods=4)
        with temporary_tree() as root:
            for _,target,run,method in slots(args,store):
                directory=model_path(root,target,run,method);directory.mkdir(parents=True)
                # Profiler validation is better: that cannot change the fixed primary set.
                r2=.9 if run.startswith('profiler_') else -.5
                (directory/'selected.json').write_text(json.dumps(dict(info=dict(metrics=dict(r2=r2)))))
                pd.DataFrame(dict(timestamp=dates,prediction=[1.,2.,3.,4.])).to_csv(directory/'validation.csv',index=False)
            chosen=choices(args,root,store)
            self.assertEqual(chosen['ph']['run'],'noprofiler_differential')
            self.assertEqual(chosen['ph']['r2'],-.5)
            self.assertFalse(list(root.rglob('assessment')))


if __name__=='__main__':unittest.main()
