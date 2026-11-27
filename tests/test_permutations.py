"""Pairing, repetition scope and original-data reselection of the reported experiment."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import unittest

import numpy as np
import pandas as pd

from test_workflow import temporary_tree
from sparse_wq.evaluate.permutations import paired_result, same_configuration, run
from sparse_wq.evaluate.block_permutation import read_json, save_json


class Permutations(unittest.TestCase):
    def test_paired_errors_use_the_same_available_dates_and_all_references(self):
        dates=pd.date_range('2020-01-01',periods=4)
        original=pd.DataFrame(dict(observed=[1.,2.,3.,4.],prediction=[1.,2.,2.,4.],
            **{'reference: Training mean':[2.,2.,2.,2.]}),index=dates)
        permuted=original.copy()
        permuted['observed']=[4.,3.,2.,1.]
        permuted.loc[dates[1],'prediction']=np.nan
        result=paired_result(original,permuted,dates,['Training mean'])
        self.assertEqual(result['n_matched'],3)
        self.assertFalse(result['fixed_set_complete'])
        self.assertEqual(result['missing_timestamps'],[str(dates[1])])
        self.assertAlmostEqual(result['observed_matched_scores']['rmse'],np.sqrt(1/3))
        self.assertAlmostEqual(result['matched_scores']['rmse'],np.sqrt(6))

    def test_configuration_identity_accepts_json_sequences(self):
        self.assertTrue(same_configuration(dict(variant='gp_01',features=('a',),seeds=(0,)),
                                           dict(variant='gp_01',features=['a'],seeds=[0])))

    def test_one_repetition_searches_original_and_rearranged_records_and_resumes(self):
        from sparse_wq.evaluate.block_permutation import REFERENCE_NAMES
        calls=[]
        with temporary_tree() as root:
            prepared=root/'prepared';prepared.mkdir()
            dates=pd.date_range('2020-01-01',periods=9)
            records=pd.DataFrame(dict(TIMESTAMP=dates,target=['pH']*9,parameter=['pH']*9,
                reported_number=np.arange(9,dtype=float)+1,is_left_censored=False,limit=np.nan))
            records.to_csv(prepared/'qualified_observations.csv',index=False)
            (prepared/'search.yml').write_text('models:\n  xgb:\n    hyperparameters: {}\n')
            save_json(prepared/'preparation.json',dict(outputs_sha256={}))
            source=root/'results/models/off/half/profiler_differential/ph'
            save_json(source/'identity.json',dict(rows=24,train=list(map(str,dates[:6])),test=list(map(str,dates[6:]))))
            selected=dict(features=['x'],variant='xgb_01',seeds=[0],objective=0.)
            save_json(source/'selected.json',selected)
            save_json(source/'assessment/fit.json',dict(details=[]))
            pd.DataFrame(dict(timestamp=dates[6:],observed=[7.,8.,9.],prediction=[7.,8.,9.],prior=[1.,1.,1.])).to_csv(source/'assessment/predictions.csv',index=False)
            case=dict(target='ph',target_name='pH',laboratory_target='pH',run='profiler_differential',
                method='off',eligible_for_followup=True,rows=24,evaluation=dict(timestamps=list(map(str,dates[6:]))),
                configuration='XGBoost',features=['x'],variant='xgb_01',scores=dict(r2=1.,baselines={name:{} for name in REFERENCE_NAMES.values()}))
            save_json(root/'results/bundle/results.json',dict(targets=[case]))
            args=SimpleNamespace(results=root/'results',output=root/'outputs',prepared=prepared,
                repetitions=1,target=None,case=None,budget=240,trials=250,epochs=250)

            class Model:
                def __init__(self,local,store,modelroot,*unused):
                    self.out=modelroot/'off/half/profiler_differential/ph'
                    self.rows=24;self.train=dates[:6];self.test=dates[6:]
                    self.lab=SimpleNamespace(value=records.set_index('TIMESTAMP').reported_number)
                    self.target='pH';self.settings={}
                    self.local=local
                def search(self):
                    values=pd.read_csv(self.local.prepared/'qualified_observations.csv').reported_number.tolist()
                    calls.append(values);save_json(self.out/'selected.json',selected);return selected
                def assess(self,chosen):
                    frame=pd.DataFrame(dict(timestamp=dates[6:],observed=[7.,8.,9.],prediction=[7.,8.,9.],prior=[1.,1.,1.]))
                    save_json(self.out/'assessment/fit.json',dict(details=[]))
                    return frame,{}

            def scored(frame,*unused,**kwargs):
                frame=frame.copy();frame['timestamp']=pd.to_datetime(frame.timestamp)
                frame=frame.set_index('timestamp')
                for name in REFERENCE_NAMES.values():frame['reference: '+name]=5.
                return frame
            with patch('sparse_wq.workflow.analysis',side_effect=Model),patch('sparse_wq.workflow.run_signature',return_value={}),\
                 patch('sparse_wq.workflow.configuration_name',return_value='XGBoost'),\
                 patch('sparse_wq.data.native.NativeInputs',return_value=object()),\
                 patch('sparse_wq.evaluate.permutations.score_predictions',side_effect=scored):
                run(args,object());run(args,object())
            self.assertEqual(len(calls),2,'Search original once and the one rearrangement once; resume neither')
            self.assertEqual(calls[0],list(np.arange(9,dtype=float)+1))
            self.assertNotEqual(calls[0],calls[1])
            summary=read_json(root/'outputs/permutations/report/summary.json')
            self.assertEqual(summary['balanced_repetitions'],1)
            self.assertEqual(summary['targets'][0]['completed'],1)


if __name__=='__main__':unittest.main()
