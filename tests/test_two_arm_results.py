"""Numerical checks of the supplied two-set evidence and common references."""
from pathlib import Path
import unittest
import numpy as np
import pandas as pd
from sparse_wq import workflow
from sparse_wq.evaluate.block_permutation import read_json

ROOT=Path(__file__).resolve().parents[1]


class SuppliedTwoArmResults(unittest.TestCase):
    def test_all_28_forecast_sets_keep_nominal_partitions_and_common_references(self):
        records=workflow.load_records(ROOT/'data/derived')
        names=read_json(ROOT/'data/derived/preparation.json')['study']['targets']
        count=0
        for name,target in names.items():
            views=[];identities=[]
            for run,method in [('noprofiler_differential','shared'),('profiler_differential','off')]:
                source=workflow.model_path(ROOT/'results/models',target,run,method)
                selected=read_json(source/'selected.json');identity=read_json(source/'identity.json')
                identities.append(identity)
                lab=records.loc[records.target.eq(name)].set_index('TIMESTAMP').value
                view=workflow.view(source,selected,lab,identity['rows'],'evaluation',target=name)
                self.assertFalse(pd.DatetimeIndex(identity['train']).intersection(view['table'].index).size)
                self.assertTrue(set(view['available'])<=set(pd.DatetimeIndex(identity['test'])))
                views.append(view);count+=1
            self.assertEqual(identities[0]['train'],identities[1]['train'])
            self.assertEqual(identities[0]['test'],identities[1]['test'])
            self.assertEqual(identities[0]['rows'],identities[1]['rows'])
            dates=views[0]['available'].intersection(views[1]['available'])
            np.testing.assert_array_equal(views[0]['table'].loc[dates,'observed'],views[1]['table'].loc[dates,'observed'])
            np.testing.assert_allclose(views[0]['references'].loc[dates],views[1]['references'].loc[dates],rtol=0,atol=1e-12)
        self.assertEqual(count,28)

    def test_gp_controls_use_same_training_and_evaluation_membership(self):
        source=ROOT/'results/models/shared/half/noprofiler_differential/total_coliforms_37c'
        fit=read_json(source/'assessment/fit.json')
        expected={pd.to_datetime(m.removeprefix('segment_').removesuffix('.csv'),format='%Y%m%d%H') for d in fit['details'] for m in d['training_observations']}
        dates=pd.DatetimeIndex(read_json(source/'identity.json')['test'])
        for control in ['prior_only','prior_and_calendar']:
            folder=ROOT/'results/gp_controls/controls'/control
            value=read_json(folder/'fit.json')
            actual=set(pd.to_datetime(value['details'][0]['training_observations']))
            self.assertEqual(actual,expected)
            for detail in value['validation_details']:
                self.assertLess(max(pd.to_datetime(detail['training_observations'])),min(pd.to_datetime(detail['evaluation_observations'])))
            table=pd.read_csv(folder/'predictions_and_references.csv',parse_dates=['timestamp'])
            pd.testing.assert_index_equal(pd.DatetimeIndex(table.timestamp),dates,check_names=False)


if __name__=='__main__':unittest.main()
