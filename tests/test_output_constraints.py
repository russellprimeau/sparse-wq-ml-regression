"""Regression tests for physical-only output constraints and reconstruction order."""
from pathlib import Path
import sys,unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from sparse_wq.fitting import Analysis
from sparse_wq.models.samples import Window
from sparse_wq.output_constraints import constrain_predictions

TARGET='E.coli (CFU/100mL)'


class PhysicalOutputs(unittest.TestCase):
    def predictions(self,target,difference,minimum,maximum,raw):
        model=Analysis.__new__(Analysis);model.target=target;model.difference=difference
        model.rows=1;model.device='cpu';model.fit_count=0
        model.variant=lambda name:('xgb','mean')
        output=target+('_diff' if difference else '')
        context=dict(bounds={output:dict(min=minimum,max=maximum)},output=output,offsets=None,errors={})
        model.context=lambda *args,**kwargs:context
        train=[Window(np.array([[float(i)]]),np.array([[0.]]),f'train_{i}.csv') for i in range(6)]
        test=[Window(np.array([[1.]]),np.array([[0.]]),f'test_{i}.csv') for i in range(len(raw))]
        metadata={w.name:dict(timestamp=pd.Timestamp('2020-02-01')+pd.Timedelta(days=i),prior=10.,observed=2.) for i,w in enumerate(test)}
        training=pd.date_range('2020-01-01',periods=6);evaluation=pd.date_range('2020-02-01',periods=len(raw))
        model.raw_windows=lambda stamps,*args,**kwargs:(train if stamps.equals(training) else test,metadata)
        model.transformed=lambda windows,*args,**kwargs:(windows,0)
        with patch('sparse_wq.models.xgb.fit',return_value=(object(),{})),patch('sparse_wq.models.xgb.predict',return_value=np.array(raw,dtype=np.float32).reshape(-1,1)):
            return model.predict(training,evaluation,['A'],'xgb_01',hyper_override={'n_estimators':1})

    def test_nonnegative_assays_have_zero_floor_and_no_upper_cap(self):
        np.testing.assert_array_equal(constrain_predictions([-3.,0.,7.,1e8],TARGET),[0.,0.,7.,1e8])

    def test_ph_is_unconstrained(self):
        np.testing.assert_array_equal(constrain_predictions([-2.,6.,16.],'pH'),[-2.,6.,16.])

    def test_unknown_assay_requires_an_explicit_constraint(self):
        with self.assertRaisesRegex(ValueError,'registered'):constrain_predictions([1.],'unknown assay')

    def test_absolute_forecasts_may_exceed_training_maximum(self):
        frame,info=self.predictions(TARGET,False,0.,5.,[-.4,1.5])
        np.testing.assert_allclose(frame.prediction,[0.,7.5])
        np.testing.assert_allclose(frame.prediction_unclipped,[-2.,7.5])
        self.assertEqual(info['negative_prediction_floors'],1)

    def test_differential_response_is_reconstructed_before_the_floor(self):
        frame,_=self.predictions(TARGET,True,-5.,5.,[-.4,1.5])
        np.testing.assert_allclose(frame.prediction,[1.,20.])
        np.testing.assert_allclose(frame.prediction_unclipped,[1.,20.])

    def test_ph_forecasts_may_cross_both_training_bounds(self):
        frame,_=self.predictions('pH',False,6.,7.,[-8.,10.])
        np.testing.assert_allclose(frame.prediction,[-2.,16.])


if __name__=='__main__':unittest.main()
