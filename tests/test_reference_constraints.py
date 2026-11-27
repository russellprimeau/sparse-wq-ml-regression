"""References use the same physical forecast bounds as retained ML models."""
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from sparse_wq.evaluate.forecasts import reference_forecasts


class ReferenceConstraints(unittest.TestCase):
    def test_negative_regression_forecasts_are_floored_except_for_ph(self):
        lab=pd.Series([4.,3.,2.,1.],index=pd.date_range('2020-01-01',periods=4))
        args=(lab,lab.index,pd.DatetimeIndex(['2020-01-06']),23)
        unconstrained=reference_forecasts(*args,target='pH')
        constrained=reference_forecasts(*args,target='Color')
        for name in ['Calendar','Calendar + prior','Linear']:
            self.assertLess(unconstrained[name][0],0)
            self.assertEqual(constrained[name][0],0)
        self.assertTrue(np.isnan(constrained['Seasonal'][0]))
        self.assertEqual(constrained['Naive'][0],unconstrained['Naive'][0])

    def test_reference_forecasts_can_exceed_the_training_maximum(self):
        lab=pd.Series([1.,2.,3.,4.],index=pd.date_range('2020-01-01',periods=4))
        values=reference_forecasts(lab,lab.index,pd.DatetimeIndex(['2020-01-06']),23,target='Color')
        self.assertGreater(values['Linear'][0],lab.max())

    def test_target_identity_is_explicit_and_must_be_registered(self):
        lab=pd.Series([1.,2.,3.,4.],index=pd.date_range('2020-01-01',periods=4))
        with self.assertRaises(TypeError):reference_forecasts(lab,lab.index,['2020-01-06'],23)
        with self.assertRaisesRegex(ValueError,'No physical output constraint'):
            reference_forecasts(lab,lab.index,['2020-01-06'],23,target='unknown assay')


if __name__=='__main__':unittest.main()
