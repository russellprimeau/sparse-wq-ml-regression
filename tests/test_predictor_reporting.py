"""Report the variables actually used, not pre-screening candidate subsets."""
from pathlib import Path
import unittest
from sparse_wq.evaluate.block_permutation import read_json
from sparse_wq.report.publication import fitted_predictors

ROOT = Path(__file__).resolve().parents[1]


class PredictorReporting(unittest.TestCase):
    def test_screened_and_intercept_only_mlr_inputs_are_authoritative(self):
        details = [dict(family='mlr', input_features=['weather', 'prior_state'], selected_features=['prior_state']),
                   dict(family='mlr', input_features=['weather'], selected_features=[])]
        self.assertEqual(fitted_predictors(details), ['prior_state'])
        self.assertEqual(fitted_predictors(details[1:]), [])

    def test_union_preserves_final_fit_order(self):
        details = [dict(family='xgb', input_features=['wind', 'prior_state']),
                   dict(family='xgb', input_features=['temperature', 'wind'])]
        self.assertEqual(fitted_predictors(details), ['wind', 'prior_state', 'temperature'])

    def test_supplied_positive_primary_mlr_models_use_only_the_prior(self):
        for target in ['colony_count_22c', 'zinc']:
            folder = ROOT / 'results/models/shared/half/noprofiler_differential' / target
            fit = read_json(folder / 'assessment/fit.json')
            final = fitted_predictors(fit['details'])
            self.assertEqual(len(final), 1)
            self.assertTrue(final[0].endswith('_state'))
            self.assertGreater(len(read_json(folder / 'selected.json')['features']), len(final))


if __name__ == '__main__':
    unittest.main()
