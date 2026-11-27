from pathlib import Path
import sys
import unittest
import json

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

import numpy as np
import pandas as pd

from sparse_wq.data.qualified import substitute
from sparse_wq.evaluate.targeted_checks import render_targeted_tables
from sparse_wq.report.publication import number
from test_workflow import temporary_tree


class TargetedChecks(unittest.TestCase):
    def test_substitution_keeps_unqualified_zeros_and_qualifiers(self):
        records = pd.DataFrame(dict(reported_number=[0., .004, .008],
            is_left_censored=[False, True, False], limit=[np.nan, .004, np.nan]))
        half, alternative = [substitute(records, policy) for policy in ('half', 'sqrt2')]
        np.testing.assert_array_equal(half.loc[[0, 2], 'value'], alternative.loc[[0, 2], 'value'])
        self.assertEqual(alternative.loc[0, 'value'], 0.)
        self.assertAlmostEqual(alternative.loc[1, 'value'], .004 / np.sqrt(2))
        pd.testing.assert_frame_equal(alternative[records.columns], records)

    def test_rounding_hides_only_the_sign_of_rounded_zero(self):
        self.assertEqual(number(-0.0000206, 2), '0.00')
        self.assertEqual(number(-0.006, 2), '-0.01')
        self.assertEqual(number(-0.006, 3), '-0.006')
        self.assertEqual(number(None, 2), '--')

    def test_changed_reported_model_cannot_inherit_old_sensitivity_checks(self):
        with temporary_tree() as root:
            source, output = root / 'results/targeted_checks', root / 'outputs/analysis'
            source.mkdir(parents=True);output.mkdir(parents=True)
            (source / 'source_models.json').write_text(json.dumps(dict(
                case_fields=['target', 'run'], cases={'ph': dict(target='ph', run='old_arm')}, files={})))
            (output / 'results.json').write_text(json.dumps(dict(targets=[dict(target='ph', run='new_arm')])))
            with self.assertRaisesRegex(ValueError, 'do not match'):
                render_targeted_tables(source, output)


if __name__ == '__main__':
    unittest.main()
