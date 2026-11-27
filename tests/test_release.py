"""Keep the supplied numerical release small and independently check its forecasts."""
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

from sparse_wq.evaluate.block_permutation import read_json, metrics, digest

ROOT=Path(__file__).resolve().parents[1]


class Release(unittest.TestCase):
    def test_results_exclude_search_caches_and_fit_github_limits(self):
        files=[p for p in (ROOT/'results').rglob('*') if p.is_file()]
        forbidden={'finalists.json','search_trace.json','campaign.json','status.json',
                   'trial_protocol.json','finalizer_context.json','continuation.json'}
        self.assertFalse([str(p) for p in files if p.name in forbidden])
        self.assertTrue(all(p.stat().st_size < 50*2**20 for p in files))
        self.assertLess(sum(p.stat().st_size for p in files),50*2**20)

    def test_all_10_supplied_pairs_are_recoverable_from_forecasts(self):
        root=ROOT/'results/permutations'
        plan=read_json(root/'experiment.json')
        self.assertEqual(len(plan['cases']),1)
        self.assertEqual(plan['repetitions'],10)
        count=0
        for case in plan['cases']:
            original=pd.read_csv(root/'jobs'/case['target']/'original/predictions_and_references.csv',
                parse_dates=['timestamp'],float_precision='round_trip').set_index('timestamp')
            for repetition in range(1,11):
                job=root/'jobs'/case['target']/f'{repetition:06d}'
                record=read_json(job/'result.json')
                rearranged=pd.read_csv(job/'predictions_and_references.csv',parse_dates=['timestamp'],
                    float_precision='round_trip').set_index('timestamp')
                dates=pd.DatetimeIndex(record['timestamps'])
                refs=list(case['scores']['baselines'])
                self.assertTrue(set(dates)<=set(pd.DatetimeIndex(case['evaluation']['timestamps'])))
                self.assertEqual(len(dates),record['n_matched'])
                for label,table in [('observed_matched_scores',original),('matched_scores',rearranged)]:
                    recomputed=metrics(table.loc[dates],refs)
                    for metric in ['r2','rmse','mae','skill']:
                        self.assertAlmostEqual(recomputed[metric],record[label][metric],places=12)
                    for reference in refs:
                        self.assertAlmostEqual(recomputed['baselines'][reference]['rmse'],
                            record[label]['baselines'][reference]['rmse'],places=12)
                count+=1
        self.assertEqual(count,10)
        for name,expected in read_json(root/'release_manifest.json').items():
            self.assertEqual(digest(root/name),expected,name)


if __name__=='__main__':unittest.main()
