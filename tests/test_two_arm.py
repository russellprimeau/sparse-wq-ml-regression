"""Guard the retained study scope and ordinary GP covariance."""
from pathlib import Path
from types import SimpleNamespace
import sys,unittest,json
from test_workflow import temporary_tree
from unittest.mock import patch
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from sparse_wq import workflow,design
from sparse_wq.models import gp
from sparse_wq.fitting import Analysis
from sparse_wq.evaluate.compatibility import copy_horizon_scores
from sparse_wq.report.permutation_results import copy_supplied


class TwoArmStudy(unittest.TestCase):
    def test_registry_contains_two_unique_original_input_arms(self):
        args=SimpleNamespace(target=None,run=workflow.RUNS)
        store=SimpleNamespace(manifest={'study':{'targets':{'pH':'ph'}}})
        slots=list(workflow.slots(args,store))
        self.assertEqual(len(slots),2)
        self.assertEqual({(run,method) for _,_,run,method in slots},set(design.ARM_KEYS))
        self.assertEqual({method for *_,method in slots},{'off','shared'})

    def test_fitting_refuses_sensor_error_propagation_before_loading_inputs(self):
        model=Analysis.__new__(Analysis)
        with patch.object(model,'context',create=True,side_effect=AssertionError('Should not load input files')):
            with self.assertRaisesRegex(ValueError,'original measured'):
                model.predict([],[],[],'gp_01',uncertainty=True)

    def test_ordinary_covariance_is_symmetric_psd_and_has_consistent_diagonal(self):
        torch.set_num_threads(1)
        x=torch.tensor([[0.,.2],[.7,.4],[.7,.4],[-.3,.8]],dtype=torch.float64)
        for name in ['matern52','matern52+linear']:
            kernel=gp.make_kernel(name,None,2).double()
            matrix=kernel(x).to_dense()
            torch.testing.assert_close(matrix,matrix.T)
            torch.testing.assert_close(kernel(x,diag=True),matrix.diagonal())
            self.assertGreaterEqual(float(torch.linalg.eigvalsh(matrix).min().detach()),-1e-10)

    def test_kernel_refuses_error_draws(self):
        with self.assertRaisesRegex(ValueError,'ordinary GP'):
            gp.make_kernel('matern52',torch.zeros((4,2)),2)

    def test_changed_horizon_model_removes_stale_summary(self):
        with temporary_tree() as root:
            results=root/'results';output=root/'output'
            (results/'horizons').mkdir(parents=True);(results/'bundle').mkdir();output.mkdir()
            (results/'horizons/source_models.json').write_text(json.dumps({'ph':{'run':'old'}}))
            (results/'bundle/horizon_scores.csv').write_text('target,horizon,r2\nph,0,0.5\n')
            stale=output/'horizon_scores.csv';stale.write_text('stale')
            cases=[dict(target='ph',eligible_for_followup=True,run='changed')]
            self.assertFalse(copy_horizon_scores(results,root/'models',output,cases))
            self.assertFalse(stale.exists())

    def test_changed_permutation_model_removes_stale_products(self):
        with temporary_tree() as root:
            results=root/'results';output=root/'output'
            (results/'permutations').mkdir(parents=True);(output/'tables').mkdir(parents=True)
            (results/'permutations/source_models.json').write_text(json.dumps({'case_fields':['run'],'cases':{'ph':{'run':'old'}}}))
            names=['tables/block_permutation.tex','block_permutation_summary.json','block_permutation_comparisons.csv']
            for name in names:(output/name).write_text('stale')
            cases=[dict(target='ph',scores={'r2':.4},run='changed')]
            self.assertFalse(copy_supplied(output,cases,results=results,models=root/'models'))
            self.assertTrue(all(not (output/name).exists() for name in names))


if __name__=='__main__':unittest.main()
