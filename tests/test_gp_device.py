"""A requested CPU fit must also predict on CPU on a CUDA-capable machine."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from sparse_wq.models import gp
from sparse_wq.models.samples import Window


class GPDevice(unittest.TestCase):
    def test_cpu_fit_and_prediction_do_not_request_cuda(self):
        windows = [Window(np.array([[float(i)]]), np.array([[float(i) / 5]]), str(i))
                   for i in range(6)]
        prediction_windows = [Window(np.array([[x]]), np.array([[0.]]), str(x))
                              for x in [.5, 2.5, 4.5]]
        hyper = dict(kernel='matern52', num_epochs=2, revision_fixed_steps=2,
                     learning_rate=.01, random_state=0)
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

        def cpu_only(requested):
            self.assertEqual(str(requested), 'cpu', 'CPU prediction must not select CUDA')
            return torch.device('cpu')

        try:
            # Guard device requests directly; catches CUDA selection even when
            # the test host itself has no GPU.
            with patch.object(gp, 'resolve_device', side_effect=cpu_only):
                fitted = gp.fit(hyper, windows, ['x'], 1, 'mean', {}, None, 'cpu')
                self.assertEqual(fitted.device_name, 'cpu')
                mean, variance = gp.predict(fitted, windows, prediction_windows)
            self.assertEqual(mean.shape, (3, 1))
            self.assertTrue(np.isfinite(mean).all())
            self.assertTrue(np.isfinite(variance).all())
            self.assertTrue((variance >= 0).all())
        finally:
            torch.set_num_threads(previous_threads)


if __name__ == '__main__':
    unittest.main()
