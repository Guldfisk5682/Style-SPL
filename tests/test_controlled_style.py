import unittest

import torch

from style import DomainStyleProjector, FixedStyleBank, calibrate_initial_output
from test_style import synthetic_bank


class ControlledStyleTests(unittest.TestCase):
    def test_silu_calibration_preserves_first_layer_and_expansion(self):
        bank = FixedStyleBank(synthetic_bank())
        projector = DomainStyleProjector(architecture="silu", bottleneck_dim=32)
        first = [{k: v.clone() for k, v in s[0].state_dict().items()} for s in projector.stage]
        second = [{k: v.clone() for k, v in s[2].state_dict().items()} for s in projector.stage]
        expansion = projector.expansion.clone()
        report = calibrate_initial_output(projector, bank)
        self.assertAlmostEqual(report["after"]["global_rms"], .02, places=6)
        for stage, before, last in zip(projector.stage, first, second):
            for k, v in stage[0].state_dict().items():
                torch.testing.assert_close(v, before[k], rtol=0, atol=0)
            for k, v in stage[2].state_dict().items():
                torch.testing.assert_close(v, last[k] * report["factor"], rtol=0, atol=0)
        torch.testing.assert_close(projector.expansion, expansion, rtol=0, atol=0)

    def test_nonlinearity_breaks_affine_pooled_identity(self):
        torch.manual_seed(7)
        bank = FixedStyleBank(synthetic_bank())
        weights = torch.tensor([.2, .3, .5])
        linear = DomainStyleProjector()
        mlp = DomainStyleProjector(architecture="silu")
        for projector in (linear, mlp):
            calibrate_initial_output(projector, bank)
        def residual(projector):
            source = torch.stack([projector(bank.entry(i)) for i in range(3)])
            return projector(bank.entry(3)) - (source * weights[:, None, None]).sum(0)
        self.assertLess(residual(linear).norm().item(), 1e-6)
        self.assertGreater(residual(mlp).norm().item(), .001)


if __name__ == "__main__":
    unittest.main()
