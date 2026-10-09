import unittest

import torch

from descriptor_controls import descriptor_control, bank_digest
from style import FixedStyleBank
from test_style import synthetic_bank


class DescriptorControlTests(unittest.TestCase):
    def setUp(self):
        self.bank = FixedStyleBank(synthetic_bank())

    def test_shuffle_whole_domains_and_preserve_training_rng(self):
        before, digest = torch.get_rng_state().clone(), bank_digest(self.bank)
        changed, report = descriptor_control(self.bank, "d", "shuffled", seed=71)
        torch.testing.assert_close(torch.get_rng_state(), before, rtol=0, atol=0)
        self.assertEqual(bank_digest(self.bank), digest)
        names, rows = ["a", "b", "c", "d"], {"a": 0, "b": 1, "c": 2, "d": 4}
        for name in names:
            donor = report["domain_to_descriptor"][name]
            self.assertNotEqual(name, donor)
            for stage in range(4):
                for statistic in ("mean", "std"):
                    a, b = getattr(changed, f"{statistic}_{stage}"), getattr(self.bank, f"{statistic}_{stage}")
                    torch.testing.assert_close(a[rows[name]], b[rows[donor]], rtol=0, atol=0)
                    torch.testing.assert_close(a[3], b[3], rtol=0, atol=0)

    def test_random_codes_match_scale_are_distinct_and_repeatable(self):
        state = torch.get_rng_state().clone()
        changed, report = descriptor_control(self.bank, "d", "random", seed=71)
        again, _ = descriptor_control(self.bank, "d", "random", seed=71)
        self.assertEqual(bank_digest(changed), bank_digest(again))
        torch.testing.assert_close(torch.get_rng_state(), state, rtol=0, atol=0)
        for block in report["blocks"]:
            self.assertAlmostEqual(block["real_rms"], block["control_rms"], places=6)
            for value in block["per_domain_mean"]:
                self.assertAlmostEqual(value, block["real_mean"], places=6)
            for value in block["per_domain_std"]:
                self.assertAlmostEqual(value, block["real_std"], places=6)
        self.assertFalse(torch.equal(changed.mean_0[0], changed.mean_0[1]))
        torch.testing.assert_close(changed.mean_0[3], self.bank.mean_0[3], rtol=0, atol=0)

    def test_real_is_exact(self):
        same, _ = descriptor_control(self.bank, "d")
        self.assertEqual(bank_digest(same), bank_digest(self.bank))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_random_cpu_cuda_construction_identical(self):
        cpu, _ = descriptor_control(self.bank, "d", "random")
        gpu, _ = descriptor_control(self.bank.cuda(), "d", "random")
        self.assertEqual(bank_digest(cpu), bank_digest(gpu))


if __name__ == "__main__":
    unittest.main()
