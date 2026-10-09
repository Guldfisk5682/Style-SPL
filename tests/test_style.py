import tempfile
import unittest
from pathlib import Path

import torch

from style import (CHANNELS, DomainStyleProjector, FixedStyleBank, new_accumulator,
                   update_accumulator, finalize, pool_accumulators, spatial_statistics,
                   calibrate_initial_output, validate_bank)
from runtime import preserve_rng, gradient_check, parameter_check, ReplayableLoader, fix_random_seed


def synthetic_bank():
    accs = []
    for n, value in [(2, 1.0), (3, 2.0), (5, 4.0), (7, 3.0)]:
        acc = new_accumulator()
        update_accumulator(acc, [(torch.full((n, c), value), torch.full((n, c), value/2)) for c in CHANNELS])
        accs.append(acc)
    return {"sources": [finalize(a) for a in accs[:3]], "pooled": finalize(pool_accumulators(accs[:3])),
            "target": finalize(accs[3]), "metadata": {"source_domain_order": ["a", "b", "c"],
            "counts": [2, 3, 5, 10, 7], "cache_identity": {"transform": "fixed", "split": 1}}}


class StyleTests(unittest.TestCase):
    def test_population_statistics_and_per_image_std(self):
        x = torch.tensor([[[[1., 3.], [5., 7.]]], [[[100., 102.], [104., 106.]]]])
        mean, std = spatial_statistics(x)
        torch.testing.assert_close(mean, torch.tensor([[4.], [103.]]))
        expected = ((x - mean[..., None, None]).square().sum((-2,-1)).add(1e-8)/4).sqrt()
        torch.testing.assert_close(std, expected, rtol=0, atol=0)
        self.assertAlmostEqual(std.mean().item(), 5**0.5, places=6)
        self.assertGreater(x.flatten().std(unbiased=False).item(), std.mean().item()*10)

    def test_weighted_pool_cache_identity_and_roundtrip(self):
        bank = synthetic_bank()
        validate_bank(bank, bank["metadata"]["cache_identity"])
        self.assertAlmostEqual(bank["pooled"][0][0][0].item(), 2.8, places=6)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"bank.pt"
            torch.save(bank, path)
            loaded = torch.load(path, weights_only=True)
            validate_bank(loaded, bank["metadata"]["cache_identity"])
            for c, (mean, std) in zip(CHANNELS, loaded["target"]):
                self.assertEqual(mean.shape, (c,))
                self.assertFalse(mean.requires_grad)
        with self.assertRaises(ValueError):
            validate_bank(bank, {"transform": "changed", "split": 1})

    def test_expansion_calibration_gradients_and_no_old_domains(self):
        bank = FixedStyleBank(synthetic_bank())
        projector = DomainStyleProjector()
        expected = projector.stage_tokens(bank.entry(0))[torch.arange(16) % 4]
        torch.testing.assert_close(projector(bank.entry(0)), expected)
        self.assertEqual(sum(p.numel() for p in projector.parameters()), 3934272)
        report = calibrate_initial_output(projector, bank)
        self.assertAlmostEqual(report["after"]["global_rms"], .02, places=6)
        with self.assertRaises(RuntimeError):
            calibrate_initial_output(projector, bank)
        self.assertFalse(any(b.requires_grad for b in bank.buffers()))
        optimizer = torch.optim.AdamW(projector.parameters(), lr=.005)
        before = {n: p.detach().clone() for n,p in projector.named_parameters()}
        loss = sum(projector(bank.entry(i)).square().mean() for i in range(5))
        loss.backward()
        gradients = {n:p.grad.clone() for n,p in projector.named_parameters()}
        with preserve_rng():
            norms = gradient_check(projector, {"loss":loss})
        for n,p in projector.named_parameters():
            torch.testing.assert_close(p.grad, gradients[n], rtol=0, atol=0)
            self.assertGreater(norms[n], 0)
        optimizer.step()
        parameter_check(projector)
        self.assertTrue(all(not torch.equal(before[n], p) for n,p in projector.named_parameters()))

    def test_replayable_loader_matches_uninterrupted_shuffle(self):
        fix_random_seed(1)
        loader = torch.utils.data.DataLoader(torch.arange(40), batch_size=4, shuffle=True)
        stream = ReplayableLoader(loader)
        for _ in range(3):
            stream.next()
        saved = stream.state_dict()
        expected = stream.next()
        restored = ReplayableLoader(loader)
        restored.load_state_dict(saved)
        torch.testing.assert_close(restored.next(), expected, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
