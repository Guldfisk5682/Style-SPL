"""Statistical validity checks, not tests of a trained model's target accuracy."""
import unittest
import numpy as np
import torch
from scripts.analyze_style_reliability import association, distances


class ReliabilityChecks(unittest.TestCase):
    def test_fixed_source_advantage_is_removed(self):
        rng = np.random.default_rng(7)
        labels = np.repeat(np.arange(6), 30)
        distance = np.tile([0., 1., 2.], (len(labels), 1)) + rng.normal(size=(len(labels), 1))
        loss = np.tile([1., 2., 3.], (len(labels), 1)) + rng.normal(size=(len(labels), 1))
        result = association(distance, loss, loss < 2, labels)
        self.assertGreater(result["within_image_similarity_vs_negative_nll_pearson"], .99)
        self.assertIsNone(result["class_source_adjusted_within_image_pearson"])

    def test_image_specific_signal_survives_fixed_effects(self):
        rng = np.random.default_rng(7)
        labels = np.repeat(np.arange(6), 30)
        distance = rng.normal(size=(len(labels), 3))
        loss = distance + np.arange(3)[None, :] * 2 + labels[:, None] * .3
        result = association(distance, loss, loss < 2, labels, permutation=True)
        self.assertGreater(result["class_source_adjusted_within_image_pearson"], .99)
        self.assertLess(result["adjusted_positive_association_permutation_p"], .01)

    def test_normalizer_does_not_consume_target_samples(self):
        generator = torch.Generator().manual_seed(77)
        styles = {d: {"blocks": [torch.rand(10, 8, generator=generator) for _ in range(4)]}
                  for d in ("s0", "s1", "s2", "target")}
        sources = ["s0", "s1", "s2"]
        bank = {"sources": [[(b.mean(0)[:4], b.mean(0)[4:]) for b in styles[s]["blocks"]]
                            for s in sources]}
        target_blocks = [b.clone() for b in styles["target"]["blocks"]]
        before, meta_before = distances(target_blocks, bank, styles, sources)
        styles["target"]["blocks"] = [b * 10000 for b in target_blocks]
        after, meta_after = distances(target_blocks, bank, styles, sources)
        self.assertEqual(meta_before, meta_after)
        for name in before:
            np.testing.assert_array_equal(before[name], after[name])


if __name__ == "__main__":
    unittest.main()
