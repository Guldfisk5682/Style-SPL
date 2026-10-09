import sys
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from round_analysis_metrics import prompt_geometry, text_geometry, stage_contributions, teacher_scores


class AnalysisTests(unittest.TestCase):
    def test_cosine_compares_same_class_instead_of_mean_class_prototype(self):
        features = torch.tensor([[[1., 0.], [0., 1.]], [[0., 1.], [1., 0.]]])
        stats, values = text_geometry(features)
        self.assertEqual(stats["mean"][0][1], 0.)
        self.assertEqual(values.shape, (2, 2, 2))

    def test_centered_geometry_removes_large_common_offset(self):
        tokens = torch.arange(30, dtype=torch.float).reshape(5, 3, 2) / 30
        small = prompt_geometry(tokens, torch.ones(2, 3, 2))
        large = prompt_geometry(tokens + 100, torch.ones(2, 3, 2))
        self.assertGreater(large["flattened_cosine"][0][4], small["flattened_cosine"][0][4])
        self.assertAlmostEqual(large["centered_style_rms"], small["centered_style_rms"], places=5)
        self.assertGreater(large["common_style_energy_fraction"], .999)

    def test_stage_shares_keep_cancellation_and_sum_to_one(self):
        value = torch.arange(5, dtype=torch.float).reshape(5, 1, 1)
        stages = value * torch.tensor([1., -.5, .25, .1]).reshape(1, 4, 1)
        report = stage_contributions(stages, torch.ones(2, 4), ["s0", "s1", "s2", "pooled", "target"])
        signed = report["source_target_pairs"]["signed_shares"]
        self.assertLess(signed[1], 0)
        self.assertGreater(signed[0], 1)
        self.assertAlmostEqual(sum(signed), 1)

    def test_teacher_preserves_classwise_routing_and_source_permutation(self):
        raw = torch.tensor([[.2, .9], [.7, .4]])
        image = F.normalize(raw, dim=-1)
        source_text = F.normalize(torch.tensor([[[1., .2], [.3, 1.]], [[.8, .4], [.6, .9]], [[.1, 1.], [1., .1]]]), dim=-1)
        centroids = torch.tensor([[[.1, 1.], [.9, .1]], [[.8, .2], [.2, .8]], [[.5, .5], [.6, .7]]])
        base = torch.eye(2); pooled = F.normalize(torch.tensor([[.8, .3], [.4, .9]]), dim=-1)
        scores, weights, _ = teacher_scores(image, raw, source_text, pooled, base, centroids, 3)
        oracle_distance = torch.stack([torch.cdist(raw, c).square() for c in centroids])
        torch.testing.assert_close(weights, torch.softmax(-3 * oracle_distance, dim=0))
        expected = torch.zeros(2, 2)
        for n in range(2):
            for k in range(2):
                for d in range(3):
                    expected[n, k] += weights[d, n, k] * torch.dot(image[n], source_text[d, k])
        torch.testing.assert_close(scores["weighted_source"], expected)
        torch.testing.assert_close(scores["combined"], (image @ base.T + image @ pooled.T + expected) / 3)
        swapped, _, _ = teacher_scores(image, raw, source_text.flip(0), pooled, base, centroids.flip(0), 3)
        torch.testing.assert_close(swapped["combined"], scores["combined"])


if __name__ == "__main__":
    unittest.main()
