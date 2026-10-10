import unittest
import numpy as np
import torch
from scripts.analyze_b0_centroids import verification_scores, candidate_centroid_selection, oracle_headroom, roc_curve
from scripts.analyze_style_reliability import auc


class CentroidDiagnosticTests(unittest.TestCase):
    def test_source_confidence_cannot_verify_class(self):
        # Both samples route overwhelmingly to source 0. The second teacher
        # candidate is nevertheless contradicted by a closer competing class.
        d = torch.tensor([[[1., 8.], [3., 9.], [5., 10.]],
                          [[8., 1.], [10., 3.], [12., 5.]]])
        weights = (-10 * d).softmax(1)
        p = torch.tensor([[.95, .05], [.95, .05]])
        scores = verification_scores(d, weights, torch.tensor([0, 0]), p)
        self.assertEqual(auc(scores["centroid_class_margin"], np.array([True, False])), 1.)
        self.assertEqual(auc(scores["teacher_confidence"], np.array([True, False])), .5)
        np.testing.assert_allclose(scores["source_weight_max_for_candidate"], [1., 1.])
        altered = d.double() + torch.tensor([300., 0.])[None, None, :]
        torch.testing.assert_close((-10*d.double()).softmax(1), (-10*altered).softmax(1), atol=1e-12, rtol=1e-12)
        self.assertFalse(torch.equal(d.min(1).values.argmin(1), altered.min(1).values.argmin(1)))

    def test_oracle_can_preserve_unique_combined_success(self):
        candidates = np.array([[1, 1, 1], [0, 1, 1], [1, 1, 1]])
        combined = np.array([0, 1, 1])
        labels = np.array([0, 0, 0])
        result = oracle_headroom(candidates, combined, labels, np.ones(3, bool))
        self.assertAlmostEqual(result["keep_combined_or_select_candidate_oracle_accuracy"], 2/3)
        self.assertEqual(result["recoverable_combined_errors"], 1)
        self.assertEqual(result["combined_correct_all_candidates_wrong"], 1)

    def test_candidate_selection_does_not_require_ground_truth(self):
        distance = torch.tensor([[5., 1., 3.], [1., 4., 2.]])
        candidates = torch.tensor([[0, 2, 0], [2, 1, 1]])
        self.assertEqual(candidate_centroid_selection(distance, candidates).tolist(), [2, 2])

    def test_tied_roc_matches_rank_auc(self):
        scores = np.array([1., 1., 2., 3., 3.])
        correct = np.array([False, True, False, True, True])
        roc = roc_curve(scores, correct)
        self.assertAlmostEqual(float(np.trapezoid(roc["tpr"], roc["fpr"])), auc(scores, correct))


if __name__ == "__main__":
    unittest.main()
