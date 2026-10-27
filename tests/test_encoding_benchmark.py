import unittest

import numpy as np
from sklearn.linear_model import LogisticRegression

from encoding_benchmark import (
    build_pipeline,
    expected_calibration_error,
    multiclass_brier_score,
)


class EncodingBenchmarkTests(unittest.TestCase):
    def test_perfect_probabilities_have_zero_calibration_error(self):
        labels = np.array(["a", "b"])
        y = np.array(["a", "b"])
        probabilities = np.array([[1.0, 0.0], [0.0, 1.0]])
        self.assertEqual(expected_calibration_error(y, probabilities, labels), 0.0)
        self.assertEqual(multiclass_brier_score(y, probabilities, labels), 0.0)

    def test_one_hot_pipeline_uses_unknown_safe_encoder(self):
        pipeline = build_pipeline("one-hot", LogisticRegression(), scale=True)
        encoder = pipeline.named_steps["preprocess"].transformers[0][1]
        self.assertEqual(encoder.handle_unknown, "ignore")
        self.assertFalse(encoder.sparse_output)

    def test_ordinal_pipeline_does_not_use_label_encoder(self):
        pipeline = build_pipeline("ordinal", LogisticRegression(), scale=True)
        encoder = pipeline.named_steps["preprocess"].transformers[0][1]
        self.assertEqual(encoder.handle_unknown, "use_encoded_value")
        self.assertEqual(encoder.unknown_value, -1)


if __name__ == "__main__":
    unittest.main()

