from dataclasses import replace
import unittest

import numpy as np
import torch

from config import GaussianConfig, StructuredConfig, selected_cases
from datasets import generate, prepare_dataset


class DataTests(unittest.TestCase):
    def test_catalog_width_and_pairing(self):
        self.assertEqual(len(selected_cases()), 7)
        self.assertTrue(all(config.n_features == 55 for config in selected_cases().values()))
        for constructor in (GaussianConfig, StructuredConfig):
            config = constructor(train_rows=120, validation_rows=40, test_rows=40)
            sample = generate(config)
            self.assertEqual(sample.features.shape, (200, 55))
            self.assertEqual(int((sample.features.std(0) > 1e-8).sum()), 54)
            self.assertTrue(np.all(sample.features[:, -1] == 0.5))
            other = generate(replace(config, scenario="interaction"))
            np.testing.assert_array_equal(sample.features, other.features)
            for split in sample.positions:
                np.testing.assert_array_equal(sample.positions[split], other.positions[split])
            self.assertFalse(np.array_equal(sample.labels, other.labels))
            prepared = prepare_dataset(config)
            expected = torch.from_numpy(sample.features[sample.positions["train"]])
            torch.testing.assert_close(prepared.mean, expected.mean(0))
            torch.testing.assert_close(prepared.scale, expected.std(0, unbiased=False).clamp_min(1e-6))

    def test_allocation_and_label_dependence_invariants(self):
        config = StructuredConfig(train_rows=120, validation_rows=40, test_rows=40)
        sample = generate(config)
        prefix = generate(replace(config, allocation_policy="source_prefix_fit_suffix_tune"))
        independent = generate(replace(config, label_persistence=0))
        np.testing.assert_array_equal(sample.features, prefix.features)
        np.testing.assert_array_equal(sample.labels, prefix.labels)
        np.testing.assert_array_equal(sample.positions["test"], prefix.positions["test"])
        np.testing.assert_array_equal(sample.probability, independent.probability)
        self.assertLess(prefix.positions["train"].max(), prefix.positions["validation"].min())
        for current in (sample, prefix):
            ids = np.concatenate(list(current.positions.values()))
            self.assertEqual(len(np.unique(ids)), 200)
            self.assertTrue(np.isfinite(current.features).all())

    def test_configuration_rejects_invalid_values(self):
        for values in ({"n_features": 54}, {"correlation": float("nan")}, {"seed": True}):
            with self.assertRaises(ValueError):
                GaussianConfig(**values)
        with self.assertRaises(ValueError):
            StructuredConfig(label_persistence=1)


if __name__ == "__main__":
    unittest.main()