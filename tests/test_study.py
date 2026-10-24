from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from config import TrainingConfig, selected_cases, case_models
from study import export_results, make_bundle, run_experiment
from training_process import probability_metrics


class StudyTests(unittest.TestCase):
    def test_selected_models_validation_selection_and_exports(self):
        torch.set_num_threads(1)
        for name, config in selected_cases().items():
            config = replace(config, train_rows=64, validation_rows=32, test_rows=32)
            result = run_experiment(config, variants=case_models(name), progress=False,
                                    training_config=TrainingConfig(epochs=2, patience=1))
            self.assertEqual(len(result["summary"]), len(case_models(name)))
            for row, run in zip(result["summary"], result["runs"]):
                selected = min(run["trials"], key=lambda trial: trial["validation_log_loss"])
                self.assertEqual(row["learning_rate"], selected["learning_rate"])
                for split in ("validation", "test"):
                    observed = probability_metrics(np.asarray(result["labels"][split]),
                                                   np.asarray(run["predictions"][split]))
                    self.assertAlmostEqual(row[f"{split}_log_loss"], observed["log_loss"], places=12)
                self.assertTrue(all(np.isfinite(row[key]) for key in ("test_log_loss", "validation_log_loss")))
            bundle = make_bundle({name: [result]})
            with tempfile.TemporaryDirectory() as directory:
                output = export_results(bundle, directory)
                self.assertEqual(json.loads((output / "results.json").read_text())["summary"], bundle["summary"])
                self.assertTrue((output / "comparison.csv").exists())
                self.assertTrue((output / "summary.csv").exists())
            from presentation import comparison_markdown
            table = comparison_markdown(bundle)
            self.assertIn(name, table)
            for row in bundle["comparison"]:
                self.assertIn(f"{row['mean_test_log_loss']:.6f}", table)

    def test_parameter_counts_match_between_generators(self):
        from models import build_model
        from config import ModelConfig, MODEL_VARIANTS
        counts = {}
        for name in ("gaussian_additive", "structured_additive"):
            config = selected_cases()[name]
            counts[name] = {variant: sum(value.numel() for value in
                                        build_model(variant, config.n_features, ModelConfig(),
                                                    {key: torch.zeros(config.n_features) for key in
                                                     ("correlation", "reconstructability", "combined")},
                                                    seed=config.seed).parameters() if value.requires_grad)
                            for variant in MODEL_VARIANTS}
        self.assertEqual(counts["gaussian_additive"], counts["structured_additive"])


if __name__ == "__main__":
    unittest.main()