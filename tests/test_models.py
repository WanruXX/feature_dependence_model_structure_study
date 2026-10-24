import importlib.util
import unittest

AVAILABLE = importlib.util.find_spec("torch") is not None
if AVAILABLE:
    import torch
    from config import MODEL_VARIANTS, ModelConfig
    from models import build_model


@unittest.skipUnless(AVAILABLE, "requires torch")
class ModelTests(unittest.TestCase):
    def setUp(self):
        self.config = ModelConfig()
        self.scores = {name: torch.linspace(0, 1, 12) for name in
                       ("correlation", "reconstructability", "combined")}
        self.features = torch.randn(8, 12, generator=torch.Generator().manual_seed(3))

    def test_all_models_have_finite_outputs_and_gradients(self):
        for variant in MODEL_VARIANTS:
            with self.subTest(variant=variant):
                model = build_model(variant, 12, self.config, self.scores)
                logits = model(self.features)
                self.assertEqual(logits.shape, (8,))
                self.assertTrue(bool(torch.isfinite(logits).all()))
                logits.square().mean().backward()
                self.assertTrue(all(parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
                                    for parameter in model.parameters()))

    def test_paired_initialization_and_zero_residual(self):
        plain = build_model("additive", 12, self.config, self.scores)
        free = build_model("learned_gate", 12, self.config, self.scores)
        residual = build_model("additive_residual", 12, self.config, self.scores)
        informed = build_model("redundancy_gate", 12, self.config, self.scores)
        informed_residual = build_model("redundancy_residual", 12, self.config, self.scores)
        torch.testing.assert_close(plain(self.features), free(self.features))
        torch.testing.assert_close(plain(self.features), residual(self.features))
        torch.testing.assert_close(informed(self.features), informed_residual(self.features))
        torch.testing.assert_close(plain.core.input_weight, informed.core.input_weight)
        self.assertEqual(float(informed.gate.regularization()), 0)

    def test_shuffled_control_and_parameter_cost(self):
        informed = build_model("redundancy_gate", 12, self.config, self.scores)
        shuffled = build_model("shuffled_gate", 12, self.config, self.scores)
        plain = build_model("additive", 12, self.config, self.scores)
        torch.testing.assert_close(informed.gate.prior.sort().values, shuffled.gate.prior.sort().values)
        self.assertFalse(torch.equal(informed.gate.prior, shuffled.gate.prior))
        count = lambda model: sum(parameter.numel() for parameter in model.parameters())
        self.assertEqual(count(informed) - count(plain), 12)

    def test_fixed_gate_is_additive_reparameterization(self):
        plain = build_model("additive", 12, self.config, self.scores)
        fixed = build_model("fixed_recon_gate", 12, self.config, self.scores)
        with torch.no_grad():
            gates = fixed.gate()
            fixed.core.output_weight.div_(gates[:, None])
            fixed.core.output_bias.div_(gates)
        torch.testing.assert_close(plain(self.features), fixed(self.features), atol=1e-7, rtol=1e-6)

    def test_pure_additive_has_no_mixed_input_effect(self):
        model = build_model("redundancy_gate", 12, self.config, self.scores)
        base = torch.zeros(1, 12)
        first, second, both = base.clone(), base.clone(), base.clone()
        first[:, 0] = 1
        second[:, 1] = 1
        both[:, :2] = 1
        difference = model(both) - model(first) - model(second) + model(base)
        torch.testing.assert_close(difference, torch.zeros_like(difference), atol=1e-6, rtol=0)

    def test_reconstruction_controls_preserve_prior_distribution(self):
        informed = build_model("recon_gate", 12, self.config, self.scores)
        shuffled = build_model("shuffled_recon_gate", 12, self.config, self.scores)
        fixed = build_model("fixed_recon_gate", 12, self.config, self.scores)
        torch.testing.assert_close(informed.gate.prior.sort().values, shuffled.gate.prior.sort().values)
        self.assertFalse(torch.equal(informed.gate.prior, shuffled.gate.prior))
        torch.testing.assert_close(informed(self.features), fixed(self.features))
        count = lambda model: sum(parameter.numel() for parameter in model.parameters())
        self.assertEqual(count(informed) - count(fixed), 12)


if __name__ == "__main__":
    unittest.main()