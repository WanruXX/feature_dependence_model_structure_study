from __future__ import annotations

import torch
from torch import Tensor, nn

from config import MODEL_VARIANTS, ModelConfig


class AdditiveCore(nn.Module):
    def __init__(self, n_features: int, hidden: int) -> None:
        super().__init__()
        self.input_weight = nn.Parameter(torch.empty(n_features, hidden))
        self.hidden_bias = nn.Parameter(torch.zeros(n_features, hidden))
        self.output_weight = nn.Parameter(torch.empty(n_features, hidden))
        self.output_bias = nn.Parameter(torch.zeros(n_features))
        nn.init.normal_(self.input_weight, std=0.5)
        nn.init.normal_(self.output_weight, std=0.05)

    def forward(self, features: Tensor) -> Tensor:
        hidden = torch.nn.functional.silu(
            features.unsqueeze(-1) * self.input_weight + self.hidden_bias
        )
        return (hidden * self.output_weight).sum(-1) + self.output_bias


class RedundancyGate(nn.Module):
    def __init__(self, scores: Tensor, config: ModelConfig, *, learned: bool) -> None:
        super().__init__()
        scores = scores.detach().float().clone()
        if scores.ndim != 1 or not bool(torch.isfinite(scores).all()) or bool(((scores < 0) | (scores > 1)).any()):
            raise ValueError("gate scores must be a finite vector in [0, 1]")
        self.register_buffer("scores", scores)
        self.register_buffer("prior", config.gate_floor + (1 - config.gate_floor)
                             * torch.exp(-config.gate_strength * scores))
        self.correction_bound = config.correction_bound
        if learned:
            self.correction_raw = nn.Parameter(torch.zeros_like(scores))
        else:
            self.register_buffer("correction_raw", torch.zeros_like(scores))

    def correction(self) -> Tensor:
        return (self.correction_bound * self.correction_raw.tanh()).exp()

    def forward(self) -> Tensor:
        return self.prior * self.correction()

    def regularization(self) -> Tensor:
        return (self.correction() - 1).square().mean()


class SmallResidual(nn.Module):
    def __init__(self, n_features: int, hidden: int) -> None:
        super().__init__()
        self.network = nn.Sequential(nn.Linear(n_features, hidden), nn.SiLU(), nn.Linear(hidden, 1))
        self.strength = nn.Parameter(torch.tensor(-2.0))
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, contributions: Tensor) -> Tensor:
        return self.strength.sigmoid() * self.network(contributions).squeeze(-1)


class AdditivePredictor(nn.Module):
    def __init__(self, n_features: int, config: ModelConfig, scores: Tensor | None,
                 *, learned_gate: bool = True, residual: bool = False,
                 low_rank: bool = False) -> None:
        super().__init__()
        self.core = AdditiveCore(n_features, config.feature_hidden)
        self.bias = nn.Parameter(torch.zeros(()))
        self.gate = None if scores is None else RedundancyGate(scores, config, learned=learned_gate)
        self.residual = SmallResidual(n_features, config.residual_hidden) if residual else None
        if low_rank:
            self.interaction_weight = nn.Parameter(torch.empty(n_features, config.interaction_rank))
            nn.init.normal_(self.interaction_weight, std=0.02)
        else:
            self.register_parameter("interaction_weight", None)

    def contributions(self, features: Tensor) -> Tensor:
        contributions = self.core(features)
        return contributions if self.gate is None else contributions * self.gate()

    def forward(self, features: Tensor) -> Tensor:
        contributions = self.contributions(features)
        logits = self.bias + contributions.sum(-1)
        if self.residual is not None:
            logits = logits + self.residual(contributions)
        if self.interaction_weight is not None:
            interaction = ((features @ self.interaction_weight).square()
                           - features.square() @ self.interaction_weight.square())
            logits = logits + 0.5 * interaction.sum(-1)
        return logits

    def regularization(self) -> Tensor:
        return self.bias.new_zeros(()) if self.gate is None else self.gate.regularization()


class MLP(nn.Module):
    def __init__(self, n_features: int, config: ModelConfig) -> None:
        super().__init__()
        self.network = nn.Sequential(nn.Linear(n_features, config.mlp_hidden),
                                     nn.SiLU(), nn.Linear(config.mlp_hidden, 1))

    def forward(self, features: Tensor) -> Tensor:
        return self.network(features).squeeze(-1)


class FeatureAttention(nn.Module):
    def __init__(self, n_features: int, config: ModelConfig) -> None:
        super().__init__()
        self.value_embedding = nn.Linear(1, config.attention_dim)
        self.feature_embedding = nn.Parameter(torch.empty(n_features, config.attention_dim))
        nn.init.normal_(self.feature_embedding, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=config.attention_dim, nhead=config.attention_heads,
            dim_feedforward=2 * config.attention_dim, dropout=0.0,
            activation="gelu", batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=1, enable_nested_tensor=False)
        self.output = nn.Linear(config.attention_dim, 1)

    def forward(self, features: Tensor) -> Tensor:
        tokens = self.value_embedding(features.unsqueeze(-1)) + self.feature_embedding
        return self.output(self.encoder(tokens).mean(1)).squeeze(-1)


def build_model(variant: str, n_features: int, config: ModelConfig,
                scores: dict[str, Tensor], *, seed: int = 42) -> nn.Module:
    if variant not in MODEL_VARIANTS:
        raise ValueError(f"unknown model variant: {variant}")
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        if variant == "mlp":
            return MLP(n_features, config)
        if variant == "attention":
            return FeatureAttention(n_features, config)
        prior = None
        if variant == "learned_gate":
            prior = torch.zeros(n_features)
        elif variant == "corr_gate":
            prior = scores["correlation"]
        elif variant in {"recon_gate", "shuffled_recon_gate", "fixed_recon_gate"}:
            prior = scores["reconstructability"]
        elif variant in {"redundancy_gate", "redundancy_residual", "fixed_redundancy_gate", "shuffled_gate"}:
            prior = scores["combined"]
        if variant in {"shuffled_gate", "shuffled_recon_gate"}:
            order = torch.randperm(n_features, generator=torch.Generator().manual_seed(seed + 23))
            prior = prior[order]
        if prior is not None and prior.shape != (n_features,):
            raise ValueError("one redundancy score per feature is required")
        return AdditivePredictor(
            n_features, config, prior,
            learned_gate=variant not in {"fixed_redundancy_gate", "fixed_recon_gate"},
            residual=variant in {"additive_residual", "redundancy_residual"},
            low_rank=variant == "low_rank",
        )