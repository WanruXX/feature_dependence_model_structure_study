from __future__ import annotations

from dataclasses import dataclass, replace
import math
from numbers import Integral


MODEL_VARIANTS = (
    "additive", "learned_gate", "corr_gate", "recon_gate", "redundancy_gate",
    "shuffled_gate", "fixed_redundancy_gate", "additive_residual",
    "redundancy_residual", "low_rank", "mlp", "attention",
    "shuffled_recon_gate", "fixed_recon_gate",
)
CONDITION_VARIANTS = ("additive", "recon_gate", "shuffled_recon_gate", "fixed_recon_gate", "low_rank", "mlp")


def integer(name: str, value: int, minimum: int = 0) -> None:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def finite(name: str, value: float, *, positive: bool = False) -> None:
    if not math.isfinite(value) or (value <= 0 if positive else value < 0):
        raise ValueError(f"invalid {name}")


def validate_data(config) -> None:
    for name in ("seed", "split_seed"):
        integer(name, getattr(config, name))
        if getattr(config, name) >= 2**32 - 1:
            raise ValueError("seed must be below 2**32 - 1")
    integer("n_features", config.n_features, 13)
    for name in ("train_rows", "validation_rows", "test_rows"):
        integer(name, getattr(config, name), 32)
    if config.scenario not in ("additive", "mixed", "interaction"):
        raise ValueError("unknown target scenario")
    if config.allocation_policy not in ("random", "source_prefix_fit_suffix_tune"):
        raise ValueError("unknown development allocation")


@dataclass(frozen=True)
class GaussianConfig:
    seed: int = 42
    split_seed: int = 42
    n_features: int = 55
    train_rows: int = 12000
    validation_rows: int = 4000
    test_rows: int = 4000
    group_size: int = 6
    correlation: float = 0.9
    scenario: str = "additive"
    signal_strength: float = 1.5
    interaction_strength: float = 1.5
    allocation_policy: str = "random"

    def __post_init__(self) -> None:
        validate_data(self)
        integer("group_size", self.group_size, 1)
        active = self.n_features - 1
        if active % self.group_size or active // self.group_size < 3:
            raise ValueError("active columns must form at least three complete groups")
        if not math.isfinite(self.correlation) or not 0 <= self.correlation <= 1:
            raise ValueError("correlation must lie in [0,1]")
        finite("signal_strength", self.signal_strength)
        finite("interaction_strength", self.interaction_strength)


@dataclass(frozen=True)
class StructuredConfig:
    seed: int = 42
    split_seed: int = 42
    n_features: int = 55
    train_rows: int = 12000
    validation_rows: int = 4000
    test_rows: int = 4000
    scenario: str = "additive"
    temporal_persistence: float = 0.8
    label_persistence: float = 0.92
    noise_scale: float = 0.22
    nonlinear_strength: float = 0.35
    positive_rate: float = 0.31
    signal_strength: float = 0.7
    interaction_strength: float = 3.0
    mixed_interaction_fraction: float = 0.2
    signal_location: str = "unique"
    allocation_policy: str = "random"

    def __post_init__(self) -> None:
        validate_data(self)
        if self.signal_location not in ("unique", "redundant"):
            raise ValueError("unknown signal location")
        for name in ("temporal_persistence", "label_persistence"):
            if not math.isfinite(getattr(self, name)) or not 0 <= getattr(self, name) < 1:
                raise ValueError(f"invalid {name}")
        for name in ("noise_scale", "signal_strength", "interaction_strength"):
            finite(name, getattr(self, name), positive=True)
        for name in ("nonlinear_strength", "mixed_interaction_fraction"):
            if not math.isfinite(getattr(self, name)) or not 0 <= getattr(self, name) <= 1:
                raise ValueError(f"invalid {name}")
        if not math.isfinite(self.positive_rate) or not 0 < self.positive_rate < 1:
            raise ValueError("invalid positive_rate")


@dataclass(frozen=True)
class ModelConfig:
    feature_hidden: int = 4
    residual_hidden: int = 8
    interaction_rank: int = 4
    mlp_hidden: int = 32
    attention_dim: int = 16
    attention_heads: int = 2
    gate_strength: float = 1.0
    gate_floor: float = 0.25
    correction_bound: float = 0.5
    reconstruction_ridge: float = 0.01

    def __post_init__(self) -> None:
        for name in ("feature_hidden", "residual_hidden", "interaction_rank", "mlp_hidden", "attention_dim", "attention_heads"):
            integer(name, getattr(self, name), 1)
        if self.attention_dim % self.attention_heads or not 0 < self.gate_floor <= 1:
            raise ValueError("invalid attention dimensions or gate floor")
        for name in ("gate_strength", "correction_bound"):
            finite(name, getattr(self, name))
        finite("reconstruction_ridge", self.reconstruction_ridge, positive=True)


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 300
    patience: int = 30
    batch_size: int = 256
    learning_rates: tuple[float, ...] = (0.001, 0.003)
    weight_decays: tuple[float, ...] = (0.0001,)
    gate_penalty: float = 0.001
    min_delta: float = 0.00001

    def __post_init__(self) -> None:
        for name in ("epochs", "patience", "batch_size"):
            integer(name, getattr(self, name), 1)
        if not self.learning_rates or not self.weight_decays:
            raise ValueError("empty tuning grid")
        for value in self.learning_rates:
            finite("learning_rate", value, positive=True)
        for value in (*self.weight_decays, self.gate_penalty, self.min_delta):
            finite("nonnegative training setting", value)


def selected_cases() -> dict[str, GaussianConfig | StructuredConfig]:
    gaussian, structured = GaussianConfig(), StructuredConfig()
    return {
        "gaussian_additive": gaussian,
        "gaussian_interaction": replace(gaussian, scenario="interaction"),
        **{f"structured_{scenario}": replace(structured, scenario=scenario)
           for scenario in ("additive", "mixed", "interaction")},
        "structured_independent_prefix": replace(structured, signal_location="redundant", label_persistence=0,
                                                  allocation_policy="source_prefix_fit_suffix_tune"),
        "structured_persistent_prefix": replace(structured, signal_location="redundant",
                                                 allocation_policy="source_prefix_fit_suffix_tune"),
    }


def case_models(name: str) -> tuple[str, ...]:
    return CONDITION_VARIANTS if name.endswith("_prefix") else MODEL_VARIANTS


def case_seeds(name: str) -> tuple[int, ...]:
    return (73, 74, 75) if name.endswith("_prefix") else (42,)
