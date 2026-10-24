from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from config import GaussianConfig, StructuredConfig


@dataclass
class Sample:
    features: np.ndarray
    labels: np.ndarray
    probability: np.ndarray
    positions: dict[str, np.ndarray]


@dataclass
class PreparedData:
    config: GaussianConfig | StructuredConfig
    features: dict[str, torch.Tensor]
    labels: dict[str, torch.Tensor]
    row_ids: dict[str, torch.Tensor]
    mean: torch.Tensor
    scale: torch.Tensor


def positions(config) -> dict[str, np.ndarray]:
    rows = config.train_rows + config.validation_rows + config.test_rows
    order = np.random.default_rng(config.split_seed).permutation(rows)
    boundary = config.train_rows + config.validation_rows
    if config.allocation_policy == "source_prefix_fit_suffix_tune":
        order[:boundary] = np.sort(order[:boundary])
    return {"train": order[:config.train_rows], "validation": order[config.train_rows:boundary],
            "test": order[boundary:]}


def persistent_normal(rows: int, width: int, persistence: float, generator: np.random.Generator) -> np.ndarray:
    innovations = generator.normal(size=(rows, width))
    values = innovations.copy()
    scale = np.sqrt(1 - persistence ** 2)
    for position in range(1, rows):
        values[position] = persistence * values[position - 1] + scale * innovations[position]
    return values


def normal_cdf(values: np.ndarray) -> np.ndarray:
    return (0.5 * (1 + torch.erf(torch.from_numpy(values) / np.sqrt(2)))).numpy()


def generate_gaussian(config: GaussianConfig) -> Sample:
    rows = config.train_rows + config.validation_rows + config.test_rows
    active = config.n_features - 1
    generator = torch.Generator().manual_seed(config.seed)
    groups = torch.randn(rows, active // config.group_size, generator=generator)
    noise = torch.randn(rows, active, generator=generator)
    group_index = torch.arange(active) // config.group_size
    topology = torch.Generator().manual_seed(config.seed + 31)
    loadings = torch.linspace(0.2, 1.0, active)[torch.randperm(active, generator=topology)]
    shared_variance = config.correlation * loadings
    features = shared_variance.sqrt() * groups[:, group_index] + (1 - shared_variance).sqrt() * noise
    first, second, third = features[:, 0], features[:, config.group_size], features[:, 2 * config.group_size]
    additive = 1.2 * first.tanh() - 0.8 * second.tanh() + 0.5 * (third.square() - 1)
    interaction = first * second
    logits = torch.zeros(rows)
    if config.scenario in ("additive", "mixed"):
        logits += config.signal_strength * additive
    if config.scenario in ("interaction", "mixed"):
        logits += config.interaction_strength * interaction
    probability = logits.sigmoid()
    labels = (torch.rand(rows, generator=generator) < probability).long()
    features = torch.cat((features, torch.full((rows, 1), 0.5)), dim=1)
    return Sample(features.numpy(), labels.numpy(), probability.numpy(), positions(config))


def generate_structured(config: StructuredConfig) -> Sample:
    rows = config.train_rows + config.validation_rows + config.test_rows
    width = config.n_features
    generator = np.random.default_rng(config.seed)
    normal = persistent_normal(rows, 11, config.temporal_persistence, generator)
    factors = normal.copy()
    factors[:, 0] = (normal[:, 0] + 0.5 * (normal[:, 0] ** 2 - 1)) / np.sqrt(1.5)
    noise = persistent_normal(rows, width - 1, config.temporal_persistence, generator)
    loadings = np.zeros((11, width - 1))
    for feature in range(3, width - 1):
        loadings[0, feature] = 0.45
        loadings[1 + feature % 8, feature] = 0.8
        loadings[1 + (feature + 1) % 8, feature] = 0.2 + 0.15 * (feature % 5) / 4
        loadings[9, feature] = 0.1 * np.cos(feature)
    values = factors @ loadings
    noise_scales = config.noise_scale * (0.35 + 1.3 * (np.arange(width - 1) % 7) / 6)
    values += noise * noise_scales
    values[:, :3] = noise[:, :3]
    nonlinear = factors[:, 0] * factors[:, 10]
    nonlinear_weights = np.zeros(width - 1)
    for feature in range(3, width - 1):
        nonlinear_weights[feature] = config.nonlinear_strength * (0.5 + 0.5 * np.sin(feature))
        values[:, feature] += nonlinear * nonlinear_weights[feature]
    source_scales = np.sqrt(np.sum(loadings ** 2, axis=0) + noise_scales ** 2 + nonlinear_weights ** 2)
    source_scales[:3] = 1
    values /= source_scales
    features = np.empty((rows, width), dtype=np.float32)
    features[:, :-1] = normal_cdf(values)
    for feature in range(width - 1):
        features[:, feature] **= 0.85 + 0.35 * (feature % 5) / 4
        features[:, feature] = np.round(features[:, feature], 2 if feature % 11 == 10 else 5)
    features[:, -1] = 0.5
    signal = [0, 1, 2] if config.signal_location == "unique" else [3, 4, 5]
    first, second, third = (2 * features[:, index].astype(float) - 1 for index in signal)
    additive = config.signal_strength * (2 * first - 1.5 * second + 0.8 * (third ** 2 - 1 / 3))
    interaction = config.interaction_strength * first * second
    logits = additive if config.scenario == "additive" else interaction
    if config.scenario == "mixed":
        logits = additive + config.mixed_interaction_fraction * interaction
    lower, upper = -40.0, 40.0
    for iteration in range(60):
        intercept = (lower + upper) / 2
        probability = 1 / (1 + np.exp(-np.clip(logits + intercept, -60, 60)))
        if probability.mean() < config.positive_rate:
            lower = intercept
        else:
            upper = intercept
    probability = 1 / (1 + np.exp(-np.clip(logits + (lower + upper) / 2, -60, 60)))
    label_noise = persistent_normal(rows, 1, config.label_persistence, np.random.default_rng(config.seed + 10000))
    labels = (normal_cdf(label_noise[:, 0]) < probability).astype(np.int64)
    return Sample(features, labels, probability, positions(config))


def generate(config: GaussianConfig | StructuredConfig) -> Sample:
    return generate_gaussian(config) if isinstance(config, GaussianConfig) else generate_structured(config)


def prepare_dataset(config: GaussianConfig | StructuredConfig) -> PreparedData:
    sample = generate(config)
    fitting = torch.from_numpy(sample.features[sample.positions["train"]])
    mean, scale = fitting.mean(0), fitting.std(0, unbiased=False).clamp_min(1e-6)
    return PreparedData(config,
                        {name: ((torch.from_numpy(sample.features[ids]) - mean) / scale).contiguous()
                         for name, ids in sample.positions.items()},
                        {name: torch.from_numpy(sample.labels[ids]) for name, ids in sample.positions.items()},
                        {name: torch.from_numpy(ids.copy()) for name, ids in sample.positions.items()}, mean, scale)
