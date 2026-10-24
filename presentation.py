from __future__ import annotations

import numpy as np

from config import MODEL_VARIANTS
from datasets import generate


def comparison_markdown(bundle):
    names = list(bundle["cases"])
    lookup = {(row["case"], row["variant"]): row for row in bundle["comparison"]}
    lines = ["| Model | " + " | ".join(names) + " |",
             "| --- | " + " | ".join("---:" for name in names) + " |"]
    for variant in MODEL_VARIANTS:
        values = [f"{lookup[name, variant]['mean_test_log_loss']:.6f}"
                  if (name, variant) in lookup else "-" for name in names]
        lines.append(f"| {variant} | " + " | ".join(values) + " |")
    return "\n".join(lines)


def plot_features(config):
    import matplotlib.pyplot as plt

    sample = generate(config)
    correlation = np.zeros((config.n_features, config.n_features))
    correlation[:-1, :-1] = np.corrcoef(sample.features[:, :-1], rowvar=False)
    figure, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
    image = axes[0].imshow(correlation, vmin=-1, vmax=1, cmap="coolwarm", interpolation="nearest")
    axes[0].set(title="Feature correlation (constant = 0)", xlabel="Feature", ylabel="Feature")
    figure.colorbar(image, ax=axes[0], shrink=0.8)
    for feature in (0, 3, 30):
        axes[1].hist(sample.features[:, feature], bins=35, density=True,
                     histtype="step", label=f"Feature {feature}")
    axes[1].set(title="Selected marginal distributions", xlabel="Value", ylabel="Density")
    axes[1].legend()
    axes[2].hist(sample.probability, bins=35, color="#267d78")
    axes[2].set(title="Target probabilities", xlabel="Probability", ylabel="Rows")
    return figure


def plot_comparison(bundle):
    import matplotlib.pyplot as plt

    count = len(bundle["cases"])
    figure, axes = plt.subplots((count + 1) // 2, 2, figsize=(15, 4 * ((count + 1) // 2)),
                                squeeze=False, constrained_layout=True)
    for axis, name in zip(axes.flat, bundle["cases"]):
        rows = sorted((row for row in bundle["comparison"] if row["case"] == name),
                      key=lambda row: row["mean_delta_vs_additive"])
        offsets = np.arange(len(rows))
        axis.barh(offsets, [row["mean_delta_vs_additive"] for row in rows],
                   color=["#267d78" if row["mean_delta_vs_additive"] < 0 else "#b96a44" for row in rows])
        for position, row in enumerate(rows):
            individual = [value["test_delta_vs_additive"] for result in bundle["cases"][name]
                          for value in result["summary"] if value["variant"] == row["variant"]]
            axis.scatter(individual, np.full(len(individual), position), s=12, color="#252525", zorder=3)
        axis.set_yticks(offsets, [row["variant"] for row in rows])
        axis.invert_yaxis()
        axis.axvline(0, color="#555555", linewidth=0.8)
        axis.set(title=name, xlabel="Test log-loss difference from additive (lower is better)")
    for axis in list(axes.flat)[count:]:
        axis.set_visible(False)
    return figure


def plot_history(bundle, case="gaussian_interaction", variant="attention", seed_index=0):
    import matplotlib.pyplot as plt

    result = bundle["cases"][case][seed_index]
    run = next(run for run in result["runs"] if run["variant"] == variant)
    figure, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    for trial in run["trials"]:
        epochs = [row["epoch"] for row in trial["history"]]
        label = f"lr={trial['learning_rate']}, wd={trial['weight_decay']}"
        for axis, key in zip(axes, ("train_log_loss", "validation_log_loss")):
            axis.plot(epochs, [row[key] for row in trial["history"]], label=label)
            axis.axvline(trial["best_epoch"], linestyle=":", alpha=0.4)
            axis.set(xlabel="Epoch", ylabel=key.replace("_", " "))
            axis.legend()
    figure.suptitle(f"{case}: {variant}, seed={result['data_config']['seed']}")
    return figure