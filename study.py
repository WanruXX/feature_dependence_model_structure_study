from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, replace
from datetime import datetime, timezone
import itertools
import json
from pathlib import Path
import platform
import statistics

import numpy as np
import torch

from config import (MODEL_VARIANTS, GaussianConfig, StructuredConfig, ModelConfig,
                    TrainingConfig, selected_cases, case_models, case_seeds)
from datasets import prepare_dataset
from models import AdditivePredictor, build_model
from redundancy import compute_redundancy_scores
from training_process import fit_model, predict, probability_metrics


def metrics(labels, probability):
    values = probability_metrics(labels.cpu().numpy(), probability.cpu().double().numpy())
    return {"log_loss": values["log_loss"], "brier": values["brier"],
            "roc_auc": values["auc"], "accuracy": values["accuracy"]}


def run_experiment(data_config: GaussianConfig | StructuredConfig, *,
                   variants=MODEL_VARIANTS, model_config=None, training_config=None,
                   device="cpu", progress=True):
    if not variants or len(set(variants)) != len(variants) or not set(variants) <= set(MODEL_VARIANTS):
        raise ValueError("provide a nonempty unique selection of known models")
    model_config = ModelConfig() if model_config is None else model_config
    training_config = TrainingConfig() if training_config is None else training_config
    device = torch.device(device)
    data = prepare_dataset(data_config)
    scores = compute_redundancy_scores(data.features["train"], seed=data_config.seed,
                                       ridge=model_config.reconstruction_ridge)
    prior = float(data.labels["train"].float().mean())
    baselines = {split: metrics(data.labels[split], torch.full((len(data.labels[split]),), prior,
                                                             dtype=torch.float64))
                 for split in ("validation", "test")}
    rows, runs = [], []
    for variant in variants:
        if progress:
            print(f"seed={data_config.seed} | {variant}", flush=True)
        best_fit, best_setting, trials = None, None, []
        for learning_rate, weight_decay in itertools.product(training_config.learning_rates,
                                                             training_config.weight_decays):
            model = build_model(variant, data_config.n_features, model_config, scores,
                                seed=data_config.seed)
            fitted = fit_model(model, data.features["train"], data.labels["train"],
                               data.features["validation"], data.labels["validation"], training_config,
                               learning_rate=learning_rate, weight_decay=weight_decay,
                               seed=data_config.seed, device=device)
            setting = {"learning_rate": learning_rate, "weight_decay": weight_decay}
            trials.append({**setting, "best_epoch": fitted.best_epoch,
                           "validation_log_loss": fitted.validation_loss,
                           "training_seconds": fitted.seconds, "history": fitted.history})
            if best_fit is None or fitted.validation_loss < best_fit.validation_loss:
                best_fit, best_setting = fitted, setting
        predictions, evaluations = {}, {}
        for split in ("validation", "test"):
            probability, _ = predict(best_fit.model, data.features[split], device=device,
                                     batch_size=training_config.batch_size)
            predictions[split] = probability.tolist()
            evaluations[split] = metrics(data.labels[split], probability)
        rows.append({"variant": variant, "seed": data_config.seed,
                     "trainable_parameters": sum(parameter.numel() for parameter in best_fit.model.parameters()
                                                 if parameter.requires_grad),
                     "best_epoch": best_fit.best_epoch, "epochs_run": len(best_fit.history),
                     "search_training_seconds": sum(trial["training_seconds"] for trial in trials),
                     **best_setting,
                     **{f"{split}_{name}": value for split, values in evaluations.items()
                        for name, value in values.items()},
                     "test_excess_log_loss": evaluations["test"]["log_loss"] - baselines["test"]["log_loss"]})
        gate_details = None
        if isinstance(best_fit.model, AdditivePredictor) and best_fit.model.gate is not None:
            gate = best_fit.model.gate
            gate_details = {"scores": gate.scores.detach().cpu().tolist(),
                            "prior": gate.prior.detach().cpu().tolist(),
                            "correction": gate.correction().detach().cpu().tolist(),
                            "final": gate().detach().cpu().tolist()}
        runs.append({"variant": variant, "selected_hyperparameters": best_setting,
                     "trials": trials, "predictions": predictions, "gate": gate_details})
        best_fit.model.cpu()
    additive_loss = next((row["test_log_loss"] for row in rows if row["variant"] == "additive"), None)
    for row in rows:
        row["test_delta_vs_additive"] = None if additive_loss is None else row["test_log_loss"] - additive_loss
    return {"generator": "gaussian" if isinstance(data_config, GaussianConfig) else "structured",
            "data_config": asdict(data_config), "model_config": asdict(model_config),
            "training_config": asdict(training_config), "device": str(device),
            "row_ids": {name: ids.tolist() for name, ids in data.row_ids.items()},
            "labels": {name: labels.tolist() for name, labels in data.labels.items()},
            "scaling": {"mean": data.mean.tolist(), "scale": data.scale.tolist(), "fit_split": "train"},
            "redundancy": {name: values.tolist() for name, values in scores.items()},
            "baselines": baselines, "summary": rows, "runs": runs}


def summarize(rows):
    comparison = []
    for case, variant in sorted({(row["case"], row["variant"]) for row in rows}):
        selected = [row for row in rows if row["case"] == case and row["variant"] == variant]
        deltas = [row["test_delta_vs_additive"] for row in selected if row["test_delta_vs_additive"] is not None]
        comparison.append({"case": case, "variant": variant, "seeds": len(selected),
                           "mean_test_log_loss": statistics.mean(row["test_log_loss"] for row in selected),
                           "mean_delta_vs_additive": statistics.mean(deltas) if deltas else None,
                           "delta_seed_sd": statistics.stdev(deltas) if len(deltas) > 1 else None,
                           "seeds_better_than_additive": sum(delta < 0 for delta in deltas) if deltas else None,
                           "trainable_parameters": selected[0]["trainable_parameters"]})
    return comparison


def make_bundle(cases, *, device="cpu", threads=None):
    rows = [{"case": name, **row} for name, results in cases.items()
            for result in results for row in result["summary"]]
    return {"schema_version": 1, "study": "feature_dependence_model_structure_study",
            "environment": {"python": platform.python_version(), "numpy": np.__version__,
                            "torch": str(torch.__version__), "device": str(device),
                            "torch_threads": torch.get_num_threads() if threads is None else threads},
            "cases": cases, "summary": rows, "comparison": summarize(rows)}


def run_study(names=None, *, smoke=False, device="cpu", progress=True):
    catalog = selected_cases()
    names = tuple(catalog) if names is None else tuple(names)
    if not names or len(set(names)) != len(names) or not set(names) <= set(catalog):
        raise ValueError("provide unique selected case names")
    training = TrainingConfig(epochs=2, patience=1, learning_rates=(0.003,)) if smoke else TrainingConfig()
    cases = {}
    for name in names:
        config = catalog[name]
        if smoke:
            config = replace(config, train_rows=120, validation_rows=40, test_rows=40)
        cases[name] = []
        for seed in case_seeds(name)[:1] if smoke else case_seeds(name):
            if progress:
                print(f"\nCase: {name}", flush=True)
            cases[name].append(run_experiment(replace(config, seed=seed), variants=case_models(name),
                                              training_config=training, device=device, progress=progress))
    bundle = make_bundle(cases, device=device)
    bundle["protocol"] = "smoke" if smoke else "selected_study"
    return bundle


def export_results(bundle, output_root="results"):
    if not bundle.get("summary"):
        raise ValueError("there are no fitted results to export")
    serialized = json.dumps(bundle, indent=2, allow_nan=False)
    output = Path(output_root) / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output.mkdir(parents=True, exist_ok=False)
    (output / "results.json").write_text(serialized, encoding="utf-8")
    for name in ("summary", "comparison"):
        rows = bundle[name]
        with (output / f"{name}.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return output


def main():
    parser = argparse.ArgumentParser(description="Selected synthetic feature-dependence study")
    parser.add_argument("--cases", nargs="+", choices=tuple(selected_cases()))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--output-dir", default="results")
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("threads must be positive")
    torch.set_num_threads(args.threads)
    print(export_results(run_study(args.cases, smoke=args.smoke, device=args.device), args.output_dir))


if __name__ == "__main__":
    main()