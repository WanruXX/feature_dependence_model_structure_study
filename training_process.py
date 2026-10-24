from __future__ import annotations

from dataclasses import dataclass
import math
from time import perf_counter

import numpy as np
import torch
from torch import Tensor, nn

from config import TrainingConfig
from models import AdditivePredictor


@dataclass
class FitResult:
    model: nn.Module
    history: list[dict]
    best_epoch: int
    validation_loss: float
    seconds: float


def probability_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict:
    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    if labels.ndim != 1 or probabilities.shape != labels.shape or not len(labels):
        raise ValueError("expected nonempty aligned label/probability vectors")
    if not np.isin(labels, (0, 1)).all() or not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("expected binary labels and finite probabilities in [0, 1]")
    safe = np.clip(probabilities, np.finfo(float).eps, 1 - np.finfo(float).eps)
    positives, negatives = int(labels.sum()), int((1 - labels).sum())
    unique, inverse, counts = np.unique(probabilities, return_inverse=True, return_counts=True)
    ends = np.cumsum(counts)
    ranks = ((2 * ends - counts + 1) / 2)[inverse]
    auc = ((ranks[labels == 1].sum() - positives * (positives + 1) / 2) / (positives * negatives)
           if positives and negatives else None)
    return {"log_loss": float(-np.mean(labels * np.log(safe) + (1 - labels) * np.log1p(-safe))),
            "brier": float(np.mean((labels - probabilities) ** 2)),
            "auc": float(auc) if auc is not None else None,
            "accuracy": float(np.mean(labels == (probabilities >= 0.5))),
            "positive_fraction": float(labels.mean()), "mean_prediction": float(probabilities.mean())}


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def fit_model(model: nn.Module, train_features: Tensor, train_labels: Tensor,
              validation_features: Tensor, validation_labels: Tensor,
              config: TrainingConfig, *, learning_rate: float,
              weight_decay: float, seed: int, device: torch.device) -> FitResult:
    model = model.to(device)
    train_features, train_labels = train_features.to(device), train_labels.float().to(device)
    validation_features = validation_features.to(device)
    validation_labels = validation_labels.float().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    generator = torch.Generator().manual_seed(seed + 1000)
    criterion = nn.BCEWithLogitsLoss()
    best_loss, best_epoch, best_state = math.inf, 0, None
    history = []
    synchronize(device)
    started = perf_counter()
    for epoch in range(1, config.epochs + 1):
        model.train()
        order = torch.randperm(len(train_labels), generator=generator).to(device)
        total_data_loss, total_objective = 0.0, 0.0
        for batch in order.split(config.batch_size):
            optimizer.zero_grad(set_to_none=True)
            data_loss = criterion(model(train_features[batch]), train_labels[batch])
            penalty = model.regularization() if isinstance(model, AdditivePredictor) else data_loss.new_zeros(())
            objective = data_loss + config.gate_penalty * penalty
            if not bool(torch.isfinite(objective)):
                raise RuntimeError("nonfinite training objective; inspect data and settings")
            objective.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            total_data_loss += float(data_loss.detach()) * len(batch)
            total_objective += float(objective.detach()) * len(batch)
        model.eval()
        with torch.no_grad():
            total_validation = 0.0
            for start in range(0, len(validation_labels), config.batch_size):
                stop = start + config.batch_size
                labels = validation_labels[start:stop]
                total_validation += float(criterion(model(validation_features[start:stop]), labels)) * len(labels)
            validation_loss = total_validation / len(validation_labels)
        if not math.isfinite(validation_loss):
            raise RuntimeError("nonfinite validation loss")
        history.append({"epoch": epoch,
                        "train_log_loss": total_data_loss / len(train_labels),
                        "train_objective": total_objective / len(train_labels),
                        "validation_log_loss": validation_loss})
        if validation_loss < best_loss - config.min_delta:
            best_loss, best_epoch = validation_loss, epoch
            best_state = {name: value.detach().cpu().clone()
                          for name, value in model.state_dict().items()}
        if epoch - best_epoch >= config.patience:
            break
    synchronize(device)
    seconds = perf_counter() - started
    if best_state is None:
        raise RuntimeError("no valid checkpoint")
    model.load_state_dict(best_state)
    return FitResult(model.cpu().eval(), history, best_epoch, best_loss, seconds)


@torch.no_grad()
def predict(model: nn.Module, features: Tensor, *, device: torch.device,
            batch_size: int) -> tuple[Tensor, float]:
    model = model.to(device).eval()
    synchronize(device)
    started = perf_counter()
    probabilities = [model(batch.to(device)).sigmoid().cpu()
                     for batch in features.split(batch_size)]
    synchronize(device)
    elapsed = perf_counter() - started
    probability = torch.cat(probabilities)
    if not bool(torch.isfinite(probability).all()):
        raise RuntimeError("nonfinite predicted probability")
    return probability, elapsed