"""Evaluation and experiment-logging helpers for ChestMNIST multi-label models.

Expected targets/logits shape: (number_of_images, number_of_labels).
This module deliberately does not train a model or choose a checkpoint for you.
Call collect_logits() during validation, compute mean AUROC/AP, and use that
metric in your training loop to decide which checkpoint to save.

Dependencies: numpy, torch, scikit-learn.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import KFold


LABEL_NAMES = [
    "atelectasis",
    "cardiomegaly",
    "effusion",
    "infiltration",
    "mass",
    "nodule",
    "pneumonia",
    "pneumothorax",
    "consolidation",
    "edema",
    "emphysema",
    "fibrosis",
    "pleural",
    "hernia",
]


def _as_2d_array(values: Any, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a 2D array; received shape {array.shape}.")
    return array


def _sigmoid(logits: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid for NumPy arrays."""
    clipped = np.clip(np.asarray(logits, dtype=np.float64), -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def collect_logits(
    model: torch.nn.Module,
    dataloader: Iterable,
    device: torch.device | str,
) -> tuple[np.ndarray, np.ndarray]:
    """Collect raw model logits and targets from a (images, labels) dataloader.

    The model is put in eval mode. Each batch must contain images and labels as
    its first two elements. Returned arrays are float32 and have shape (N, C).
    """
    model.eval()
    all_logits: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []

    with torch.no_grad():
        for batch_index, batch in enumerate(dataloader):
            if not isinstance(batch, (tuple, list)) or len(batch) < 2:
                raise ValueError(
                    "Each dataloader batch must be a tuple/list containing images and labels."
                )

            images, labels = batch[0], batch[1]
            images = images.to(device, non_blocking=True)
            outputs = model(images)
            if isinstance(outputs, (tuple, list)):
                outputs = outputs[0]

            if not torch.is_tensor(outputs):
                raise TypeError("The model must return a tensor of logits.")

            all_logits.append(outputs.detach().float().cpu().numpy())
            all_targets.append(torch.as_tensor(labels).detach().float().cpu().numpy())

    if not all_logits:
        raise ValueError("The dataloader produced no batches.")

    logits = np.concatenate(all_logits, axis=0).astype(np.float32, copy=False)
    targets = np.concatenate(all_targets, axis=0).astype(np.float32, copy=False)

    if logits.shape != targets.shape:
        raise ValueError(
            f"Logit and target shapes must match; got {logits.shape} and {targets.shape}."
        )
    return logits, targets


def _expand_thresholds(thresholds: Any, n_samples: int, n_labels: int) -> np.ndarray:
    values = np.asarray(thresholds, dtype=np.float64)
    if values.ndim == 0:
        return np.full((n_samples, n_labels), float(values))
    if values.shape == (n_labels,):
        return np.broadcast_to(values.reshape(1, -1), (n_samples, n_labels))
    if values.shape == (n_samples, n_labels):
        return values
    raise ValueError(
        "thresholds must be a scalar, a vector with one value per label, "
        "or a matrix matching the targets."
    )


def evaluate_probabilities(
    targets: Any,
    probabilities: Any,
    thresholds: Any = 0.5,
    label_names: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Calculate threshold-free ranking metrics and threshold-based F1 metrics.

    `thresholds` may be one scalar, one threshold per class, or a per-sample
    threshold matrix (the latter is useful for cross-fitted evaluation).
    Classes with only one target value have undefined AUROC and are excluded
    from the macro AUROC mean.
    """
    y_true = _as_2d_array(targets, "targets").astype(np.uint8)
    probs = _as_2d_array(probabilities, "probabilities").astype(np.float64)
    if y_true.shape != probs.shape:
        raise ValueError(f"Shape mismatch: targets {y_true.shape}, probabilities {probs.shape}.")

    n_samples, n_labels = y_true.shape
    names = list(label_names) if label_names is not None else [f"label_{i}" for i in range(n_labels)]
    if len(names) != n_labels:
        raise ValueError(f"Expected {n_labels} label names, received {len(names)}.")

    threshold_matrix = _expand_thresholds(thresholds, n_samples, n_labels)
    y_pred = (probs >= threshold_matrix).astype(np.uint8)

    per_class: list[dict[str, Any]] = []
    auc_values: list[float] = []
    ap_values: list[float] = []

    for index, name in enumerate(names):
        truth = y_true[:, index]
        prediction = y_pred[:, index]
        score = probs[:, index]
        precision, recall, f1, support = precision_recall_fscore_support(
            truth, prediction, average="binary", zero_division=0
        )

        auc = float("nan")
        if np.unique(truth).size == 2:
            auc = float(roc_auc_score(truth, score))
            auc_values.append(auc)

        ap = float("nan")
        positive_count = int(truth.sum())
        prevalence = float(truth.mean())
        if positive_count > 0:
            ap = float(average_precision_score(truth, score))
            ap_values.append(ap)

        per_class.append({
            "label": name,
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "support": int(truth.sum()),
            "prevalence": prevalence,
            "auroc": auc,
            "average_precision": ap,
            "ap_over_prevalence": float(ap / prevalence) if prevalence > 0 and np.isfinite(ap) else float("nan"),
            "threshold_mean": float(threshold_matrix[:, index].mean()),
        })

    micro_p, micro_r, micro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="micro", zero_division=0
    )
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    return {
        "n_samples": int(n_samples),
        "n_labels": int(n_labels),
        "micro_precision": float(micro_p),
        "micro_recall": float(micro_r),
        "micro_f1": float(micro_f1),
        "macro_precision": float(macro_p),
        "macro_recall": float(macro_r),
        "macro_f1": float(macro_f1),
        "weighted_precision": float(weighted_p),
        "weighted_recall": float(weighted_r),
        "weighted_f1": float(weighted_f1),
        "macro_auroc": float(np.mean(auc_values)) if auc_values else float("nan"),
        "mean_average_precision": float(np.mean(ap_values)) if ap_values else float("nan"),
        "per_class": per_class,
    }


def find_best_thresholds(
    targets: Any,
    probabilities: Any,
    threshold_grid: Sequence[float] | None = None,
    label_names: Sequence[str] | None = None,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Find one F1-maximizing threshold per label on the supplied data.

    IMPORTANT: Do not report F1 on the same examples used here as an unbiased
    result. Use crossfit_threshold_evaluation() for a less optimistic validation
    estimate, then fit final thresholds on all validation data for later use.
    """
    y_true = _as_2d_array(targets, "targets").astype(np.uint8)
    probs = _as_2d_array(probabilities, "probabilities").astype(np.float64)
    if y_true.shape != probs.shape:
        raise ValueError("targets and probabilities must have identical shapes.")

    grid = np.asarray(
        threshold_grid if threshold_grid is not None else np.arange(0.05, 0.96, 0.05),
        dtype=np.float64,
    )
    if grid.ndim != 1 or grid.size == 0 or np.any((grid <= 0) | (grid >= 1)):
        raise ValueError("threshold_grid must be a non-empty 1D sequence strictly between 0 and 1.")

    n_labels = y_true.shape[1]
    names = list(label_names) if label_names is not None else [f"label_{i}" for i in range(n_labels)]
    if len(names) != n_labels:
        raise ValueError(f"Expected {n_labels} label names, received {len(names)}.")

    best_thresholds = np.full(n_labels, 0.5, dtype=np.float64)
    details: list[dict[str, Any]] = []
    for index, name in enumerate(names):
        truth = y_true[:, index]
        if truth.sum() == 0 or truth.sum() == len(truth):
            details.append({"label": name, "threshold": 0.5, "best_f1_on_fit_data": float("nan"), "status": "only_one_target_class"})
            continue

        scores = [f1_score(truth, probs[:, index] >= threshold, zero_division=0) for threshold in grid]
        max_score = max(scores)
        tied_indices = [i for i, score in enumerate(scores) if np.isclose(score, max_score)]
        # Deterministic tie-break: choose the threshold closest to 0.5.
        best_index = min(tied_indices, key=lambda i: abs(float(grid[i]) - 0.5))
        best_thresholds[index] = grid[best_index]
        details.append({
            "label": name,
            "threshold": float(best_thresholds[index]),
            "best_f1_on_fit_data": float(scores[best_index]),
            "status": "ok",
        })

    return best_thresholds, details


def crossfit_threshold_evaluation(
    targets: Any,
    probabilities: Any,
    n_splits: int = 2,
    random_state: int = 42,
    threshold_grid: Sequence[float] | None = None,
    label_names: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Evaluate F1 with thresholds selected on other validation folds.

    A threshold is fitted on all folds except one and applied to the held-out
    fold. This reduces the optimism caused by selecting and evaluating a
    threshold on the exact same validation examples. It does not replace an
    independent test set or correct any split-level data leakage.
    """
    y_true = _as_2d_array(targets, "targets").astype(np.uint8)
    probs = _as_2d_array(probabilities, "probabilities").astype(np.float64)
    if y_true.shape != probs.shape:
        raise ValueError("targets and probabilities must have identical shapes.")
    if n_splits < 2 or n_splits > len(y_true):
        raise ValueError("n_splits must be between 2 and the number of samples.")

    threshold_matrix = np.full(probs.shape, 0.5, dtype=np.float64)
    fold_thresholds: list[list[float]] = []
    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    for fit_indices, heldout_indices in splitter.split(y_true):
        fold_thresholds_array, _ = find_best_thresholds(
            y_true[fit_indices],
            probs[fit_indices],
            threshold_grid=threshold_grid,
            label_names=label_names,
        )
        threshold_matrix[heldout_indices, :] = fold_thresholds_array.reshape(1, -1)
        fold_thresholds.append(fold_thresholds_array.tolist())

    metrics = evaluate_probabilities(
        y_true,
        probs,
        thresholds=threshold_matrix,
        label_names=label_names,
    )
    return {
        "method": "KFold cross-fitted per-class thresholds",
        "n_splits": int(n_splits),
        "random_state": int(random_state),
        "fold_thresholds": fold_thresholds,
        "metrics": metrics,
    }


def _bootstrap_score(y: np.ndarray, p: np.ndarray, metric: str) -> float:
    if metric == "micro_auroc":
        flat_y, flat_p = y.reshape(-1), p.reshape(-1)
        return float(roc_auc_score(flat_y, flat_p)) if np.unique(flat_y).size == 2 else float("nan")
    if metric == "micro_ap":
        flat_y, flat_p = y.reshape(-1), p.reshape(-1)
        return float(average_precision_score(flat_y, flat_p)) if flat_y.sum() > 0 else float("nan")

    values: list[float] = []
    for class_index in range(y.shape[1]):
        truth = y[:, class_index]
        if metric == "macro_auroc":
            if np.unique(truth).size < 2:
                continue
            values.append(float(roc_auc_score(truth, p[:, class_index])))
        elif metric == "macro_ap":
            if truth.sum() == 0:
                continue
            values.append(float(average_precision_score(truth, p[:, class_index])))
        else:
            raise ValueError("metric must be macro_auroc, macro_ap, micro_auroc, or micro_ap.")
    return float(np.mean(values)) if values else float("nan")


def paired_bootstrap_difference(
    targets: Any,
    probabilities_a: Any,
    probabilities_b: Any,
    metric: str = "macro_auroc",
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    random_state: int = 42,
) -> dict[str, Any]:
    """Paired image-level bootstrap for metric(A) - metric(B).

    Both models are evaluated on exactly the same resampled image indices.
    For macro AUROC/AP, a class is omitted from a replicate if that replicate
    contains no positive or (for AUROC) no negative examples for that class.
    """
    y = _as_2d_array(targets, "targets").astype(np.uint8)
    pa = _as_2d_array(probabilities_a, "probabilities_a").astype(np.float64)
    pb = _as_2d_array(probabilities_b, "probabilities_b").astype(np.float64)
    if y.shape != pa.shape or y.shape != pb.shape:
        raise ValueError("targets and both probability arrays must have identical shapes.")
    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be at least 1.")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1.")

    point_a = _bootstrap_score(y, pa, metric)
    point_b = _bootstrap_score(y, pb, metric)
    point_difference = point_a - point_b

    rng = np.random.default_rng(random_state)
    differences: list[float] = []
    n_samples = len(y)
    for _ in range(n_bootstrap):
        indices = rng.integers(0, n_samples, size=n_samples)
        score_a = _bootstrap_score(y[indices], pa[indices], metric)
        score_b = _bootstrap_score(y[indices], pb[indices], metric)
        difference = score_a - score_b
        if np.isfinite(difference):
            differences.append(float(difference))

    if not differences:
        lower = upper = float("nan")
    else:
        alpha = 1.0 - confidence
        lower, upper = np.quantile(differences, [alpha / 2, 1 - alpha / 2]).tolist()

    return {
        "metric": metric,
        "model_a_score": float(point_a),
        "model_b_score": float(point_b),
        "difference_a_minus_b": float(point_difference),
        "confidence": float(confidence),
        "ci_lower": float(lower),
        "ci_upper": float(upper),
        "valid_bootstrap_replicates": int(len(differences)),
        "requested_bootstrap_replicates": int(n_bootstrap),
        "random_state": int(random_state),
    }


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable.")


def save_run_artifacts(
    output_dir: str | Path,
    run_name: str,
    config: dict[str, Any],
    validation_logits: Any,
    validation_targets: Any,
    metrics: dict[str, Any] | None = None,
    thresholds: Any | None = None,
    training_history: dict[str, Any] | None = None,
) -> tuple[Path, Path]:
    """Save raw validation logits/targets in NPZ and run metadata in JSON.

    Save raw logits rather than only binary predictions. This lets you recompute
    probabilities, metrics, thresholds, calibration, and ensembles later without
    running inference again.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logits = _as_2d_array(validation_logits, "validation_logits").astype(np.float32)
    targets = _as_2d_array(validation_targets, "validation_targets").astype(np.float32)
    if logits.shape != targets.shape:
        raise ValueError("validation_logits and validation_targets must have identical shapes.")

    npz_path = out_dir / f"{run_name}_validation_outputs.npz"
    arrays: dict[str, np.ndarray] = {
        "validation_logits": logits,
        "validation_targets": targets,
    }
    if thresholds is not None:
        arrays["thresholds"] = np.asarray(thresholds, dtype=np.float32)
    np.savez_compressed(npz_path, **arrays)

    json_path = out_dir / f"{run_name}_metadata.json"
    payload = {
        "run_name": run_name,
        "config": config,
        "metrics": metrics,
        "training_history": training_history,
        "validation_outputs_file": npz_path.name,
        "note": "Validation outputs are raw logits and targets; thresholds/metrics can be recomputed.",
    }
    json_path.write_text(json.dumps(payload, indent=2, default=_json_default, allow_nan=True), encoding="utf-8")
    return npz_path, json_path


def save_training_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    epoch: int,
    best_metric: float,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    config: dict[str, Any] | None = None,
    history: dict[str, Any] | None = None,
) -> Path:
    """Save a resumable checkpoint (model + optional optimizer/scheduler state)."""
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "epoch": int(epoch),
        "best_metric": float(best_metric),
        "model_state_dict": model.state_dict(),
        "config": config or {},
        "history": history or {},
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler_state_dict"] = scheduler.state_dict()
    torch.save(payload, checkpoint_path)
    return checkpoint_path
