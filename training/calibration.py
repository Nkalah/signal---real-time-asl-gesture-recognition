"""Validation-only probability calibration and rejection threshold selection."""

from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F


def expected_calibration_error(
    logits: np.ndarray,
    targets: np.ndarray,
    bins: int = 15,
    temperature: float = 1.0,
) -> float:
    logits_tensor = torch.as_tensor(logits, dtype=torch.float64)
    target_tensor = torch.as_tensor(targets, dtype=torch.long)
    probabilities = torch.softmax(logits_tensor / temperature, dim=1)
    confidences, predictions = probabilities.max(dim=1)
    correct = predictions.eq(target_tensor)
    edges = torch.linspace(0.0, 1.0, bins + 1, dtype=torch.float64)

    ece = torch.tensor(0.0, dtype=torch.float64)
    for index in range(bins):
        in_bin = (confidences > edges[index]) & (confidences <= edges[index + 1])
        if in_bin.any():
            accuracy = correct[in_bin].double().mean()
            confidence = confidences[in_bin].mean()
            ece += in_bin.double().mean() * torch.abs(accuracy - confidence)
    return float(ece.item())


def fit_temperature(logits: np.ndarray, targets: np.ndarray) -> float:
    logits_tensor = torch.as_tensor(logits, dtype=torch.float64)
    target_tensor = torch.as_tensor(targets, dtype=torch.long)
    log_temperature = torch.nn.Parameter(torch.zeros((), dtype=torch.float64))
    optimizer = torch.optim.LBFGS(
        [log_temperature],
        lr=0.1,
        max_iter=100,
        line_search_fn="strong_wolfe",
    )

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        temperature = log_temperature.exp()
        loss = F.cross_entropy(logits_tensor / temperature, target_tensor)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(log_temperature.detach().exp().clamp(min=1e-3, max=1e3).item())


def _select_rejection_thresholds(
    probabilities: np.ndarray,
    targets: np.ndarray,
    labels: List[str],
) -> Dict[str, float]:
    top_indices = probabilities.argmax(axis=1)
    sorted_probabilities = np.sort(probabilities, axis=1)
    confidences = sorted_probabilities[:, -1]
    margins = (
        confidences - sorted_probabilities[:, -2]
        if probabilities.shape[1] > 1
        else confidences
    )
    none_index = labels.index("NONE") if "NONE" in labels else None
    true_none = targets == none_index if none_index is not None else np.zeros_like(targets, dtype=bool)
    correctly_signed = (top_indices == targets) & ~true_none
    if not correctly_signed.any():
        raise ValueError(
            "Validation data contains no correctly classified known signs; "
            "cannot calibrate a usable rejection rule."
        )
    if (~correctly_signed).sum() == 0:
        return {
            "max_probability_below": 0.0,
            "top_two_margin_below": 0.0,
            "validation_balanced_acceptance_accuracy": 1.0,
            "validation_correct_sign_coverage": 1.0,
            "validation_overall_coverage": 1.0,
        }

    confidence_candidates = np.unique(np.quantile(confidences, np.linspace(0, 1, 21)))
    margin_candidates = np.unique(np.quantile(margins, np.linspace(0, 1, 21)))
    best: tuple[float, float, float, float, float] | None = None
    for confidence_threshold in confidence_candidates:
        for margin_threshold in margin_candidates:
            accepted = (
                (confidences >= confidence_threshold)
                & (margins >= margin_threshold)
                & (top_indices != none_index if none_index is not None else True)
            )
            accepted_correct = accepted & correctly_signed
            rejected_incorrect = ~accepted & ~correctly_signed
            balanced_accuracy = 0.5 * (
                accepted_correct.sum() / correctly_signed.sum()
                + rejected_incorrect.sum() / (~correctly_signed).sum()
            )
            coverage = accepted.mean()
            correct_coverage = accepted_correct.sum() / correctly_signed.sum()
            candidate = (
                float(balanced_accuracy),
                float(correct_coverage),
                float(coverage),
                float(confidence_threshold),
                float(margin_threshold),
            )
            if best is None or candidate[:3] > best[:3]:
                best = candidate

    if best is None:
        raise RuntimeError("Unable to select rejection thresholds from validation data.")
    return {
        "max_probability_below": best[3],
        "top_two_margin_below": best[4],
        "validation_balanced_acceptance_accuracy": best[0],
        "validation_correct_sign_coverage": best[1],
        "validation_overall_coverage": best[2],
    }


def calibrate_validation(
    logits: np.ndarray,
    targets: np.ndarray,
    labels: List[str],
) -> Dict[str, object]:
    """Fit temperature and rejection thresholds using validation samples only."""
    temperature = fit_temperature(logits, targets)
    probabilities = torch.softmax(
        torch.as_tensor(logits, dtype=torch.float64) / temperature,
        dim=1,
    ).numpy()
    thresholds = _select_rejection_thresholds(probabilities, targets, labels)
    return {
        "temperature": temperature,
        "ece_before": expected_calibration_error(logits, targets),
        "ece_after": expected_calibration_error(logits, targets, temperature=temperature),
        "rejection": {
            **thresholds,
            "selection_method": (
                "Validation grid search maximizing balanced accuracy for accepting "
                "correct known signs and rejecting incorrect or NONE predictions; "
                "ties prefer correct-sign coverage, then overall coverage."
            ),
            "none_label": "NONE",
        },
    }
