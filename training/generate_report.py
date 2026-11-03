"""Generate the real-data training report and held-out confusion matrix."""

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_DIR = ROOT_DIR / "backend" / "models"
DEFAULT_REPORT = ROOT_DIR / "docs" / "TRAINING_REPORT.md"


def _read_json(path: Path) -> Dict[str, object]:
    with path.open(encoding="utf-8") as file:
        value = json.load(file)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return value


def _format_matrix(labels: List[str], matrix: np.ndarray) -> str:
    lines = [
        "| True \\ Predicted | " + " | ".join(labels) + " |",
        "|" + "---|" * (len(labels) + 1),
    ]
    for label, row in zip(labels, matrix):
        lines.append(f"| {label} | " + " | ".join(str(int(value)) for value in row) + " |")
    return "\n".join(lines)


def _confused_pairs(labels: List[str], matrix: np.ndarray) -> List[str]:
    pairs = []
    for first in range(len(labels)):
        for second in range(first + 1, len(labels)):
            forward = int(matrix[first, second])
            reverse = int(matrix[second, first])
            pairs.append((forward + reverse, labels[first], labels[second], forward, reverse))
    pairs.sort(key=lambda pair: (-pair[0], pair[1], pair[2]))
    return [
        f"{first} / {second}: {total} total "
        f"({first}→{second} {forward}, {second}→{first} {reverse})"
        for total, first, second, forward, reverse in pairs[:10]
    ]


def generate_report(
    model_dir: Path = DEFAULT_MODEL_DIR,
    report_path: Path = DEFAULT_REPORT,
) -> Path:
    config = _read_json(model_dir / "config.json")
    metrics = _read_json(model_dir / "metrics_summary.json")
    calibration = _read_json(model_dir / "calibration.json")
    metadata = config["training_metadata"]
    if metadata["dataset_source"] != "collected":
        raise ValueError("A real-data report cannot be generated from synthetic training.")

    labels = config["labels"]
    with np.load(model_dir / "evaluation_test.npz") as held_out:
        y_true = held_out["y"].astype(np.int64)
        y_pred = held_out["predictions"].astype(np.int64)
    matrix = confusion_matrix(y_true, y_pred, labels=list(range(len(labels))))
    configured_matrix = np.asarray(config["confusion_matrix"], dtype=np.int64)
    if not np.array_equal(matrix, configured_matrix):
        raise ValueError("Held-out predictions do not match the saved confusion matrix.")

    accuracy = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(
        f1_score(
            y_true,
            y_pred,
            labels=list(range(len(labels))),
            average="macro",
            zero_division=0,
        )
    )
    if not np.isclose(accuracy, metadata["test_accuracy"], atol=1e-12):
        raise ValueError("Held-out accuracy does not match the saved training config.")
    if not np.isclose(macro_f1, metadata["test_macro_f1"], atol=1e-12):
        raise ValueError("Held-out macro-F1 does not match the saved training config.")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    image_path = report_path.with_name("confusion_matrix.png")
    figure, axis = plt.subplots(figsize=(13, 11))
    image = axis.imshow(matrix, interpolation="nearest", cmap="Blues")
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    axis.set(
        xticks=np.arange(len(labels)),
        yticks=np.arange(len(labels)),
        xticklabels=labels,
        yticklabels=labels,
        xlabel="Predicted label",
        ylabel="True label",
        title=f"Held-out confusion matrix (accuracy {accuracy:.6f})",
    )
    plt.setp(axis.get_xticklabels(), rotation=90, ha="center")
    figure.tight_layout()
    figure.savefig(image_path, dpi=200)
    plt.close(figure)

    counts = metadata["sample_counts_per_label"]
    hand_counts = metadata["sample_counts_per_hand"]
    participants = ", ".join(metadata["participants"])
    sessions = ", ".join(metadata["sessions"])
    conditions = ", ".join(metadata["conditions"])
    pairs = _confused_pairs(labels, matrix)
    per_letter = [
        "| {label} | {support:.0f} | {precision:.6f} | {recall:.6f} |".format(
            label=label,
            support=metrics[label]["support"],
            precision=metrics[label]["precision"],
            recall=metrics[label]["recall"],
        )
        for label in labels
    ]
    fold_lines = [
        f"- {fold['name']}: {fold['method']}; "
        f"accuracy {fold['test_accuracy']:.6f}; macro-F1 {fold['test_macro_f1']:.6f}; "
        f"best epoch {fold['best_epoch']}."
        for fold in metadata["folds"]
    ]
    rejection = calibration["rejection"]
    label_counts = ", ".join(f"{label}: {counts[label]}" for label in labels)
    report_text = f"""# SIGNAL real-data training report

## Data provenance

- Source: `{metadata['dataset_source']}` browser-collected landmark data.
- CSV files: {', '.join(metadata['data_files'])}.
- Participants ({metadata['participant_count']}): {participants}.
- Sessions ({metadata['session_count']}): {sessions}.
- Conditions: {conditions}.
- Usable rows: {sum(counts.values())}; removed for hand scale below 0.06: {metadata['dropped_small_hands']}.
- Handedness counts after filtering: Left {hand_counts['Left']}, Right {hand_counts['Right']}.
- Counts per letter: {label_counts}.
- `NONE` samples: {counts.get('NONE', 0)}.

## Split and evaluation

- Split: {metadata['evaluation_split']}.
- Folds:
{chr(10).join(fold_lines)}
- Held-out accuracy: {accuracy:.6f}.
- Held-out macro-F1: {macro_f1:.6f}.
- Held-out samples: {len(y_true)}.

## Per-letter precision and recall

| Letter | Support | Precision | Recall |
|---|---:|---:|---:|
{chr(10).join(per_letter)}

## Confusion matrix

Rows are true labels; columns are predicted labels. Counts are held-out samples.

![Held-out confusion matrix](confusion_matrix.png)

{_format_matrix(labels, matrix)}

## Ten most confused letter pairs

Pairs rank by the sum of both confusion directions.

{chr(10).join(f'{index}. {pair}' for index, pair in enumerate(pairs, start=1))}

## Calibration and rejection

- Temperature fitted on the validation block: {calibration['temperature']:.8f}.
- Validation ECE before temperature scaling: {calibration['ece_before']:.8f}.
- Validation ECE after temperature scaling: {calibration['ece_after']:.8f}.
- Reject if calibrated maximum probability is below {rejection['max_probability_below']:.8f}, or the top-two margin is below {rejection['top_two_margin_below']:.8f}, or the predicted class is `NONE`.
- Validation rejection selection: {rejection['selection_method']}
- Validation balanced acceptance accuracy: {rejection['validation_balanced_acceptance_accuracy']:.6f}.
- Validation correct-sign coverage: {rejection['validation_correct_sign_coverage']:.6f}.
- Validation overall coverage: {rejection['validation_overall_coverage']:.6f}.

## Known limitations

- This is a same-person, same-session, NOT generalization evaluation: only {metadata['participant_count']} participant and {metadata['session_count']} session are represented.
- The data contains only these collection conditions: {conditions}; it does not establish performance across lighting, camera angles, distances, or participants.
- No `NONE` transition examples were collected, so unknown-sign rejection is not validated against real transitions.
- Held-out accuracy is {accuracy:.6f}; classes with zero held-out recall include {', '.join(label for label in labels if metrics[label]['recall'] == 0.0)}.
- Temperature was fit by minimizing validation cross-entropy, but measured validation ECE changed from {calibration['ece_before']:.8f} to {calibration['ece_after']:.8f}; the ECE did not improve in this run.
- J and Z are excluded because they require motion trajectories; this is a static-pose classifier.
- Adjacent collected frames are temporally correlated, so the held-out score does not demonstrate independent-session or real-world generalization.
"""
    report_path.write_text(report_text, encoding="utf-8")
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    print(generate_report(args.model_dir, args.report))


if __name__ == "__main__":
    main()
