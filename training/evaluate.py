"""Evaluation script for SIGNAL ASL Landmark Classifier."""

import sys
import json
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from backend.app.model import ASLLandmarkMLP
from backend.app.config import SUPPORTED_LETTERS, INPUT_FEATURES, NUM_CLASSES
from training.prepare_data import load_training_dataset, split_dataset


def run_evaluation(model_path: Path = Path("backend/models/asl_mlp_model.pth")):
    if not model_path.exists():
        print(f"Error: Model not found at {model_path}. Please run train.py first.")
        sys.exit(1)
        
    print(f"Loading model weights from {model_path}...")
    device = torch.device("cpu")
    model = ASLLandmarkMLP(
        input_dim=INPUT_FEATURES,
        hidden_dim_1=128,
        hidden_dim_2=64,
        num_classes=NUM_CLASSES
    ).to(device)
    
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    
    evaluation_data_path = model_path.with_name("evaluation_test.npz")
    if evaluation_data_path.exists():
        with np.load(evaluation_data_path) as data:
            X_test = data["X"].astype(np.float32)
            y_test = data["y"].astype(np.int64)
        print(f"Evaluating on the held-out split saved at {evaluation_data_path}...")
    else:
        X, y, participants, dataset_source = load_training_dataset(
            samples_per_class=100,
            seed=1337,
        )
        _, _, test_indices, split_strategy = split_dataset(y, participants, seed=42)
        X_test, y_test = X[test_indices], y[test_indices]
        print(
            f"WARNING: No saved held-out test split. Evaluating a regenerated "
            f"{dataset_source} split ({split_strategy}); this is not guaranteed "
            "to match the model's training split."
        )
    
    test_tensor = torch.from_numpy(X_test).to(device)
    with torch.no_grad():
        outputs = model(test_tensor)
        _, preds = torch.max(outputs, 1)
        
    preds_np = preds.cpu().numpy()
    acc = accuracy_score(y_test, preds_np)
    
    print("\n" + "=" * 60)
    print(f"SIGNAL ASL MODEL EVALUATION RESULTS (Overall Accuracy: {acc * 100:.2f}%)")
    print("=" * 60)
    
    print("\nDetailed Per-Class Performance:")
    print(classification_report(y_test, preds_np, target_names=SUPPORTED_LETTERS, digits=3))
    
    cm = confusion_matrix(y_test, preds_np)
    print("Confusion Matrix shape:", cm.shape)
    
    # Check for any classes below 85%
    per_class_acc = cm.diagonal() / cm.sum(axis=1)
    for idx, letter in enumerate(SUPPORTED_LETTERS):
        print(f"Letter '{letter}': {per_class_acc[idx] * 100:.1f}% accuracy")
        
    return acc


if __name__ == "__main__":
    run_evaluation()
