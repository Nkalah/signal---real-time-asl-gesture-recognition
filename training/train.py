"""PyTorch training script for SIGNAL ASL Landmark Classifier."""

import os
import sys
import json
import time
import copy
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Ensure root is in path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from backend.app.model import ASLLandmarkMLP
from backend.app.config import SUPPORTED_LETTERS, INPUT_FEATURES, NUM_CLASSES, PREPROCESSING_VERSION
from training.prepare_data import (
    augment_landmarks,
    load_training_dataset,
    normalize_landmarks,
    split_dataset,
)


def train_model(
    epochs: int = 40,
    batch_size: int = 32,
    lr: float = 0.001,
    seed: int = 42,
    save_dir: Path = Path("backend/models")
):
    print("=" * 60)
    print("SIGNAL ASL MODEL TRAINING PIPELINE")
    print("=" * 60)
    
    # Set seeds
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # 1. Dataset generation / loading
    print("\n[1/5] Preparing landmark dataset (24 static ASL classes)...")
    X, y, participants, dataset_source = load_training_dataset(
        samples_per_class=250,
        seed=seed,
    )
    if dataset_source == "synthetic":
        print(
            "WARNING: No collected webcam landmarks found. "
            "This run uses synthetic canonical poses and is not a real-world accuracy evaluation."
        )
    print(f"Total dataset samples: {X.shape[0]} | Features per sample: {X.shape[1]}")
    
    # 2. Train / Val / Test split (70 / 15 / 15)
    train_idx, val_idx, test_idx, split_strategy = split_dataset(y, participants, seed)
    X_train, X_val, X_test = X[train_idx], X[val_idx], X[test_idx]
    y_train, y_val, y_test = y[train_idx], y[val_idx], y[test_idx]

    if dataset_source == "collected":
        rng = np.random.default_rng(seed)
        augmented = np.stack([
            normalize_landmarks(augment_landmarks(sample.reshape((21, 3)), rng))
            for sample in X_train
        ])
        X_train = np.concatenate((X_train, augmented))
        y_train = np.concatenate((y_train, y_train.copy()))
        print(f"Added mirrored/spatial training augmentation: {len(augmented)} samples")
    
    print(f"Split sizes -> Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
    print(f"Dataset source: {dataset_source} | Split strategy: {split_strategy}")
    print(f"Independent split groups: {len(set(participants))}")
    
    # Convert to PyTorch tensors
    train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_dataset = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
    test_dataset = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # 3. Model Architecture
    print("\n[2/5] Initializing PyTorch MLP (63 -> 128 -> 64 -> 24)...")
    device = torch.device("cpu")
    model = ASLLandmarkMLP(
        input_dim=INPUT_FEATURES,
        hidden_dim_1=128,
        hidden_dim_2=64,
        num_classes=NUM_CLASSES,
        dropout_rate=0.2
    ).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=4)
    
    # 4. Training loop
    print(f"\n[3/5] Starting training loop for {epochs} epochs...")
    best_val_loss = float('inf')
    best_val_acc = 0.0
    best_state_dict = None
    
    start_time = time.time()
    
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * batch_x.size(0)
            _, predicted = torch.max(outputs, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
            
        epoch_loss = running_loss / total
        epoch_acc = correct / total
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for val_x, val_y in val_loader:
                val_x, val_y = val_x.to(device), val_y.to(device)
                val_outputs = model(val_x)
                v_loss = criterion(val_outputs, val_y)
                val_loss += v_loss.item() * val_x.size(0)
                _, v_pred = torch.max(val_outputs, 1)
                val_total += val_y.size(0)
                val_correct += (v_pred == val_y).sum().item()
                
        val_epoch_loss = val_loss / val_total
        val_epoch_acc = val_correct / val_total
        scheduler.step(val_epoch_loss)
        
        if val_epoch_loss < best_val_loss:
            best_val_loss = val_epoch_loss
            best_val_acc = val_epoch_acc
            best_state_dict = copy.deepcopy(model.state_dict())
            
        if epoch % 5 == 0 or epoch == epochs:
            print(f"Epoch [{epoch:02d}/{epochs:02d}] - Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc*100:.2f}% | Val Loss: {val_epoch_loss:.4f}, Val Acc: {val_epoch_acc*100:.2f}%")
            
    training_duration = time.time() - start_time
    print(f"\nTraining completed in {training_duration:.2f}s | Best Val Acc: {best_val_acc*100:.2f}%")
    
    # 5. Evaluate on held-out test set
    print("\n[4/5] Evaluating on held-out test set...")
    model.load_state_dict(best_state_dict)
    model.eval()
    
    test_preds = []
    test_targets = []
    with torch.no_grad():
        for test_x, test_y in test_loader:
            test_x = test_x.to(device)
            out = model(test_x)
            _, pred = torch.max(out, 1)
            test_preds.extend(pred.cpu().numpy().tolist())
            test_targets.extend(test_y.numpy().tolist())
            
    test_acc = accuracy_score(test_targets, test_preds)
    print(f"\n>>> TEST ACCURACY: {test_acc * 100:.2f}% <<<")
    
    report = classification_report(
        test_targets, test_preds,
        target_names=SUPPORTED_LETTERS,
        digits=4,
        output_dict=True
    )
    cm = confusion_matrix(test_targets, test_preds)
    
    # 6. Save Model Artifacts
    print(f"\n[5/5] Saving model artifacts to {save_dir}...")
    save_dir.mkdir(parents=True, exist_ok=True)
    
    model_weights_path = save_dir / "asl_mlp_model.pth"
    torch.save(best_state_dict, model_weights_path)
    print(f"Saved PyTorch weights: {model_weights_path}")
    np.savez_compressed(save_dir / "evaluation_test.npz", X=X_test, y=y_test)
    
    # Export weights for Node / Web inference if needed
    weights_json = {}
    for k, v in best_state_dict.items():
        weights_json[k] = v.cpu().numpy().tolist()
    with open(save_dir / "weights.json", "w", encoding="utf-8") as f:
        json.dump(weights_json, f)
    print(f"Saved JSON weights for universal runtime: {save_dir / 'weights.json'}")
    
    # Save config
    if dataset_source == "collected":
        signer_ids = {
            (group.split("::")[1] if str(group).startswith(("signer::", "session::")) else str(group))
            for group in participants
        }
        participant_count = len(signer_ids)
    else:
        participant_count = 0  # Synthetic pseudo-groups are not real human participants.

    config_data = {
        "model_name": "SIGNAL ASL Landmark MLP",
        "version": "1.0.0",
        "architecture": "MLP-63-128-64-24",
        "input_features": INPUT_FEATURES,
        "num_classes": NUM_CLASSES,
        "supported_letters": SUPPORTED_LETTERS,
        "excluded_letters": ["J", "Z"],
        "exclusion_reason": "Letters J and Z involve dynamic motion and cannot be reliably detected with static landmark pose classification.",
        "preprocessing": {
            "version": PREPROCESSING_VERSION,
            "wrist_relative": True,
            "scale_normalization": "p_i' = (p_i - p_0) / (max_dist + 1e-6)",
            "epsilon": 1e-6
        },
        "training_metadata": {
            "epochs": epochs,
            "batch_size": batch_size,
            "initial_lr": lr,
            "random_seed": seed,
            "dataset_source": dataset_source,
            "evaluation_split": split_strategy,
            "participant_count": participant_count,
            "split_group_count": len(set(participants)),
            "training_samples": len(X_train),
            "val_samples": len(X_val),
            "test_samples": len(X_test),
            "test_accuracy": round(float(test_acc), 4),
            "training_time_seconds": round(training_duration, 2)
        },
        "confusion_matrix": cm.tolist()
    }
    
    config_path = save_dir / "config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config_data, f, indent=2)
    print(f"Saved configuration and metadata: {config_path}")
    
    # Also save training summary metrics
    summary_path = save_dir / "metrics_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Saved classification report: {summary_path}")
    
    print("\nTraining completed successfully!")
    return test_acc, model_weights_path, config_path


if __name__ == "__main__":
    train_model(epochs=35, batch_size=32, lr=0.001)
