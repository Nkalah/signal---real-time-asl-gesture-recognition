"""PyTorch model definition for ASL Landmark Classification."""

import torch
import torch.nn as nn
from typing import Dict, Any


class ASLLandmarkMLP(nn.Module):
    """
    Multilayer Perceptron for 21-landmark (63-dimensional) ASL Alphabet classification.
    
    Architecture:
      Input (63 features) -> Linear(128) -> BatchNorm1d(128) -> ReLU -> Dropout(0.2)
                          -> Linear(64)  -> BatchNorm1d(64)  -> ReLU -> Dropout(0.2)
                          -> Linear(24)  -> Logits
    """
    def __init__(self, input_dim: int = 63, hidden_dim_1: int = 128, hidden_dim_2: int = 64, num_classes: int = 24, dropout_rate: float = 0.2):
        super(ASLLandmarkMLP, self).__init__()
        
        self.input_dim = input_dim
        self.num_classes = num_classes
        
        self.fc1 = nn.Linear(input_dim, hidden_dim_1)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_rate)
        
        self.fc2 = nn.Linear(hidden_dim_1, hidden_dim_2)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout_rate)
        
        self.fc3 = nn.Linear(hidden_dim_2, num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning raw logits."""
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.dropout1(out)
        
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.dropout2(out)
        
        logits = self.fc3(out)
        return logits
    
    def get_config(self) -> Dict[str, Any]:
        """Return model architectural hyperparameters."""
        return {
            "input_dim": self.input_dim,
            "hidden_dim_1": self.fc1.out_features,
            "hidden_dim_2": self.fc2.out_features,
            "num_classes": self.num_classes,
            "architecture": "MLP-63-128-64-24",
            "activation": "ReLU",
            "dropout": 0.2
        }
