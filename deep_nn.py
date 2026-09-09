"""
Deep Neural Network (Deep-NN) using PyTorch.
Fully connected feed-forward MLP: 5 hidden layers (64-32-16-8-4).
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


class _MLP(nn.Module):
    """PyTorch MLP architecture."""

    def __init__(self, input_dim, hidden_layers=None):
        super().__init__()
        hidden_layers = hidden_layers or config.DNN_PARAMS["hidden_layers"]

        layers = []
        prev_dim = input_dim
        for h in hidden_layers:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.Dropout(0.1))
            prev_dim = h

        # Output layer: linear activation
        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x).squeeze(-1)


class DeepNNModel:
    """
    Deep Neural Network for return prediction.
    Architecture: Input → 64 → 32 → 16 → 8 → 4 → 1
    ReLU activations, BatchNorm, Dropout(0.1), Linear output.
    Trained with Adam optimizer, MSE loss, early stopping.
    """

    def __init__(self):
        self.name = "DeepNN"
        self.model = None
        self.scaler = StandardScaler()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.input_dim = None

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """
        Train the Deep-NN with early stopping.
        """
        # Scale features
        X_scaled = self.scaler.fit_transform(X_train)
        X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)
        y_clean = np.nan_to_num(y_train, nan=0.0)

        self.input_dim = X_scaled.shape[1]

        # Split for validation if not provided
        if X_val is None or y_val is None:
            split = int(0.8 * len(X_scaled))
            X_tr, X_v = X_scaled[:split], X_scaled[split:]
            y_tr, y_v = y_clean[:split], y_clean[split:]
        else:
            X_tr, X_v = X_scaled, np.nan_to_num(
                self.scaler.transform(X_val), nan=0.0, posinf=0.0, neginf=0.0
            )
            y_tr, y_v = y_clean, np.nan_to_num(y_val, nan=0.0)

        # Convert to tensors
        X_tr_t = torch.FloatTensor(X_tr).to(self.device)
        y_tr_t = torch.FloatTensor(y_tr).to(self.device)
        X_v_t = torch.FloatTensor(X_v).to(self.device)
        y_v_t = torch.FloatTensor(y_v).to(self.device)

        train_dataset = TensorDataset(X_tr_t, y_tr_t)
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.DNN_PARAMS["batch_size"],
            shuffle=True,
        )

        # Initialize model
        self.model = _MLP(self.input_dim).to(self.device)
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.DNN_PARAMS["learning_rate"],
        )
        criterion = nn.MSELoss()

        # Training with early stopping
        best_val_loss = float("inf")
        patience_counter = 0
        best_state = None

        for epoch in range(config.DNN_PARAMS["max_epochs"]):
            self.model.train()
            epoch_loss = 0.0

            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                predictions = self.model(batch_X)
                loss = criterion(predictions, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                epoch_loss += loss.item()

            # Validation
            self.model.eval()
            with torch.no_grad():
                val_pred = self.model(X_v_t)
                val_loss = criterion(val_pred, y_v_t).item()

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_state = self.model.state_dict().copy()
            else:
                patience_counter += 1
                if patience_counter >= config.DNN_PARAMS["patience"]:
                    break

        # Restore best model
        if best_state is not None:
            self.model.load_state_dict(best_state)

        self.model.eval()
        return self

    def predict(self, X):
        """Predict returns."""
        X_scaled = self.scaler.transform(X)
        X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)
        X_t = torch.FloatTensor(X_scaled).to(self.device)

        self.model.eval()
        with torch.no_grad():
            predictions = self.model(X_t).cpu().numpy()
        return predictions

    def get_torch_model(self):
        """Return the underlying PyTorch model for SHAP analysis."""
        return self.model
