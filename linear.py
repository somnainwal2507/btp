"""
Linear models: OLS and LASSO.
"""

import os
import numpy as np
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.preprocessing import StandardScaler

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


class OLSModel:
    """Ordinary Least Squares regression using the kitchen-sink approach."""

    def __init__(self):
        self.name = "OLS"
        self.model = LinearRegression(n_jobs=-1)
        self.scaler = StandardScaler()

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """Fit OLS model."""
        X_scaled = self.scaler.fit_transform(X_train)
        # Replace any infinities or NaNs after scaling
        X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)
        self.model.fit(X_scaled, y_train)
        return self

    def predict(self, X):
        """Predict returns."""
        X_scaled = self.scaler.transform(X)
        X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)
        return self.model.predict(X_scaled)


class LASSOModel:
    """LASSO regression with L1 penalty."""

    def __init__(self, alpha=None):
        self.name = "LASSO"
        self.alpha = alpha or config.LASSO_ALPHA
        self.model = Lasso(alpha=self.alpha, max_iter=10000, random_state=42)
        self.scaler = StandardScaler()

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """Fit LASSO model."""
        X_scaled = self.scaler.fit_transform(X_train)
        X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)
        self.model.fit(X_scaled, y_train)
        return self

    def predict(self, X):
        """Predict returns."""
        X_scaled = self.scaler.transform(X)
        X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)
        return self.model.predict(X_scaled)
