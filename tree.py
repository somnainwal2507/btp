"""
Tree-based models: Decision Tree and Random Forest.
"""

import os
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


class DecisionTreeModel:
    """Standard CART Decision Tree with greedy optimal splits."""

    def __init__(self):
        self.name = "DecisionTree"
        self.model = DecisionTreeRegressor(**config.DT_PARAMS)
        self.scaler = StandardScaler()

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """Fit Decision Tree."""
        X_scaled = self.scaler.fit_transform(X_train)
        X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)
        self.model.fit(X_scaled, y_train)
        return self

    def predict(self, X):
        """Predict returns."""
        X_scaled = self.scaler.transform(X)
        X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)
        return self.model.predict(X_scaled)


class RandomForestModel:
    """Random Forest with 50 estimators and max depth 8."""

    def __init__(self):
        self.name = "RandomForest"
        self.model = RandomForestRegressor(**config.RF_PARAMS)
        self.scaler = StandardScaler()

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """Fit Random Forest."""
        X_scaled = self.scaler.fit_transform(X_train)
        X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)
        self.model.fit(X_scaled, y_train)
        return self

    def predict(self, X):
        """Predict returns."""
        X_scaled = self.scaler.transform(X)
        X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)
        return self.model.predict(X_scaled)
