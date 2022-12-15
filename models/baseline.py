import numpy as np
from .base import Model, ModelType
from enum import Enum


class Strategy(Enum):
    sliding_mean = "sliding_mean"
    expanding_mean = "expanding_mean"
    naive = "naive"
    expanding_drift = "drift"
    sliding_drift = "sliding_drift"


class BaselineModel(Model):

    name = "BaselineModel"
    strategies = Strategy
    type = ModelType.Univariate

    def __init__(self, strategy: Strategy, window_size: int = 100) -> None:
        self.strategy = strategy
        self.window_size = window_size

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        pass

    def predict_in_sample(self, X: np.ndarray) -> np.ndarray:
        if self.strategy == Strategy.sliding_mean:
            return np.array(
                [
                    np.mean(X[max(i - self.window_size, 0) : i + 1])
                    for i in range(len(X))
                ]
            )
        if self.strategy == Strategy.expanding_mean:
            return np.array([np.mean(X[: i + 1]) for i in range(len(X))])
        elif self.strategy == Strategy.naive:
            return X
        elif self.strategy == Strategy.sliding_drift:
            return np.array(
                [
                    calculate_drift_predictions(X[max(i - self.window_size, 0) : i + 1])
                    for i in range(len(X))
                ]
            )
        elif self.strategy == Strategy.expanding_drift:
            return np.array([calculate_drift_predictions(X[: i + 1]) for i in len(X)])
        else:
            raise ValueError(f"Strategy {self.strategy} not implemented")

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.predict_in_sample(X)


def calculate_drift_predictions(y: np.ndarray) -> np.ndarray:
    return y[-1] + np.mean(np.diff(y))
