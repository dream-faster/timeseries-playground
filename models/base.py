from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np


class Model(ABC):

    name: str

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        raise NotImplementedError

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def predict_in_sample(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError
