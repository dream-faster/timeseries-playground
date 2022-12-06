from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np
from typing import Union
import pandas as pd


class Model(ABC):

    name: str = ""

    @abstractmethod
    def fit(
        self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def predict(
        self, X: Union[pd.DataFrame, np.ndarray]
    ) -> Union[np.ndarray, pd.Series]:
        raise NotImplementedError

    @abstractmethod
    def predict_in_sample(
        self, X: Union[pd.DataFrame, np.ndarray]
    ) -> Union[np.ndarray, pd.Series]:
        raise NotImplementedError
