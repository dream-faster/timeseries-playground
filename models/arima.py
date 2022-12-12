from .base import Model
import numpy as np
from typing import Optional, Tuple, Union
from dataclasses import dataclass
from statsmodels.tsa.arima.model import ARIMA
import pandas as pd
from all_types import X, y, InSamplePredictions, OutSamplePredictions


@dataclass
class ArimaConfig:
    """
    order=(p,d,q).
        p is the autoregressive order
        d is the number of times the series has been differenced
        q is the moving average order
    """

    order: Tuple[float, float, float]  # Forecasting Horizon
    trend: Optional[Union[str, Tuple[float]]] = None


class ArimaWrapper(Model):

    name: str = ""

    def __init__(self, config: ArimaConfig) -> None:
        self.config = config

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        unfitted_model = ARIMA(X, order=self.config.order, trend=self.config.trend)
        self.model = unfitted_model.fit()

    def predict_in_sample(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(end=len(X) - 1)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model:
            return self.model.forecast(len(X))
        else:
            raise ValueError("Model has to be fitted first")


default_arima_model = ArimaWrapper(ArimaConfig(order=(1, 0, 1)))
