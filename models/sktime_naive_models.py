from sktime.forecasting.naive import NaiveForecaster
from .base import Model, ModelType
import numpy as np
from typing import List, Optional, Union
from enum import Enum
from dataclasses import dataclass
from sktime.forecasting.base import ForecastingHorizon
from all_types import OutSamplePredictions


class StrategyTypes(Enum):
    mean = "mean"
    last = "last"
    drift = "drift"


@dataclass
class NaiveForecasterConfig:
    strategy: StrategyTypes
    fh: Optional[Union[int, List[int]]] = None  # Forecasting Horizon
    window_length: Optional[int] = None  # The window of the mean if strategy is 'mean'
    sp: int = 1  # Seasonal periodicity


class NaiveForecasterWrapper(Model):

    strategy = StrategyTypes

    name = "NaiveForecaster"
    type = ModelType.Univariate

    def __init__(self, config: NaiveForecasterConfig) -> None:
        self.model = NaiveForecaster(
            strategy=config.strategy.value,
            sp=config.sp,
            window_length=config.window_length,
        )
        self.config = config

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.model.fit(X)

    def predict_in_sample(self, X: np.ndarray) -> np.ndarray:
        fh = ForecastingHorizon([x + 1 for x in range(len(X))], is_relative=False)
        fh = fh.to_relative(cutoff=len(X))
        return self.model.predict(fh=fh)

    def predict(self, X: np.ndarray) -> OutSamplePredictions:
        fh = (
            self.config.fh
            if self.config.fh is not None
            else ForecastingHorizon([x + 1 for x in range(len(X))], is_relative=True)
        )
        return self.model.predict(fh=fh)
