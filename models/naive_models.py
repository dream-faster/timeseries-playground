from sktime.forecasting.naive import NaiveForecaster
from .base import Model
import numpy as np
from typing import Literal, List, Optional
from enum import Enum
from dataclasses import dataclass

class StrategyTypes(Enum):
    mean = 'mean'
    last = 'last'
    drift = 'drift'
    
@dataclass
class NaiveForecasterConfig:
    fh: List[int] # Forecasting Horizon
    window_length: Optional[int] = None # The window of the mean if strategy is 'mean'
    sp: Optional[int] = 1 # Seasonal periodicity
    

class NaiveForecaster(Model):

    name: str = ""
    forecaster_type: StrategyTypes 
    
    def __init__(self, strategy: StrategyTypes = StrategyTypes.last) -> None:
        self.model = NaiveForecaster(strategy=strategy.value)
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.model.fit(y)

    def predict_in_sample(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_residuals(X) + X

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(h = len(X))['mean']

mean = NaiveForecaster