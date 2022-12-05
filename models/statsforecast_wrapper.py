
import numpy as np
from .base import Model
from statsforecast.models import _TS

class UnivariateStatsForecastModel(Model):

    name: str = ""

    def __init__(self, model: _TS) -> None:
        self.model = model

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.model.fit(y=y)

    def predict_in_sample(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_in_sample()['mean']

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(h = len(X))['mean']


