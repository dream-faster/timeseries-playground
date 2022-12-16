import numpy as np
from drift.models.base import Model, ModelType
from statsforecast.models import _TS


class UnivariateStatsForecastModel(Model):

    name = "UnivariateStatsForecastModel"
    type = ModelType.Univariate

    def __init__(self, model: _TS) -> None:
        self.model = model

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.model.fit(y=X)

    def predict_in_sample(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_in_sample()["mean"]

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(h=len(X))["mean"]
