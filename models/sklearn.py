import numpy as np
from drift.models.base import Model, ModelType
from sklearn.base import BaseEstimator


class SKLearnModel(Model):

    name = "SKLearnModel"
    type = ModelType.Multivariate

    def __init__(self, model: BaseEstimator) -> None:
        self.model = model

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.model.fit(X, y)

    def predict_in_sample(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)
