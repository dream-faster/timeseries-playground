from .base import Model
import numpy as np
from sktime.forecasting.base import ForecastingHorizon, BaseForecaster
from all_types import OutSamplePredictions


class SKTimeModelWrapper(Model):

    name: str = "SKTimeModelWrapper"

    def __init__(self, model: BaseForecaster, fh: ForecastingHorizon) -> None:
        self.model = model
        self.fh = fh

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.model.fit(y, X, self.fh)

    def predict_in_sample(self, X: np.ndarray) -> np.ndarray:
        fh = ForecastingHorizon([x + 1 for x in range(len(X))], is_relative=False)
        fh = fh.to_relative(cutoff=len(X))
        return self.model.predict(fh=fh, X)

    def predict(self, X: np.ndarray) -> OutSamplePredictions:
        return self.model.predict(fh=fh, X)


default_naive_model = NaiveForecasterWrapper(
    NaiveForecasterConfig(strategy=NaiveForecasterWrapper.strategy_types.last)
)