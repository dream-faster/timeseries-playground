import pandas as pd
from krisi.evaluate import evaluate as krisi_evaluate
from krisi.evaluate import SampleTypes
from krisi.evaluate.type import CalculationTypes
from sktime.forecasting.model_selection import ExpandingWindowSplitter
from sktime.forecasting.ets import AutoETS
from sktime.forecasting.statsforecast import StatsForecastAutoARIMA
from sktime.forecasting.theta import ThetaForecaster
from sktime.forecasting.model_evaluation import evaluate
from sktime.forecasting.naive import NaiveForecaster
from sktime.forecasting.arima import ARIMA
from data.energy import get_energy_data
from sktime.forecasting.ets import AutoETS
from sktime.forecasting.var import VAR
from sktime.forecasting.sarimax import SARIMAX
from sktime.forecasting.croston import Croston

data = get_energy_data()
y = data["P"].values

models = [
    # AutoETS(auto=True, sp=24, n_jobs=-1, random_state=42),
    # ThetaForecaster(sp=24),
    StatsForecastAutoARIMA(1, 1),
    # NaiveForecaster(strategy="last"),
    # NaiveForecaster(strategy="mean", window_length=400),
    # NaiveForecaster(strategy="drift", window_length=20),
    # SARIMAX(),
    # Croston(),
    # ARIMA(order=(1, 0, 0), suppress_warnings=True),
]

cv = ExpandingWindowSplitter(fh=400, initial_window=400, step_length=1)


results = [
    evaluate(forecaster=model, y=y, cv=cv, strategy="refit", return_data=True)
    for model in models
]

for index, result in enumerate(results):
    krisi_evaluate(
        models[index].__class__.__name__,
        "energy",
        sample_type=SampleTypes.outsample,
        calculation_type=CalculationTypes.single,
        y=result["y_test"],
        predictions=result["y_pred"],
    ).print_summary()
