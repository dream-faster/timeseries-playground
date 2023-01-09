import pandas as pd
from krisi.evaluate import evaluate as krisi_evaluate
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
    # StatsForecastAutoARIMA(1, 1),
    # NaiveForecaster(strategy="last"),
    # NaiveForecaster(strategy="mean", window_length=400),
    # NaiveForecaster(strategy="drift", window_length=20),
    SARIMAX(),
    # Croston(),
    # ARIMA(order=(1, 0, 0), suppress_warnings=True),
]

cv = ExpandingWindowSplitter(fh=1, initial_window=400, step_length=1)


results = [
    evaluate(
        forecaster=model,
        y=y,
        cv=cv,
        strategy="update",
        return_data=True,
        error_score="raise",
    )
    for model in models
]

for index, result in enumerate(results):

    def squeeze_weird_sktime_return_format(series: pd.Series) -> pd.Series:
        idx, items = zip(*[(row.index[0], row[0].iloc[0]) for row in series])
        return pd.Series(items, index=idx)

    krisi_evaluate(
        y=squeeze_weird_sktime_return_format(result["y_test"]),
        predictions=squeeze_weird_sktime_return_format(result["y_pred"]),
    ).print_summary()
