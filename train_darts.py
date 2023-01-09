import pandas as pd
from darts import TimeSeries
from darts.models import ExponentialSmoothing, TBATS, ARIMA, Theta, AutoARIMA
from darts.metrics import rmse

import numpy as np
import pandas as pd
from krisi.evaluate import evaluate as krisi_evaluate
from krisi.evaluate import SampleTypes
from krisi.evaluate.type import CalculationTypes
from darts.models import NaiveSeasonal, NaiveDrift, NaiveMean, AutoARIMA
from darts.utils.utils import SeasonalityMode
from data.energy import get_energy_data

data = get_energy_data()

# Create a TimeSeries, specifying the time and value columns
series = TimeSeries.from_dataframe(data[["load"]].dropna(), value_cols=["load"])
models = [
    NaiveDrift(),
    NaiveMean(),
    # NaiveSeasonal(K=1),
    # NaiveSeasonal(K=96),
    # ExponentialSmoothing(),
    # TBATS(),
    # ARIMA(p=1, d=1, q=0),
    # AutoARIMA(),
    # Theta(season_mode=SeasonalityMode.NONE),
]
for model in models:
    print(model)
    model.fit(series)
    # history = model.backtest(
    #     series, start=400, forecast_horizon=1, stride=400, verbose=True
    # )
    history = model.historical_forecasts(
        series, start=1000, forecast_horizon=1, verbose=True
    )
    print(f"{model} RMSE = {rmse(history, series)}%")
