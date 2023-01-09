import numpy as np
from tqdm import tqdm
from data.energy import get_energy_data
from orbit.diagnostics.backtest import BackTester, TimeSeriesSplitter
from orbit.diagnostics.plot import plot_bt_predictions
from orbit.diagnostics.metrics import smape, wmape

data = get_energy_data()
min_train_len = 380  # minimal length of window length
forecast_len = 1  # length forecast window
incremental_len = 1  # step length for moving forward
ex_splitter = TimeSeriesSplitter(
    df=data,
    min_train_len=min_train_len,
    incremental_len=incremental_len,
    forecast_len=forecast_len,
    window_type="expanding",
    date_col="time",
)

# instantiate a model
dlt = DLT(
    date_col="week",
    response_col="claims",
    regressor_col=["trend.unemploy", "trend.filling", "trend.job"],
    seasonality=52,
    estimator="stan-map",
    # reduce number of messages
    verbose=False,
)
# configs
min_train_len = 100
forecast_len = 20
incremental_len = 100
window_type = "expanding"

bt = BackTester(
    model=dlt,
    df=data,
    min_train_len=min_train_len,
    incremental_len=incremental_len,
    forecast_len=forecast_len,
    window_type=window_type,
)
