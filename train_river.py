from river import datasets
from river import compose
from river import linear_model
from river import preprocessing
from river import metrics
from river import evaluate
from river import optim
from river import time_series
from river import datasets
import pandas as pd
from tqdm import tqdm
from river import stream

X_y = stream.iter_csv(
    "data/energy.csv",
    target="P",
    parse_dates={
        "time": "%Y-%m-%d %H:%M:%S",
    },
    converters={
        "load": float,
        "Gb(i)": float,
        "Gd(i)": float,
        "H_sun": float,
        "T2m": float,
        "WS10m": float,
        "P": float,
        "residual_load": float,
    },
)


# model = compose.Pipeline(("arimax", time_series.SNARIMAX(1, 2, 3, 4, 5, 6, 7)))
model = time_series.HoltWinters(
    alpha=0.3, beta=0.1, gamma=0.6, seasonality=12, multiplicative=True
)
model = time_series.SNARIMAX(1, 1, 1)

for x, y in tqdm(X_y):
    print(x, y)

print(time_series.evaluate(X_y, model, metrics.RMSE(), 1))
