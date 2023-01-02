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
    "data/energy.csv", target="P", parse_dates={"time": "%Y-%m-%d %H:%M:%S"}
)


# model = compose.Pipeline(("arimax", time_series.SNARIMAX(1, 2, 3, 4, 5, 6, 7)))
model = compose.Pipeline(("arimax", time_series.HoltWinters(alpha=0.3,beta=0.1, gamma=0.6,seasonality=12,multiplicative=True)))

print(
    evaluate.progressive_val_score(
        model=model,
        dataset=datasets.AirlinePassengers(),
        metric=metrics.RMSE(),
        print_every=10,
    )
)
