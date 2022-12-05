from eda.utils import generating_arima_synthetic_data
from eda.modeling import fit_arima_model
from models.statsforecast_wrapper import UnivariateStatsForecastModel
from statsforecast.models import AutoARIMA
from train import walk_forward_train_predict
import pandas as pd

from typing import Tuple
from statsmodels.tsa.stattools import acf, pacf, q_stat


def generate_artificial_dataset_prediction(nsample:int=1000, prediction_len:int = 100)->Tuple[pd.DataFrame,pd.DataFrame, pd.DataFrame]:
    target_col= "example"
    df = generating_arima_synthetic_data(target_col=target_col, nsample=nsample).to_frame()

    model_res = fit_arima_model(df[target_col])
    insample_prediction = model_res.predict(start=0, end=len(df))
    outsample_prediction = model_res.predict(start=len(df)+1, end=len(df)+prediction_len)
    
    return df, insample_prediction, outsample_prediction


def evaluate():
    df, insample_prediction, outsample_prediction = generate_artificial_dataset_prediction()
    
    alpha = 0.05
    pacf_res = pacf(insample_prediction.dropna(), alpha=alpha)
    acf_res = acf(insample_prediction.dropna(), alpha=alpha)

    ljung_box = q_stat(acf_res, len(df))
    
    print(ljung_box)


# df['y'] = df[target_col]
# df['X'] = df[target_col].shift(1)

# df = df.dropna()
# X = df[['X']]
# y = df['y']

# model = UnivariateStatsForecastModel(model=AutoARIMA())
# models_over_time, insample_predictions, outofsample_predictions = walk_forward_train_predict(model, X, y, 200, 100)