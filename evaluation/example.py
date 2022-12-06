from eda.utils import generating_arima_synthetic_data
from eda.modeling import fit_arima_model
from models.statsforecast_wrapper import UnivariateStatsForecastModel
from statsforecast.models import AutoARIMA
from train import walk_forward_train_predict
import pandas as pd

from typing import Tuple
from statsmodels.tsa.stattools import acf, pacf, q_stat

from models.naive_models import NaiveForecaster


def generate_artificial_dataset_prediction(nsample:int=1000, prediction_len:int = 100)->Tuple[pd.DataFrame,pd.Series, pd.Series]:
    target_col= "example"
    df = generating_arima_synthetic_data(target_col=target_col, nsample=nsample).to_frame()

    model_res = fit_arima_model(df[target_col])
    insample_prediction = model_res.predict(start=0, end=len(df))
    outsample_prediction = model_res.predict(start=len(df)+1, end=len(df)+prediction_len)
    
    return df, insample_prediction, outsample_prediction


def evaluate(features:pd.DataFrame, predictions:pd.Series):    
    alpha = 0.05
    pacf_res = pacf(predictions, alpha=alpha)
    acf_res = acf(predictions, alpha=alpha)

    ljung_box = q_stat(acf_res, len(features))
    
    print(ljung_box)


def evaluate_both():
    df, insample_prediction, outsample_prediction = generate_artificial_dataset_prediction()
    evaluate(df, insample_prediction)
    # evaluate(df, outsample_prediction)
    

if __name__ == "__main__":
    evaluate_both()