from typing import Tuple, Union, Optional, Any
from dataclasses import dataclass
import pandas as pd
import numpy as np

from statsmodels.tsa.stattools import acf, pacf, q_stat

from models.base import Model
from eda.utils import generating_arima_synthetic_data
from models.naive_models import default_naive_model
from models.arima import default_arima_model


def generate_univariate_predictions(
    model: Model,
    df: pd.DataFrame,
    target_col: str,
) -> Tuple[Union[pd.Series, np.ndarray], Union[pd.Series, np.ndarray]]:

    model.fit(df[[target_col]], df[target_col])
    outsample_prediction = model.predict(df[[target_col]])
    insample_prediction = model.predict_in_sample(df[[target_col]])

    return insample_prediction, outsample_prediction


@dataclass
class ScoreCard:
    ljung_box_score: Optional[int] = None

    def update(self, key: str, value: Any):
        if hasattr(self, key):
            setattr(self, key, value)


def evaluate(
    features: pd.DataFrame, predictions: Union[np.ndarray, pd.Series]
) -> ScoreCard:
    alpha = 0.05
    pacf_res = pacf(predictions, alpha=alpha)
    acf_res = acf(predictions, alpha=alpha)

    summary = ScoreCard()
    summary.update("ljung_box", q_stat(acf_res, len(features)))

    print(summary)

    return summary


def evaluate_both():

    target_col = "example"
    df = generating_arima_synthetic_data(
        target_col=target_col,
        nsample=1000,
    ).to_frame()

    # model = default_naive_model
    model = default_arima_model

    insample_prediction, outsample_prediction = generate_univariate_predictions(
        model, df, target_col
    )

    insample_summary = evaluate(df, insample_prediction)
    outsample_summary = evaluate(df, outsample_prediction)


if __name__ == "__main__":
    evaluate_both()
