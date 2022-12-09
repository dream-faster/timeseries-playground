from typing import Union
from all_types import (
    ModelOverTime,
    InSamplePredictions,
    OutSamplePredictions,
    X,
    y,
)
import pandas as pd
from models.base import Model
from tqdm import tqdm
from typing import Optional
from copy import deepcopy
from statsforecast.models import AutoARIMA
from models.statsforecast_wrapper import UnivariateStatsForecastModel
from utils.market_data import get_market_data


def walk_forward_train_predict(
    model: Model,
    X: X,
    y: y,
    window_size: int,
    retrain_every: int,
    from_index: Optional[pd.Timestamp] = None,
) -> Union[ModelOverTime, InSamplePredictions, OutSamplePredictions]:
    """
    Walk forward train a model on a given dataset.
    :param model: Model to train
    :param X: Features
    :param y: Target
    :param window_size: Size of the window to train on
    :param retrain_every: Retrain the model every n steps
    :param from_index: Start training from this index
    :return: Models over time
    """
    models_over_time = pd.Series(index=y.index, dtype="object").rename(model.name)
    insample_predictions = pd.Series(index=y.index, dtype="object").rename(
        f"{model.name}_insample_predictions"
    )
    outofsample_predictions = pd.Series(index=y.index, dtype="object").rename(
        f"{model.name}_outofsample_predictions"
    )

    train_from = (
        window_size + 1 if from_index is None else X.index.to_list().index(from_index)
    )
    train_till = len(y)

    for index in tqdm(range(train_from, train_till, retrain_every)):
        # it only supports expanding window now, always start from the first non-zero return (change this to support rolling window)
        train_window_start = X.index[0]
        train_window_end = X.index[index - 1]
        test_window_end = X.index[min(len(X) - 1, index - 1 + retrain_every)]

        X_train = X[train_window_start:train_window_end].to_numpy()
        y_train = y[train_window_start:train_window_end].to_numpy()

        X_test = X[train_window_end:test_window_end].to_numpy()

        current_model = deepcopy(model)
        current_model.fit(X_train, y_train)

        models_over_time[X.index[index]] = current_model

        # this means we always replace the first n values with the predictions
        insample_predictions[
            train_window_start:train_window_end
        ] = current_model.predict_in_sample(X_train)
        outofsample_predictions[
            train_window_end:test_window_end
        ] = current_model.predict(X_test)

    return models_over_time, insample_predictions, outofsample_predictions


if __name__ == "__main__":
    data = get_market_data()

    X = data['VIX']
    y = data['VIX'].shift(-1)
    model = UnivariateStatsForecastModel(model=AutoARIMA())
    models_over_time, insample_predictions, outofsample_predictions = walk_forward_train_predict(model, X, y, 1000, 400)

