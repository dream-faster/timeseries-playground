from all_types import (
    X,
    y,
)
from statsforecast.models import AutoARIMA
from models.statsforecast_wrapper import UnivariateStatsForecastModel
from transformations.no import NoTransformation
from utils.market_data import get_market_data
from utils.splitters import ExpandingWindowSplitter
from train import walk_forward_inference, walk_forward_train, fit_transformations
from transformations import NoTransformation

if __name__ == "__main__":
    data = get_market_data()

    X = data["VIX"]
    y = data["VIX"].shift(-1)

    transformations = [NoTransformation()]

    model = UnivariateStatsForecastModel(model=AutoARIMA())

    splitter = ExpandingWindowSplitter(start=0, end=len(y), window_size=1000, step=500)

    transformations_over_time = fit_transformations(X, y, splitter, transformations)
    model_over_time = walk_forward_train(
        model, X, y, splitter, transformations_over_time
    )
    insample_predictions, outofsample_predictions = walk_forward_inference(
        model_over_time, transformations_over_time, X, splitter
    )
    print(insample_predictions, outofsample_predictions)