from statsforecast.models import AutoARIMA
from drift.models.baseline import BaselineModel
from models.statsforecast_wrapper import UnivariateStatsForecastModel
from models.sktime_naive_models import NaiveForecasterConfig, NaiveForecasterWrapper
from drift.transformations.no import NoTransformation
from utils.market_data import get_market_data
from drift.utils.splitters import ExpandingWindowSplitter
from drift.loop import walk_forward_inference, walk_forward_train, fit_transformations
from drift.transformations import NoTransformation
import yfinance as yf

yf.pdr_override()

from krisi.evaluate import evaluate, SampleTypes


if __name__ == "__main__":
    data = get_market_data()

    X = data["VIX"]
    y = data["VIX"].shift(-1)

    transformations = [NoTransformation()]

    model = UnivariateStatsForecastModel(model=AutoARIMA())
    model = NaiveForecasterWrapper(
        NaiveForecasterConfig(strategy=NaiveForecasterWrapper.strategy.last)
    )
    model = BaselineModel(strategy=BaselineModel.strategies.naive)

    splitter = ExpandingWindowSplitter(start=0, end=len(y), window_size=1000, step=500)

    transformations_over_time = fit_transformations(X, y, splitter, transformations)
    model_over_time = walk_forward_train(model, X, y, splitter, None)
    insample_predictions, outofsample_predictions = walk_forward_inference(
        model_over_time, None, X, y, splitter
    )
    print(insample_predictions, outofsample_predictions)

    insample_scorecard = evaluate(
        "BaselineModel",
        "VIX",
        SampleTypes.insample,
        y[insample_predictions.index],
        insample_predictions,
    ).print_summary()
    # outsample_scorecard = evaluate(
    #     "BaselineModel",
    #     "VIX",
    #     SampleTypes.outsample,
    #     y[outofsample_predictions.index],
    #     outofsample_predictions,
    # ).print_summary()
