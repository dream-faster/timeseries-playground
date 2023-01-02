from statsforecast.models import AutoARIMA
from drift.models.baseline import BaselineModel
from drift_models.statsforecast_univariate import UnivariateStatsForecastModel
from drift.transformations.no import NoTransformation
from data.energy import get_energy_data
from data.market_data import get_market_data
from drift.utils.splitters import ExpandingWindowSplitter
from drift.loop import walk_forward_inference, walk_forward_train

from krisi.evaluate import evaluate, SampleTypes
from krisi.evaluate.type import CalculationTypes

data = get_energy_data()
y = data["P"]
X = y

model = UnivariateStatsForecastModel(model=AutoARIMA(d=1, D=1))
# model = BaselineModel(strategy=BaselineModel.strategies.naive)

splitter = ExpandingWindowSplitter(start=0, end=len(y), window_size=4000, step=1)

model_over_time = walk_forward_train(model, X, y, splitter, None)
insample_predictions, outofsample_predictions = walk_forward_inference(
    model_over_time, None, X, y, splitter
)
# print(insample_predictions, outofsample_predictions)

# evaluate(
#     model.name,
#     "energy",
#     sample_type=SampleTypes.outsample,
#     calculation_type=CalculationTypes.single,
#     y=y[outofsample_predictions.index],
#     predictions=outofsample_predictions,
# ).print_summary()
