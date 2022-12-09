from krisi import explore, evaluate, report
from krisi.evaluate.scorecard import ScoreCard
from krisi.explore.utils import generating_arima_synthetic_data

df = generating_arima_synthetic_data("example_dataset")

score_card = ScoreCard(
    model_name="example_model",
    dataset_name="example_dataset",
    sample_type=ScoreCard.sample_type.insample,
)
