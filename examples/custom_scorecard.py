from krisi import explore, report
from krisi.evaluate import evaluate, ScoreCard
from krisi.utils.models import generate_univariate_predictions
from krisi.utils.data import generating_arima_synthetic_data

score_card = ScoreCard(
    model_name="example_model",
    dataset_name="example_dataset",
    sample_type=ScoreCard.sample_type.insample,
)
