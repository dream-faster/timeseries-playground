from krisi.evaluate import evaluate_in_out_sample, ScoreCard
from krisi.utils.models import default_arima_model
from krisi.utils.runner import generate_univariate_predictions, fit_model
from krisi.utils.data import generating_arima_synthetic_data
from sklearn.model_selection import train_test_split
import pandas as pd

target_col = "arima_synthetic"
df = generating_arima_synthetic_data(target_col).to_frame(name=target_col)
train, test = train_test_split(df, test_size=0.2, shuffle=False)

fit_model(default_arima_model, train)

insample_prediction, outsample_prediction = generate_univariate_predictions(
    default_arima_model, test, target_col
)


evaluate_in_out_sample(
    model_name="default_arima_model",
    dataset_name="synthetic_arima",
    y_insample=train[target_col],
    insample_predictions=insample_prediction,
    y_outsample=test[target_col],
    outsample_predictions=outsample_prediction,
)
