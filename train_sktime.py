from sktime.datasets import load_airline
from sktime.forecasting.ets import AutoETS
from sktime.forecasting.var import VAR
from sktime.forecasting.sarimax import SARIMAX
from sktime.forecasting.croston import Croston
from sktime.forecasting.model_selection import ExpandingWindowSplitter
from sktime.forecasting.model_evaluation import evaluate
from sktime.performance_metrics.forecasting import MeanAbsolutePercentageError
from sktime.forecasting.naive import NaiveForecaster
from sktime.forecasting.fbprophet import Prophet


y = load_airline()
cv = ExpandingWindowSplitter(initial_window=12, step_length=1)
loss = MeanAbsolutePercentageError()


ets = AutoETS()
ets_results = evaluate(forecaster=ets, y=y, cv=cv, scoring=loss)

sarimax = SARIMAX()
sarimax_results = evaluate(forecaster=sarimax, y=y, cv=cv, scoring=loss)

croston = Croston()
croston_results = evaluate(forecaster=croston, y=y, cv=cv, scoring=loss)

prophet = Prophet()
prophet_results = evaluate(forecaster=prophet, y=y, cv=cv, scoring=loss)
# var = VAR()
# var_results = evaluate(forecaster=var, y=y, cv=cv, scoring=loss)

naive = NaiveForecaster(strategy="drift")
naive_results = evaluate(forecaster=naive, y=y, cv=cv, scoring=loss)


print(naive_results["test_MeanAbsolutePercentageError"].mean())

print(ets_results["test_MeanAbsolutePercentageError"].mean())
print(sarimax_results["test_MeanAbsolutePercentageError"].mean())
print(croston_results["test_MeanAbsolutePercentageError"].mean())
print(prophet_results["test_MeanAbsolutePercentageError"].mean())
# print(var_results["test_MeanAbsolutePercentageError"].mean())

