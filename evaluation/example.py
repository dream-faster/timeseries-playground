from eda.utils import generating_arima_synthetic_data
from eda.modeling import fit_arima_model

"example"
ds = generating_arima_synthetic_data(target_col="example")

model_res = fit_arima_model(ds)

model_res.predict(start=950, end=1010)