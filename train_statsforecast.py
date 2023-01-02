from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA, MSTL
from data.energy import get_energy_data

data = get_energy_data()
data["unique_id"] = 1
data["ds"] = data.index
data["y"] = data["P"]
data = data[["y", "ds", "unique_id"]]


sf = StatsForecast(
    models=[
        MSTL(season_length=96),
        # AutoARIMA(season_length=96)
    ],
    freq="15min",
)

sf.fit(data)
sf.predict(h=12, level=[95])

crossvaldation_df = sf.cross_validation(df=data, h=1, step_size=24, n_windows=10)
print(crossvaldation_df)