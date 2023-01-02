import pandas as pd


def get_energy_data() -> pd.DataFrame:
    train = pd.read_csv("data/energy.csv", index_col="time", parse_dates=True)
    train = train[~train.index.duplicated(keep="first")]
    train = train[train["dataset_id"] == 1]
    return train
