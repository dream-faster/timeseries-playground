#%%

import pandas_datareader.data as web
import pandas as pd
import datetime
import seaborn as sns
import yfinance as yf

yf.pdr_override()
sns.set(rc={"figure.figsize": (11.7, 8.27)})


def get_market_data() -> pd.DataFrame:
    def take_returns(series: pd.Series) -> pd.Series:
        return series.pct_change()

    start_date = "1990-01-01"
    end_date = "2022-10-16"

    sp500 = (
        yf.Ticker("^GSPC")
        .history(interval="1d", start=start_date, end=end_date)["Close"]
        .rename("SP500")
    )
    sp500_returns = take_returns(sp500).rename("SP500_returns")
    sp500_std = sp500_returns.rolling(21).std().rename("SP500_std")
    gld_returns = take_returns(
        yf.Ticker("GOLD").history(interval="1d", start=start_date, end=end_date)[
            "Close"
        ]
    ).rename("GOLD_returns")

    vix = (
        yf.Ticker("^VIX")
        .history(interval="1d", start=start_date, end=end_date)["Close"]
        .rename("VIX")
    )
    vix_returns = take_returns(vix).rename("VIX_returns")
    y30 = (
        yf.Ticker("^TYX")
        .history(interval="1d", start=start_date, end=end_date)["Close"]
        .rename("Y30")
    )
    y30_returns = take_returns(y30).rename("Y30_returns")
    ndx = (
        yf.Ticker("^NDX")
        .history(interval="1d", start=start_date, end=end_date)["Close"]
        .rename("NDX")
    )
    ndx_returns = take_returns(ndx).rename("NDX_returns")

    df = (
        pd.concat(
            [
                sp500,
                sp500_returns,
                sp500_std,
                gld_returns,
                vix,
                vix_returns,
                y30,
                y30_returns,
                ndx,
                ndx_returns,
            ],
            axis=1,
        )
        .fillna(method="ffill")
        .dropna()
    )
    print(f" {df.shape[0]} {df.shape[1]}")
    return df


# %%
