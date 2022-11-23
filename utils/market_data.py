#%%

import pandas_datareader.data as web
import pandas as pd
import datetime
import seaborn as sns
sns.set(rc={'figure.figsize':(11.7,8.27)})

def get_market_data() -> pd.DataFrame:

    def clean_yfinance_data(df) -> pd.Series:
        df = df['Adj Close']
        df = df.rename('Close')
        return df

    def take_returns(series: pd.Series) -> pd.Series:
        return series.pct_change()

    start_date = datetime.datetime(1990, 1, 1)
    end_date = datetime.datetime(2022,10,16)

    sp500 = web.DataReader('^GSPC', 'yahoo', start_date, end_date)['Adj Close'].rename('SP500')
    sp500_returns = take_returns(web.DataReader('^GSPC', 'yahoo', start_date, end_date)['Adj Close']).rename('SP500_returns')
    sp500_std = sp500_returns.rolling(21).std().rename('SP500_std')
    gld_returns = take_returns(web.DataReader('GOLD', 'yahoo', start_date, end_date)['Adj Close']).rename('GOLD_returns')
    vix = web.DataReader('VIXCLS', 'fred', start_date, end_date)['VIXCLS'].rename('VIX')
    vix_returns = take_returns(vix).rename("VIX_returns")
    y30 = web.DataReader('DGS30', 'fred', start_date, end_date)['DGS30'].rename('Y30')
    y30_returns = take_returns(y30).rename("Y30_returns")
    ndx = web.DataReader("^NDX", 'yahoo', start_date, end_date)['Adj Close'].rename('NDX')
    ndx_returns = take_returns(ndx).rename("NDX_returns")

    df = pd.concat([sp500, sp500_returns, sp500_std, gld_returns, vix, vix_returns, y30, y30_returns, ndx, ndx_returns], axis=1).fillna(method='ffill').dropna()
    print(f" {df.shape[0]} {df.shape[1]}")
    return df

# %%
