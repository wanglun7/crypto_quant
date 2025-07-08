import pandas as pd
from pandas_datareader import data as web


def load_btc_price(start: str, end: str) -> pd.Series:
    """Load BTC/USD daily close price from FRED."""
    df = web.DataReader("CBBTCUSD", "fred", start, end)
    df = df.rename(columns={"CBBTCUSD": "close"})
    df.index = pd.to_datetime(df.index, utc=True)
    df.index = (df.index.view("int64") // 10 ** 6)
    return df["close"]
