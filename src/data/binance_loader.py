import requests
import pandas as pd
from typing import Optional, List

BASE_URL = "https://data-api.binance.vision/api/v3/klines"


def _fetch_klines(symbol: str, interval: str, start_time: Optional[int] = None,
                  end_time: Optional[int] = None, limit: int = 1000) -> List[list]:
    params = {
        "symbol": symbol.upper(),
        "interval": interval,
        "limit": limit,
    }
    if start_time is not None:
        params["startTime"] = int(start_time)
    if end_time is not None:
        params["endTime"] = int(end_time)
    resp = requests.get(BASE_URL, params=params)
    resp.raise_for_status()
    return resp.json()


def load_klines(symbol: str, interval: str,
                start_time: Optional[int] = None,
                end_time: Optional[int] = None,
                limit: int = 1000) -> pd.DataFrame:
    """Load klines from Binance REST API.

    Parameters
    ----------
    symbol : str
        Trading pair symbol (e.g., ``"BTCUSDT"``).
    interval : str
        Kline interval (e.g., ``"1h"``).
    start_time : int, optional
        Start time in UTC milliseconds.
    end_time : int, optional
        End time in UTC milliseconds.
    limit : int, default 1000
        Number of rows to request per API call (max 1000).

    Returns
    -------
    pandas.DataFrame
        Data indexed by ``timestamp`` in UTC milliseconds.
    """
    data: List[list] = []
    fetch_start = start_time
    while True:
        chunk = _fetch_klines(symbol, interval, fetch_start, end_time, limit)
        if not chunk:
            break
        data.extend(chunk)
        if len(chunk) < limit:
            break
        fetch_start = chunk[-1][0] + 1
        if end_time is not None and fetch_start > end_time:
            break

    cols = [
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "close_time",
        "quote_asset_volume",
        "number_of_trades",
        "taker_buy_base",
        "taker_buy_quote",
        "ignore",
    ]
    df = pd.DataFrame(data, columns=cols)
    float_cols = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "quote_asset_volume",
        "taker_buy_base",
        "taker_buy_quote",
    ]
    df[float_cols] = df[float_cols].astype("float64")
    int_cols = ["timestamp", "close_time", "number_of_trades", "ignore"]
    df[int_cols] = df[int_cols].astype("int64")
    df.set_index("timestamp", inplace=True)
    df = df[~df.index.duplicated(keep="first")]
    df.sort_index(inplace=True)
    return df
