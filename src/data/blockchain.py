import io
import pandas as pd
import requests

BLOCKCHAIN_URL = "https://api.blockchain.info/charts/n-unique-addresses"


def load_active_addresses(start: str, end: str) -> pd.Series:
    """Load active addresses from blockchain.com."""
    params = {"timespan": "all", "format": "csv"}
    resp = requests.get(BLOCKCHAIN_URL, params=params, timeout=10)
    resp.raise_for_status()
    df = pd.read_csv(io.StringIO(resp.text), names=["date", "active"], parse_dates=["date"])
    df["ts"] = (df["date"].astype("int64") // 10 ** 6)
    df = df.set_index("ts").sort_index()
    # filter range
    start_ts = pd.Timestamp(start, tz="UTC").value // 10 ** 6
    end_ts = pd.Timestamp(end, tz="UTC").value // 10 ** 6
    df = df.loc[start_ts:end_ts]
    return df["active"]
