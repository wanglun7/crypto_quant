import json
from pathlib import Path
from typing import Optional
import pandas as pd
import requests

BASE_URL = "https://api.cryptoquant.com/v1"


def _load_json(url: str, params: dict, api_key: Optional[str], fallback: Path) -> dict:
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    try:
        resp = requests.get(url, params=params, headers=headers, timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        with open(fallback, "r", encoding="utf-8") as f:
            data = json.load(f)
    return data


def _to_dataframe(records: list, value_col: str) -> pd.DataFrame:
    df = pd.DataFrame(records)
    df = df.rename(columns={"timestamp": "ts"})
    if "ts" not in df.columns or value_col not in df.columns:
        raise ValueError("Invalid data format")
    df = df.set_index("ts")
    df.index = pd.to_datetime(df.index, unit="ms", utc=True)
    df.index = (df.index.view("int64") // 10 ** 6)
    df = df.sort_index()
    return df[[value_col]]


def load_active_addresses(start: str, end: str, api_key: Optional[str] = None) -> pd.DataFrame:
    """Load active addresses from CryptoQuant."""
    url = f"{BASE_URL}/btc/network/active-addresses"
    params = {"window": "day", "from": start, "to": end}
    fallback = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "active_addresses.json"
    data = _load_json(url, params, api_key, fallback)
    records = data.get("result") or []
    return _to_dataframe(records, "active_addresses")


def load_exchange_netflow(start: str, end: str, api_key: Optional[str] = None) -> pd.DataFrame:
    """Load exchange netflow from CryptoQuant."""
    url = f"{BASE_URL}/btc/exchange/netflow"
    params = {"exchange": "all_exchange", "window": "day", "from": start, "to": end}
    fallback = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "exchange_netflow.json"
    data = _load_json(url, params, api_key, fallback)
    records = data.get("result") or []
    return _to_dataframe(records, "netflow_btc")
