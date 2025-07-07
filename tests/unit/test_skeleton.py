import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import pandas as pd
import pytest

from src.data.onchain import load_active_addresses, load_exchange_netflow


def test_load_active_addresses():
    df = load_active_addresses("2020-01-01", "2020-01-02")
    assert not df.empty
    assert list(df.columns) == ["active_addresses"]
    assert df.index.is_monotonic_increasing


def test_load_exchange_netflow():
    df = load_exchange_netflow("2020-01-01", "2020-01-02")
    assert not df.empty
    assert list(df.columns) == ["netflow_btc"]
    assert df.index.is_monotonic_increasing
