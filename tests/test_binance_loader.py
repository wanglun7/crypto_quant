import sys, os; sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import pandas as pd
from src.data.binance_loader import load_klines


def test_load_klines_structure():
    df = load_klines("BTCUSDT", "1h", limit=10)
    assert isinstance(df, pd.DataFrame)
    assert df.index.is_monotonic_increasing
    assert not df.index.duplicated().any()
    assert df.index.dtype == 'int64'
    assert df['close'].dtype == 'float64'
