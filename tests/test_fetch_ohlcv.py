import pandas as pd
import pytest
from datetime import datetime, timedelta
from crypto_quant.data.fetch import fetch_ohlcv   # 待 Claude 实现

_SYMBOL = "BTC/USDT"
_TIMEFRAME = "1m"
_START = datetime(2025, 1, 1)
_END   = datetime(2025, 1, 1, 0, 30)  # 30 分钟样例，行数应为 30

def test_row_count():
    df = fetch_ohlcv(_SYMBOL, _TIMEFRAME, _START, _END)
    expected_rows = int((_END - _START) / timedelta(minutes=1))
    assert len(df) == expected_rows

def test_monotonic_index():
    df = fetch_ohlcv(_SYMBOL, _TIMEFRAME, _START, _END)
    assert isinstance(df.index, pd.DatetimeIndex)
    assert df.index.is_monotonic_increasing
    assert df.index.freq is None  # 不自动填充 freq，确保原始精度

def test_no_gaps_or_nans():
    df = fetch_ohlcv(_SYMBOL, _TIMEFRAME, _START, _END)
    assert not df.isna().any().any(), "NaNs detected"
    # 连续性校验
    delta = df.index.to_series().diff().dropna().unique()
    assert len(delta) == 1 and delta[0] == pd.Timedelta(minutes=1)

def test_value_error_on_gap(monkeypatch):
    from crypto_quant.data import fetch as mod
    # 强制打洞：模拟缺口触发
    # 保存原始函数引用，避免递归
    original_raw_fetch = mod._raw_fetch
    def _bad_fetch(*args, **kwargs):
        out = original_raw_fetch(*args, **kwargs)  # 调用保存的原始函数
        return out.iloc[:-1]  # 少一行
    monkeypatch.setattr(mod, "_raw_fetch", _bad_fetch)
    with pytest.raises(ValueError):
        fetch_ohlcv(_SYMBOL, _TIMEFRAME, _START, _END)
