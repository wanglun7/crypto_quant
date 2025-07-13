import pandas as pd
import pytest
from datetime import datetime, timedelta

from crypto_quant.data.fetch import fetch_ohlcv
from crypto_quant.features.ta_multi_scale import generate_features   # 待实现

SYMBOL = "BTC/USDT"
_START = datetime(2025, 1, 1)
_END   = _START + timedelta(hours=2)          # 120 行 1m 数据

@pytest.fixture(scope="module")
def ohlcv_1m():
    return fetch_ohlcv(SYMBOL, "1m", _START, _END)

def test_shape_and_index(ohlcv_1m):
    feat = generate_features(ohlcv_1m, scales=["1m","5m","15m"])
    assert len(feat) == len(ohlcv_1m)
    pd.testing.assert_index_equal(feat.index, ohlcv_1m.index)

def test_no_nans(ohlcv_1m):
    feat = generate_features(ohlcv_1m, scales=["1m","5m"])
    assert not feat.isna().any().any()

def test_columns_exist(ohlcv_1m):
    feat = generate_features(ohlcv_1m, scales=["1m"])
    required = {"ema_fast_1m","ema_slow_1m","macd_1m","macd_signal_1m",
                "rsi_1m","bb_width_1m","adx_1m"}
    assert required.issubset(set(feat.columns))

def test_unknown_scale_error(ohlcv_1m):
    with pytest.raises(ValueError):
        generate_features(ohlcv_1m, scales=["2m"])     # 不支持的粒度
