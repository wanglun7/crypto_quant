import pytest
from datetime import datetime, timedelta

from crypto_quant.data.fetch import fetch_ohlcv
from crypto_quant.features.ta_multi_scale import generate_features
from crypto_quant.strategy.baseline import run_backtest   # 待 Claude 实现

SYMBOL = "BTC/USDT"
_START = datetime(2025, 1, 1)
_END   = _START + timedelta(hours=2)     # 120 根 1m

@pytest.fixture(scope="module")
def base_data():
    ohlcv = fetch_ohlcv(SYMBOL, "1m", _START, _END)
    feats = generate_features(ohlcv, scales=["1m"])
    return ohlcv, feats

def test_metrics_threshold(base_data):
    ohlcv, feats = base_data
    stats = run_backtest(ohlcv, feats)
    assert stats["sharpe"] >= 0.8
    assert stats["max_dd"]  <= 0.25

def test_return_positive(base_data):
    ohlcv, feats = base_data
    stats = run_backtest(ohlcv, feats)
    assert stats["total_return"] > 0

def test_trade_count(base_data):
    ohlcv, feats = base_data
    stats = run_backtest(ohlcv, feats)
    assert stats["trade_count"] > 0

def test_missing_column_error(base_data):
    ohlcv, feats = base_data
    feats_bad = feats.drop(columns=["ema_fast_1m"])
    with pytest.raises(ValueError):
        run_backtest(ohlcv, feats_bad)
