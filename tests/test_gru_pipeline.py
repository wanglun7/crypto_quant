import os, pickle, pytest, torch, numpy as np
from datetime import datetime, timedelta, timezone
from pathlib import Path

from crypto_quant.data.fetch import fetch_ohlcv
from crypto_quant.features.ta_multi_scale import generate_features
from crypto_quant.data.dataset import build_dataset
from crypto_quant.models.gru_signal import train_gru, load_gru
from crypto_quant.strategy.gru_strategy import run_backtest, run_baseline

CACHE = Path("/tmp/btc_ohlcv_feat_72h.pkl")

@pytest.fixture(scope="session")
def sample_data():
    if CACHE.exists():
        return pickle.loads(CACHE.read_bytes())
    end   = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    start = end - timedelta(hours=72)
    ohlcv = fetch_ohlcv("BTC/USDT", "1m", start, end)
    feats = generate_features(ohlcv, scales=["1m", "5m"])
    CACHE.write_bytes(pickle.dumps((ohlcv, feats)))
    return ohlcv, feats

def test_dataset_shapes(sample_data):
    ohlcv, feats = sample_data
    X, y = build_dataset(ohlcv, feats, lookback=60, horizon=5)
    assert X.shape[0] == y.shape[0]
    assert X.shape[2] == feats.shape[1] + 5  # Original features + 5 new features
    assert not np.isnan(X).any()

def test_training_convergence(sample_data):
    X, y = build_dataset(*sample_data, lookback=60, horizon=5)
    metrics = train_gru(X, y, epochs=10, batch_size=128)
    assert metrics["val_bacc"] >= 0.52
    assert Path(metrics["ckpt_path"]).exists()

def test_prediction_no_nan(sample_data):
    model = load_gru()        # 默认载入 latest.pt
    X, _ = build_dataset(*sample_data, lookback=60, horizon=5)
    with torch.no_grad():
        proba = model.predict_proba(torch.tensor(X[:64]).float())
    assert not np.isnan(proba.numpy()).any()

def test_backtest_outperforms_baseline(sample_data):
    ohlcv, feats = sample_data
    stats_gru  = run_backtest(ohlcv, feats, ckpt_path="latest.pt")
    stats_base = run_baseline(ohlcv, feats)
    assert stats_gru["sharpe"] >= stats_base["sharpe"] + 0.10
    assert stats_gru["max_dd"]  <= 0.25

def test_no_lookahead(sample_data):
    ohlcv, feats = sample_data
    X, _ = build_dataset(ohlcv, feats, lookback=60, horizon=5)
    # 最后一个 X 使用的特征应不含未来信息
    last_index_in_X = 60 + len(X) - 1    # 最后一条样本对应原始索引
    assert last_index_in_X + 5 - 1 < len(ohlcv)   # horizon 内未越界
