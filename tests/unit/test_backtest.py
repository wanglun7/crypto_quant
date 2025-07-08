from src.backtest.strategy import run_backtest


def test_run_backtest_smoke():
    results = run_backtest("2020-01-01", "2022-01-01")
    assert "strategy_return" in results
    assert "buy_and_hold" in results
