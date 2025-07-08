from src.backtest.strategy import run_backtest

if __name__ == "__main__":
    results = run_backtest("2018-01-01", "2024-01-01")
    for k, v in results.items():
        print(f"{k}: {v}")
