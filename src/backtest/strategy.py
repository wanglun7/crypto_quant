import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from ..data.prices import load_btc_price
from ..data.blockchain import load_active_addresses


def prepare_data(start: str, end: str) -> pd.DataFrame:
    price = load_btc_price(start, end)
    active = load_active_addresses(start, end).reindex(price.index).fillna(method="ffill")
    df = pd.DataFrame({"price": price, "active": active})
    df["return"] = df["price"].pct_change()
    df["active_chg"] = df["active"].pct_change()
    df = df.dropna()
    return df


def run_backtest(start: str, end: str) -> dict:
    df = prepare_data(start, end)
    # create binary label for next day return
    df["label"] = (df["return"].shift(-1) > 0).astype(int)
    train = df.iloc[:-200]
    test = df.iloc[-200:-1]

    X_train = train[["return", "active_chg"]]
    y_train = train["label"]
    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)

    X_test = test[["return", "active_chg"]]
    y_test = test["label"]
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    # simple strategy: if model predicts 1 -> long, else flat
    signals = pd.Series(preds, index=X_test.index)
    strategy_ret = (signals.shift(1) * df.loc[signals.index, "return"]).dropna()
    buy_hold_ret = df.loc[strategy_ret.index, "return"]

    strategy_cum = (1 + strategy_ret).cumprod() - 1
    buy_hold_cum = (1 + buy_hold_ret).cumprod() - 1
    return {
        "accuracy": acc,
        "strategy_return": strategy_cum.iloc[-1],
        "buy_and_hold": buy_hold_cum.iloc[-1],
    }
