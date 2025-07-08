import pandas as pd
from src.features.micro import order_book_imbalance


def test_order_book_imbalance():
    bids = pd.DataFrame({"price": [100, 99], "size": [1, 2]})
    asks = pd.DataFrame({"price": [101, 102], "size": [1, 1]})
    result = order_book_imbalance(bids, asks)
    assert abs(result.iloc[0] - (3 - 2) / (3 + 2)) < 1e-6
