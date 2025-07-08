import pandas as pd

def order_book_imbalance(bids: pd.DataFrame, asks: pd.DataFrame) -> pd.Series:
    """Compute order book imbalance.

    Parameters
    ----------
    bids : pd.DataFrame
        DataFrame with columns ["price", "size"].
    asks : pd.DataFrame
        DataFrame with columns ["price", "size"].

    Returns
    -------
    pd.Series
        Single value series representing imbalance.
    """
    if bids.empty or asks.empty:
        raise ValueError("bids and asks must not be empty")
    bid_volume = bids["size"].sum()
    ask_volume = asks["size"].sum()
    if bid_volume + ask_volume == 0:
        return pd.Series(0.0)
    imbalance = (bid_volume - ask_volume) / (bid_volume + ask_volume)
    return pd.Series(imbalance)
