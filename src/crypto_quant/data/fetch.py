import ccxt
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional


def _datetime_to_ms(dt: datetime) -> int:
    """Convert datetime to milliseconds timestamp."""
    return int(dt.timestamp() * 1000)


def _ms_to_datetime(ms: int) -> datetime:
    """Convert milliseconds timestamp to datetime."""
    return datetime.utcfromtimestamp(ms / 1000)


def _raw_fetch(exchange, symbol: str, timeframe: str, since: int, limit: int = 1000) -> pd.DataFrame:
    """
    Raw fetch function that calls ccxt API.
    Separated for easier testing/mocking.
    """
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)
    if not ohlcv:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    
    # Convert to DataFrame
    df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df.set_index("timestamp", inplace=True)
    return df


# Store original function reference for testing
# This allows tests to access the original function even after monkeypatching
_raw_fetch_original = _raw_fetch


def fetch_ohlcv(symbol: str, timeframe: str, start: datetime, end: datetime) -> pd.DataFrame:
    """
    Fetch OHLCV data from Binance for the given symbol and time range.
    
    Args:
        symbol: Trading pair (e.g., "BTC/USDT")
        timeframe: Time interval (e.g., "1m", "5m", "1h")
        start: Start time (inclusive)
        end: End time (exclusive)
    
    Returns:
        DataFrame with DatetimeIndex and columns: ["open", "high", "low", "close", "volume"]
    
    Raises:
        ValueError: If there are gaps in the data or missing values
    """
    exchange = ccxt.binance({"enableRateLimit": True})
    
    # Convert times to milliseconds
    start_ms = _datetime_to_ms(start)
    end_ms = _datetime_to_ms(end)
    
    # Parse timeframe to get interval in milliseconds
    timeframe_ms = {
        "1m": 60 * 1000,
        "5m": 5 * 60 * 1000,
        "15m": 15 * 60 * 1000,
        "30m": 30 * 60 * 1000,
        "1h": 60 * 60 * 1000,
        "4h": 4 * 60 * 60 * 1000,
        "1d": 24 * 60 * 60 * 1000,
    }.get(timeframe)
    
    if not timeframe_ms:
        raise ValueError(f"Unsupported timeframe: {timeframe}")
    
    # Calculate expected number of candles
    expected_candles = int((end_ms - start_ms) / timeframe_ms)
    
    # Fetch data in chunks (Binance limit is typically 1000 candles per request)
    all_data = []
    current_start = start_ms
    
    while current_start < end_ms:
        # Calculate how many candles we need for this chunk
        remaining_ms = end_ms - current_start
        chunk_size = min(1000, int(remaining_ms / timeframe_ms))
        
        if chunk_size <= 0:
            break
            
        # Fetch chunk
        df_chunk = _raw_fetch(exchange, symbol, timeframe, current_start, chunk_size)
        
        if df_chunk.empty:
            # If we get no data, there might be a gap
            break
            
        all_data.append(df_chunk)
        
        # Move to next chunk
        last_timestamp = df_chunk.index[-1]
        current_start = int(last_timestamp.timestamp() * 1000) + timeframe_ms
    
    if not all_data:
        raise ValueError("No data received from exchange")
    
    # Combine all chunks
    df = pd.concat(all_data)
    
    # Remove any data beyond our end time
    if end.tzinfo is not None:
        end_ts = pd.Timestamp(end)
    else:
        end_ts = pd.Timestamp(end, tz="UTC")
    df = df[df.index < end_ts]
    
    # Check for completeness
    if len(df) != expected_candles:
        raise ValueError(f"Data gap detected: expected {expected_candles} candles, got {len(df)}")
    
    # Check for NaN values
    if df.isna().any().any():
        raise ValueError("NaN values detected in data")
    
    # Check time gaps
    if len(df) > 1:
        time_diffs = df.index.to_series().diff().dropna()
        expected_diff = pd.Timedelta(milliseconds=timeframe_ms)
        
        # Allow for minor precision differences (1ms tolerance)
        if not all(abs(diff - expected_diff) <= pd.Timedelta(milliseconds=1) for diff in time_diffs):
            raise ValueError("Time gaps detected in data")
    
    # Ensure monotonic increasing index
    if not df.index.is_monotonic_increasing:
        raise ValueError("Index is not monotonic increasing")
    
    return df