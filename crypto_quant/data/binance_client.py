"""Binance REST API client for historical data collection."""

import asyncio
import time
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
import pandas as pd
import ccxt
import asyncpg
import structlog

logger = structlog.get_logger(__name__)


class BinanceDataClient:
    """Binance client for fetching historical kline data."""
    
    def __init__(self):
        """Initialize Binance client."""
        self.exchange = ccxt.binance({
            'apiKey': '',
            'secret': '',
            'sandbox': False,
            'enableRateLimit': True,
        })
        
    def fetch_ohlcv(
        self,
        symbol: str = 'BTC/USDT',
        timeframe: str = '1h',
        since: Optional[datetime] = None,
        limit: int = 1000
    ) -> pd.DataFrame:
        """Fetch OHLCV data from Binance.
        
        Args:
            symbol: Trading pair symbol
            timeframe: Timeframe (1m, 5m, 15m, 1h, 4h, 1d)
            since: Start time
            limit: Number of candles to fetch (max 1000)
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            since_ms = None
            if since:
                since_ms = int(since.timestamp() * 1000)
                
            ohlcv = self.exchange.fetch_ohlcv(
                symbol=symbol,
                timeframe=timeframe,
                since=since_ms,
                limit=limit
            )
            
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
            df['symbol'] = symbol
            df['timeframe'] = timeframe
            
            logger.info(
                "Fetched OHLCV data",
                symbol=symbol,
                timeframe=timeframe,
                rows=len(df),
                start=df['timestamp'].min(),
                end=df['timestamp'].max()
            )
            
            return df
            
        except Exception as e:
            logger.error("Failed to fetch OHLCV data", error=str(e))
            raise
            
    def fetch_historical_data(
        self,
        start_date: datetime,
        symbol: str = 'BTC/USDT',
        timeframe: str = '1h',
        end_date: Optional[datetime] = None,
        save_to_file: bool = True,
        file_path: Optional[str] = None
    ) -> pd.DataFrame:
        """Fetch historical data in batches.
        
        Args:
            symbol: Trading pair symbol
            timeframe: Timeframe 
            start_date: Start date
            end_date: End date (default: now)
            save_to_file: Whether to save to parquet file
            file_path: Custom file path
            
        Returns:
            Complete historical DataFrame
        """
        if end_date is None:
            end_date = datetime.now()
            
        logger.info(
            "Starting historical data fetch",
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date
        )
        
        all_data = []
        current_date = start_date
        
        # Calculate time delta based on timeframe
        timeframe_deltas = {
            '1m': timedelta(minutes=1000),
            '5m': timedelta(minutes=5000), 
            '15m': timedelta(minutes=15000),
            '1h': timedelta(hours=1000),
            '4h': timedelta(hours=4000),
            '1d': timedelta(days=1000)
        }
        
        delta = timeframe_deltas.get(timeframe, timedelta(hours=1000))
        
        while current_date < end_date:
            batch_end = min(current_date + delta, end_date)
            
            try:
                batch_data = self.fetch_ohlcv(
                    symbol=symbol,
                    timeframe=timeframe,
                    since=current_date,
                    limit=1000
                )
                
                if not batch_data.empty:
                    # Ensure timezone aware comparison
                    current_date_tz = pd.Timestamp(current_date, tz='UTC')
                    batch_end_tz = pd.Timestamp(batch_end, tz='UTC')
                    
                    # Filter data within the requested range
                    batch_data = batch_data[
                        (batch_data['timestamp'] >= current_date_tz) &
                        (batch_data['timestamp'] < batch_end_tz)
                    ]
                    all_data.append(batch_data)
                    
                    # Update current_date to the last timestamp + 1 interval
                    if not batch_data.empty:
                        last_ts = batch_data['timestamp'].max()
                        current_date = last_ts.to_pydatetime().replace(tzinfo=None) + pd.Timedelta(timeframe).to_pytimedelta()
                    else:
                        current_date = batch_end
                else:
                    current_date = batch_end
                    
                # Rate limiting
                time.sleep(0.1)
                
            except Exception as e:
                logger.error(
                    "Error fetching batch",
                    current_date=current_date,
                    error=str(e)
                )
                # Continue with next batch
                current_date += delta
                time.sleep(1)
                
        if not all_data:
            logger.warning("No data fetched")
            return pd.DataFrame()
            
        # Combine all batches
        df = pd.concat(all_data, ignore_index=True)
        df = df.drop_duplicates(subset=['timestamp']).sort_values('timestamp').reset_index(drop=True)
        
        logger.info(
            "Historical data fetch completed",
            total_rows=len(df),
            start=df['timestamp'].min(),
            end=df['timestamp'].max()
        )
        
        # Save to file
        if save_to_file:
            if file_path is None:
                file_path = f"data/{symbol.replace('/', '_')}_{timeframe}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.csv"
            
            # Convert to CSV if file extension is csv, otherwise use parquet if available
            if file_path.endswith('.csv'):
                df.to_csv(file_path, index=False)
            else:
                try:
                    df.to_parquet(file_path, index=False)
                except ImportError:
                    # Fallback to CSV if parquet is not available
                    csv_path = file_path.replace('.parquet', '.csv')
                    df.to_csv(csv_path, index=False)
                    file_path = csv_path
                    
            logger.info("Data saved to file", file_path=file_path)
            
        return df


class TimescaleDBManager:
    """TimescaleDB connection and data management."""
    
    def __init__(self, database_url: str):
        """Initialize database manager."""
        self.database_url = database_url
        self.pool: Optional[asyncpg.Pool] = None
        
    async def connect(self):
        """Connect to database."""
        self.pool = await asyncpg.create_pool(self.database_url)
        logger.info("Connected to TimescaleDB")
        
    async def disconnect(self):
        """Disconnect from database.""" 
        if self.pool:
            await self.pool.close()
            logger.info("Disconnected from TimescaleDB")
            
    async def create_tables(self):
        """Create necessary tables."""
        if not self.pool:
            raise RuntimeError("Not connected to database")
            
        async with self.pool.acquire() as conn:
            # Create klines table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS klines (
                    timestamp TIMESTAMPTZ NOT NULL,
                    symbol VARCHAR(20) NOT NULL,
                    timeframe VARCHAR(10) NOT NULL,
                    open DECIMAL(20,8) NOT NULL,
                    high DECIMAL(20,8) NOT NULL,
                    low DECIMAL(20,8) NOT NULL,
                    close DECIMAL(20,8) NOT NULL,
                    volume DECIMAL(20,8) NOT NULL,
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    PRIMARY KEY (timestamp, symbol, timeframe)
                );
            """)
            
            # Create hypertable if not exists
            try:
                await conn.execute("SELECT create_hypertable('klines', 'timestamp', if_not_exists => TRUE);")
            except Exception as e:
                logger.warning("Failed to create hypertable", error=str(e))
                
            # Create indexes
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_klines_symbol_timeframe_timestamp 
                ON klines (symbol, timeframe, timestamp);
            """)
            
            logger.info("Database tables created/verified")
            
    async def insert_klines(self, df: pd.DataFrame):
        """Insert klines data into database."""
        if not self.pool:
            raise RuntimeError("Not connected to database")
            
        if df.empty:
            return
            
        async with self.pool.acquire() as conn:
            # Prepare data for insertion
            records = [
                (
                    row['timestamp'],
                    row['symbol'], 
                    row['timeframe'],
                    float(row['open']),
                    float(row['high']),
                    float(row['low']),
                    float(row['close']),
                    float(row['volume'])
                )
                for _, row in df.iterrows()
            ]
            
            # Use COPY for efficient bulk insert
            await conn.copy_records_to_table(
                'klines',
                records=records,
                columns=['timestamp', 'symbol', 'timeframe', 'open', 'high', 'low', 'close', 'volume']
            )
            
            logger.info("Inserted klines data", rows=len(records))


async def fetch_and_store_historical_data():
    """Fetch historical data and store to database."""
    client = BinanceDataClient()
    db = TimescaleDBManager("postgresql://crypto_quant:crypto_quant_2024@localhost:5432/crypto_quant")
    
    try:
        await db.connect()
        await db.create_tables()
        
        # Define data to fetch
        symbol = 'BTC/USDT'
        timeframes = ['1h', '4h', '1d']
        start_date = datetime(2023, 1, 1)
        end_date = datetime.now()
        
        for timeframe in timeframes:
            logger.info("Fetching data", symbol=symbol, timeframe=timeframe)
            
            df = client.fetch_historical_data(
                start_date=start_date,
                symbol=symbol,
                timeframe=timeframe,
                end_date=end_date,
                save_to_file=True
            )
            
            if not df.empty:
                await db.insert_klines(df)
                
    finally:
        await db.disconnect()


async def fetch_1min_data_for_cnn_lstm(
    symbol: str = 'BTC/USDT',
    months_back: int = 6,
    save_to_db: bool = True,
    save_to_file: bool = True
) -> pd.DataFrame:
    """Fetch 1-minute data specifically for CNN-LSTM model training.
    
    Args:
        symbol: Trading pair symbol
        months_back: Number of months of historical data to fetch
        save_to_db: Whether to save to TimescaleDB
        save_to_file: Whether to save to local file
        
    Returns:
        DataFrame with 1-minute OHLCV data
    """
    client = BinanceDataClient()
    
    # Calculate date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=months_back * 30)
    
    logger.info(
        "Fetching 1-minute data for CNN-LSTM",
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        months_back=months_back
    )
    
    # Fetch 1-minute data
    df = client.fetch_historical_data(
        start_date=start_date,
        symbol=symbol,
        timeframe='1m',
        end_date=end_date,
        save_to_file=save_to_file,
        file_path=f"data/{symbol.replace('/', '_')}_1m_cnn_lstm_{months_back}months.parquet"
    )
    
    if save_to_db and not df.empty:
        db = TimescaleDBManager("postgresql://crypto_quant:crypto_quant_2024@localhost:5432/crypto_quant")
        try:
            await db.connect()
            await db.create_tables()
            await db.insert_klines(df)
            logger.info("Data saved to database", rows=len(df))
        finally:
            await db.disconnect()
    
    logger.info(
        "1-minute data fetch completed",
        total_rows=len(df),
        start=df['timestamp'].min() if not df.empty else None,
        end=df['timestamp'].max() if not df.empty else None
    )
    
    return df


if __name__ == "__main__":
    import os
    os.makedirs("data", exist_ok=True)
    asyncio.run(fetch_and_store_historical_data())