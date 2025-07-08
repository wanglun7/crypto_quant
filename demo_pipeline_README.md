# Demo Pipeline: Binance WebSocket → TimescaleDB

This demo script demonstrates the integration between `BinanceWSCollector` and `TimescaleWriter` for real-time cryptocurrency data collection and storage.

## Overview

The demo connects to Binance's WebSocket API to receive real-time aggregated trade data for BTC/USDT and stores it in TimescaleDB with efficient batching.

## Prerequisites

1. **Python 3.11+** with dependencies installed:
   ```bash
   # Option 1: Using Poetry (recommended)
   poetry install
   
   # Option 2: Using pip
   pip install websockets structlog asyncpg
   ```

2. **Check Requirements**: Run the requirements checker first:
   ```bash
   python check_demo_requirements.py
   ```

3. **TimescaleDB** running locally or remotely. For local testing:
   ```bash
   # Using Docker
   docker run -d --name timescaledb -p 5432:5432 \
     -e POSTGRES_PASSWORD=postgres \
     timescale/timescaledb:latest-pg14
   
   # Create database
   docker exec -it timescaledb psql -U postgres -c "CREATE DATABASE crypto_quant;"
   
   # Enable TimescaleDB extension
   docker exec -it timescaledb psql -U postgres -d crypto_quant \
     -c "CREATE EXTENSION IF NOT EXISTS timescaledb;"
   ```

## Usage

### Basic Usage

Run with default settings (localhost PostgreSQL, unlimited duration):
```bash
python demo_pipeline.py
```

### Custom Database URL

```bash
python demo_pipeline.py --database-url postgresql://user:pass@host:5432/crypto_quant
```

### Run for Specific Duration

```bash
# Run for 60 seconds
python demo_pipeline.py --duration 60
```

### Custom Batch Size

```bash
# Write to database every 500 trades instead of default 1000
python demo_pipeline.py --batch-size 500
```

### Full Example

```bash
python demo_pipeline.py \
  --database-url postgresql://postgres:postgres@localhost:5432/crypto_quant \
  --duration 300 \
  --batch-size 500
```

## What It Does

1. **Connects to Binance WebSocket**: Subscribes to BTC/USDT aggregated trades stream
2. **Processes Trade Data**: Receives real-time trade events with price, quantity, and metadata
3. **Batches for Efficiency**: Accumulates trades in memory before bulk inserting to database
4. **Creates Tables Automatically**: Sets up TimescaleDB hypertable for time-series optimization
5. **Handles Errors Gracefully**: Reconnects on WebSocket failures, retries failed batches
6. **Logs Statistics**: Reports trades/second, total trades, and batches written

## Output Example

```
2024-01-09T10:30:45.123Z [INFO] Starting demo pipeline database_url=default duration=unlimited batch_size=1000
2024-01-09T10:30:45.234Z [INFO] Started BinanceWSCollector
2024-01-09T10:30:45.345Z [INFO] Connected to TimescaleDB batch_size=1000
2024-01-09T10:30:45.456Z [INFO] Connected to Binance WebSocket url=wss://stream.binance.com:9443/ws/btcusdt@aggTrade
2024-01-09T10:30:45.567Z [INFO] Running indefinitely (press Ctrl+C to stop)
2024-01-09T10:30:55.678Z [INFO] Pipeline statistics trades_received=523 batches_written=0 runtime_seconds=10.1 trades_per_second=51.78
2024-01-09T10:31:05.789Z [INFO] Inserted batch of trades count=1000
2024-01-09T10:31:05.890Z [INFO] Pipeline statistics trades_received=1047 batches_written=1 runtime_seconds=20.2 trades_per_second=51.83
```

## Database Schema

The demo creates a `trades` table with the following structure:

```sql
CREATE TABLE trades (
    time TIMESTAMPTZ NOT NULL,
    symbol TEXT NOT NULL,
    price NUMERIC(20, 8) NOT NULL,
    quantity NUMERIC(20, 8) NOT NULL,
    trade_id BIGINT,
    first_trade_id BIGINT,
    last_trade_id BIGINT,
    is_buyer_maker BOOLEAN,
    PRIMARY KEY (time, symbol, trade_id)
);

-- Converted to TimescaleDB hypertable for time-series optimization
SELECT create_hypertable('trades', 'time');
```

## Monitoring

### Check Data in Database

```sql
-- Connect to database
psql -U postgres -d crypto_quant

-- Count total trades
SELECT COUNT(*) FROM trades;

-- View recent trades
SELECT time, symbol, price, quantity 
FROM trades 
ORDER BY time DESC 
LIMIT 10;

-- Check data distribution
SELECT 
    time_bucket('1 minute', time) AS minute,
    COUNT(*) as trade_count,
    AVG(price::numeric) as avg_price,
    SUM(quantity::numeric) as total_volume
FROM trades
WHERE time > NOW() - INTERVAL '10 minutes'
GROUP BY minute
ORDER BY minute DESC;
```

### Performance Metrics

The demo logs performance statistics every 10 seconds:
- Total trades received
- Batches written to database
- Average trades per second
- Runtime duration

## Troubleshooting

### Connection Errors

If you see "Failed to connect to database":
1. Ensure PostgreSQL/TimescaleDB is running
2. Check database URL format
3. Verify network connectivity
4. Check PostgreSQL logs

### WebSocket Disconnections

The collector automatically reconnects with exponential backoff (1s, 2s, 4s... up to 60s).

### High Memory Usage

Reduce `--batch-size` to write more frequently and use less memory.

## Graceful Shutdown

Press `Ctrl+C` to stop the demo. It will:
1. Stop collecting new trades
2. Flush any pending trades to database
3. Log final statistics
4. Close all connections cleanly

## Next Steps

After running the demo, you can:
1. Explore the collected data with SQL queries
2. Build features on top of the raw trade data
3. Create visualizations with Grafana
4. Implement trading strategies using the data
5. Add more data sources (order book, funding rates, etc.)