# Crypto-Quant Demo Pipeline

This demo showcases the integration between `BinanceWSCollector` and `TimescaleWriter` for real-time cryptocurrency data collection and storage.

## Features

- **Real-time data collection**: Connects to Binance WebSocket for BTC/USDT aggregated trades
- **Batch processing**: Efficient batch writes to TimescaleDB with configurable batch size
- **Monitoring**: Real-time statistics reporting (trades/second, pending trades)
- **Error handling**: Automatic reconnection on WebSocket failures
- **Graceful shutdown**: Proper cleanup and data flushing on interruption

## Quick Start

### 1. Start TimescaleDB

```bash
# Start test database
docker-compose -f docker-compose.test.yml up -d

# Or use your own TimescaleDB instance
```

### 2. Install Dependencies

```bash
poetry install
```

### 3. Run the Demo

```bash
# Basic usage (connects to localhost:5432)
python demo_pipeline.py

# With custom database
python demo_pipeline.py --database-url postgresql://test:test@localhost:5433/crypto_quant_test

# Run for specific duration
python demo_pipeline.py --duration 60

# Custom batch size
python demo_pipeline.py --batch-size 500
```

## Command Line Options

- `--database-url`: PostgreSQL/TimescaleDB connection URL
- `--batch-size`: Number of trades to batch before writing (default: 1000)
- `--duration`: Duration in seconds to run (default: infinite)
- `--log-level`: Logging level (DEBUG, INFO, WARNING, ERROR)

## Example Output

```json
{
  "event": "Pipeline statistics",
  "level": "info",
  "timestamp": "2024-01-01T10:00:00.000Z",
  "trades_received": 1500,
  "trades_written": 1500,
  "pending_trades": 0,
  "elapsed_seconds": 60.5,
  "trades_per_second": 24.79
}
```

## Integration Tests

The demo includes comprehensive integration tests that use mocked WebSocket connections with real TimescaleDB:

```bash
# Start test database
docker-compose -f docker-compose.test.yml up -d

# Run integration tests
pytest tests/test_integration_pipeline.py -v

# Run with coverage
pytest tests/test_integration_pipeline.py --cov=crypto_quant
```

### Test Coverage

- **Full pipeline integration**: Mocked WebSocket → Real TimescaleDB
- **Error handling**: Connection failures and recovery
- **Batch processing**: Verification of batching logic
- **Graceful shutdown**: Proper cleanup and data flushing
- **Data integrity**: Verification of stored trade data

## Architecture

```
Binance WebSocket → BinanceWSCollector → TimescaleWriter → TimescaleDB
     (Real-time)       (Async Stream)     (Batch Insert)    (Hypertable)
```

### Key Components

1. **BinanceWSCollector**: Handles WebSocket connection, reconnection, and data streaming
2. **TimescaleWriter**: Manages database connections, batching, and efficient inserts
3. **PipelineStats**: Tracks performance metrics and provides monitoring
4. **Signal Handlers**: Ensures graceful shutdown on interruption

## Database Schema

The demo creates a TimescaleDB hypertable for optimal time-series performance:

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

SELECT create_hypertable('trades', 'time');
```

## Production Considerations

- **Resource monitoring**: Monitor connection pool usage and batch processing
- **Error alerting**: Set up alerts for connection failures and write errors
- **Data retention**: Configure TimescaleDB retention policies
- **Scaling**: Consider horizontal scaling for multiple symbols
- **Backup**: Implement proper backup strategies for trade data

## Stopping the Demo

Press `Ctrl+C` for graceful shutdown. The pipeline will:
1. Stop accepting new trades
2. Flush all pending batches to database
3. Close WebSocket connection
4. Close database connections
5. Report final statistics