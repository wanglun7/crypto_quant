# Storage Module

This module provides the TimescaleWriter for persisting trade data to TimescaleDB.

## Usage

```python
from crypto_quant.data_pipeline.storage.writer import TimescaleWriter
from crypto_quant.data_pipeline.collectors.binance_ws import BinanceWSCollector

# Initialize writer
async with TimescaleWriter() as writer:
    # Use with collector
    collector = BinanceWSCollector()
    await collector.start()
    
    async for trade in collector.produce():
        await writer.write_trade(trade)
```

## Testing

### Unit Tests
```bash
pytest crypto_quant/data_pipeline/storage/tests/
```

### Integration Tests with Docker
```bash
# Start test database
docker-compose -f docker-compose.test.yml up -d

# Run integration tests
pytest crypto_quant/data_pipeline/storage/tests/ --integration

# Stop test database
docker-compose -f docker-compose.test.yml down
```

## Configuration

Set environment variables:
- `TIMESCALE_URL`: PostgreSQL connection URL
- `TEST_DATABASE_URL`: Test database URL (for integration tests)