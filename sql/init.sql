-- Initialize crypto_quant database schema

-- Create TimescaleDB extension if not exists
CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;

-- Create klines table for OHLCV data
CREATE TABLE IF NOT EXISTS klines (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    timeframe VARCHAR(10) NOT NULL,
    open DECIMAL(20,8) NOT NULL,
    high DECIMAL(20,8) NOT NULL,
    low DECIMAL(20,8) NOT NULL,
    close DECIMAL(20,8) NOT NULL,
    volume DECIMAL(20,8) NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create hypertable for time-series optimization
SELECT create_hypertable('klines', 'timestamp', if_not_exists => TRUE);

-- Create indexes for better query performance
CREATE INDEX IF NOT EXISTS idx_klines_symbol_timeframe_timestamp 
ON klines (symbol, timeframe, timestamp);

CREATE INDEX IF NOT EXISTS idx_klines_timestamp 
ON klines (timestamp DESC);

-- Create a table for strategy performance tracking
CREATE TABLE IF NOT EXISTS strategy_performance (
    id SERIAL PRIMARY KEY,
    strategy_name VARCHAR(50) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    position_size DECIMAL(20,8),
    pnl DECIMAL(20,8),
    cumulative_pnl DECIMAL(20,8),
    drawdown DECIMAL(10,4),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create hypertable for strategy performance
SELECT create_hypertable('strategy_performance', 'timestamp', if_not_exists => TRUE);

-- Grant permissions
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO crypto_quant;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO crypto_quant;