# Crypto Quant - Advanced BTC Futures Trading System

A high-performance quantitative trading system for Bitcoin perpetual futures, featuring real-time data collection, advanced feature engineering, and deep learning models.

## Features

- **Real-time Data Pipeline**: Sub-millisecond market data collection from multiple exchanges
- **Advanced Features**: 330+ features including microstructure, on-chain metrics, and technical indicators
- **Deep Learning Models**: Transformer + LSTM hybrid architecture for price prediction
- **Risk Management**: Multi-level position sizing and stop-loss strategies
- **Low Latency Execution**: Target latency < 50ms with smart order routing
- **Comprehensive Backtesting**: Walk-forward analysis with transaction cost modeling

## Quick Start

### Prerequisites

- Python 3.11+
- Docker & Docker Compose
- 16GB+ RAM recommended
- SSD storage for database

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/crypto_quant.git
cd crypto_quant
```

2. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Start infrastructure services:
```bash
docker-compose up -d
```

5. Copy and configure environment variables:
```bash
cp .env.example .env
# Edit .env with your API keys and settings
```

6. Initialize database:
```bash
python main.py init-db
```

### Running Data Collection

Start real-time data collection:
```bash
python main.py collect --debug
```

This will:
- Connect to exchange WebSocket streams
- Collect tick data, orderbook snapshots, and funding rates
- Store data in TimescaleDB with automatic compression
- Cache real-time data in Redis

### System Architecture

See [CRYPTO_QUANT_ARCHITECTURE.md](CRYPTO_QUANT_ARCHITECTURE.md) for detailed system design.

## Project Structure

```
crypto_quant/
├── config/                 # Configuration management
│   └── settings.py        # Centralized settings
├── src/
│   ├── data_pipeline/     # Data collection and storage
│   │   ├── collectors/    # Exchange-specific collectors
│   │   ├── models.py      # Database models
│   │   ├── database.py    # Database connections
│   │   ├── cache.py       # Redis cache management
│   │   └── writer.py      # Batch data writer
│   ├── feature_engineering/  # Feature extraction
│   ├── models/              # ML/DL models
│   ├── strategy/            # Trading strategies
│   ├── execution/           # Order execution
│   ├── backtesting/         # Backtesting engine
│   └── monitoring/          # System monitoring
├── tests/                   # Test suite
├── docker-compose.yml       # Infrastructure services
├── requirements.txt         # Python dependencies
└── main.py                 # Application entry point
```

## Configuration

Key configuration options in `.env`:

```bash
# Exchange API (start with testnet)
BINANCE_API_KEY=your_testnet_key
BINANCE_API_SECRET=your_testnet_secret
BINANCE_TESTNET=True

# Database
TIMESCALE_PASSWORD=secure_password

# Trading Parameters
MAX_POSITION_SIZE=0.3      # 30% of account
DEFAULT_LEVERAGE=3
STOP_LOSS_PCT=0.02         # 2%
MAX_DAILY_LOSS=0.03        # 3%

# Data Collection
ORDERBOOK_LEVELS=20
TICK_BUFFER_SIZE=10000
```

## Monitoring

Access monitoring dashboards:
- Grafana: http://localhost:3000 (admin/admin)
- Prometheus: http://localhost:9090

## Development

### Running Tests
```bash
pytest tests/ -v --cov=src
```

### Code Quality
```bash
# Format code
black src/ tests/

# Lint
flake8 src/ tests/

# Type checking
mypy src/
```

## Development Status

### Completed Modules ✅
- **Deep Learning Models**: Transformer + LSTM + CNN hybrid architecture implemented
- **Feature Engineering**: 330+ feature calculation framework (microstructure, technical, on-chain)
- **Model Management**: Complete training, prediction, and ensemble systems
- **Data Pipeline**: Database models, collectors framework, caching system
- **Strategy Framework**: Base classes, signal generation, risk management

### In Progress 🚧
- **AI Integration**: Connecting ML models to strategy execution (current focus)
- **End-to-End Pipeline**: Data → Features → Model → Strategy → Backtest
- **Real Data Collection**: Activating live market data feeds

### Implementation Status by Module:
- `src/models/` - **90%** (Complete, needs integration)
- `src/feature_engineering/` - **85%** (Complete, needs testing)
- `src/data_pipeline/` - **70%** (Framework done, needs activation)
- `src/strategy/` - **60%** (Needs AI model integration)
- `src/backtesting/` - **40%** (Needs AI-driven version)
- `src/execution/` - **20%** (Basic framework only)

### Next Steps:
1. Integrate AI models into strategy decision making
2. Implement AI-driven backtesting engine
3. Activate real-time data collection
4. End-to-end system validation

## Performance Targets

- **Annual Return**: 80-120%
- **Sharpe Ratio**: 2.5-3.5
- **Maximum Drawdown**: < 15%
- **Win Rate**: > 55%
- **System Latency**: < 50ms

## Safety Features

- Automatic position limits
- Maximum daily loss limits
- Kill switch for emergencies
- Data validation and anomaly detection
- Redundant data sources

## Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## Disclaimer

This software is for educational purposes only. Cryptocurrency trading carries substantial risk of loss. Always test thoroughly with small amounts before using real funds.

## License

MIT License - see LICENSE file for details