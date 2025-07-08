"""
Global configuration settings for the crypto quant system
"""
from pydantic_settings import BaseSettings
from pydantic import Field, validator
from typing import List, Dict, Optional
from pathlib import Path
import os


class DatabaseSettings(BaseSettings):
    """Database configuration"""
    timescale_host: str = Field(default="localhost", env="TIMESCALE_HOST")
    timescale_port: int = Field(default=5432, env="TIMESCALE_PORT")
    timescale_user: str = Field(default="crypto_quant", env="TIMESCALE_USER")
    timescale_password: str = Field(default="", env="TIMESCALE_PASSWORD")
    timescale_db: str = Field(default="crypto_market_data", env="TIMESCALE_DB")
    
    redis_host: str = Field(default="localhost", env="REDIS_HOST")
    redis_port: int = Field(default=6379, env="REDIS_PORT")
    redis_password: Optional[str] = Field(default=None, env="REDIS_PASSWORD")
    
    @property
    def timescale_url(self) -> str:
        return f"postgresql://{self.timescale_user}:{self.timescale_password}@{self.timescale_host}:{self.timescale_port}/{self.timescale_db}"
    
    @property
    def redis_url(self) -> str:
        if self.redis_password:
            return f"redis://:{self.redis_password}@{self.redis_host}:{self.redis_port}"
        return f"redis://{self.redis_host}:{self.redis_port}"


class ExchangeSettings(BaseSettings):
    """Exchange API configuration"""
    # Binance
    binance_api_key: str = Field(default="", env="BINANCE_API_KEY")
    binance_api_secret: str = Field(default="", env="BINANCE_API_SECRET")
    binance_testnet: bool = Field(default=True, env="BINANCE_TESTNET")
    
    # OKX
    okx_api_key: str = Field(default="", env="OKX_API_KEY")
    okx_api_secret: str = Field(default="", env="OKX_API_SECRET")
    okx_passphrase: str = Field(default="", env="OKX_PASSPHRASE")
    okx_testnet: bool = Field(default=True, env="OKX_TESTNET")
    
    # Bybit
    bybit_api_key: str = Field(default="", env="BYBIT_API_KEY")
    bybit_api_secret: str = Field(default="", env="BYBIT_API_SECRET")
    bybit_testnet: bool = Field(default=True, env="BYBIT_TESTNET")


class DataSettings(BaseSettings):
    """Data collection and storage settings"""
    # Market data
    orderbook_levels: int = Field(default=20, env="ORDERBOOK_LEVELS")
    tick_buffer_size: int = Field(default=10000, env="TICK_BUFFER_SIZE")
    
    # Data retention (days)
    hot_data_retention: int = Field(default=7, env="HOT_DATA_RETENTION_DAYS")
    warm_data_retention: int = Field(default=30, env="WARM_DATA_RETENTION_DAYS")
    
    # Collection intervals (seconds)
    orderbook_snapshot_interval: float = Field(default=0.1, env="ORDERBOOK_SNAPSHOT_INTERVAL")
    funding_rate_interval: int = Field(default=300, env="FUNDING_RATE_INTERVAL")
    
    # Data paths
    data_root: Path = Field(default=Path("./data"), env="DATA_ROOT")
    cache_dir: Path = Field(default=Path("./cache"), env="CACHE_DIR")
    
    @validator("data_root", "cache_dir", pre=True)
    def create_dirs(cls, v):
        path = Path(v)
        path.mkdir(parents=True, exist_ok=True)
        return path


class ModelSettings(BaseSettings):
    """Model configuration"""
    # Training
    batch_size: int = Field(default=256, env="MODEL_BATCH_SIZE")
    learning_rate: float = Field(default=0.001, env="MODEL_LEARNING_RATE")
    epochs: int = Field(default=100, env="MODEL_EPOCHS")
    early_stopping_patience: int = Field(default=10, env="MODEL_EARLY_STOPPING_PATIENCE")
    
    # Architecture
    transformer_heads: int = Field(default=8, env="TRANSFORMER_HEADS")
    transformer_layers: int = Field(default=4, env="TRANSFORMER_LAYERS")
    lstm_hidden_size: int = Field(default=256, env="LSTM_HIDDEN_SIZE")
    lstm_layers: int = Field(default=3, env="LSTM_LAYERS")
    
    # Paths
    model_dir: Path = Field(default=Path("./models"), env="MODEL_DIR")
    checkpoint_dir: Path = Field(default=Path("./checkpoints"), env="CHECKPOINT_DIR")
    
    @validator("model_dir", "checkpoint_dir", pre=True)
    def create_model_dirs(cls, v):
        path = Path(v)
        path.mkdir(parents=True, exist_ok=True)
        return path


class StrategySettings(BaseSettings):
    """Trading strategy settings"""
    # Position sizing
    max_position_size: float = Field(default=0.3, env="MAX_POSITION_SIZE")  # 30% of account
    default_leverage: int = Field(default=3, env="DEFAULT_LEVERAGE")
    max_leverage: int = Field(default=10, env="MAX_LEVERAGE")
    
    # Risk management
    stop_loss_pct: float = Field(default=0.02, env="STOP_LOSS_PCT")  # 2%
    take_profit_pct: float = Field(default=0.05, env="TAKE_PROFIT_PCT")  # 5%
    max_daily_loss: float = Field(default=0.03, env="MAX_DAILY_LOSS")  # 3%
    max_drawdown: float = Field(default=0.15, env="MAX_DRAWDOWN")  # 15%
    
    # Signal generation
    signal_threshold: float = Field(default=0.7, env="SIGNAL_THRESHOLD")
    min_confidence: float = Field(default=0.6, env="MIN_CONFIDENCE")
    
    # Execution
    use_market_orders: bool = Field(default=False, env="USE_MARKET_ORDERS")
    slippage_model: str = Field(default="adaptive", env="SLIPPAGE_MODEL")


class BacktestSettings(BaseSettings):
    """Backtesting configuration"""
    # Data
    lookback_days: int = Field(default=365, env="BACKTEST_LOOKBACK_DAYS")
    train_test_split: float = Field(default=0.8, env="TRAIN_TEST_SPLIT")
    walk_forward_days: int = Field(default=60, env="WALK_FORWARD_DAYS")
    test_window_days: int = Field(default=7, env="TEST_WINDOW_DAYS")
    
    # Execution simulation
    commission_rate: float = Field(default=0.0004, env="COMMISSION_RATE")  # 0.04%
    slippage_bps: int = Field(default=2, env="SLIPPAGE_BPS")  # 2 basis points
    
    # Validation
    n_splits: int = Field(default=5, env="BACKTEST_N_SPLITS")
    purge_days: int = Field(default=2, env="BACKTEST_PURGE_DAYS")


class MonitoringSettings(BaseSettings):
    """System monitoring configuration"""
    # Metrics
    metrics_port: int = Field(default=8000, env="METRICS_PORT")
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    
    # Alerts
    alert_email: Optional[str] = Field(default=None, env="ALERT_EMAIL")
    alert_webhook: Optional[str] = Field(default=None, env="ALERT_WEBHOOK")
    
    # Performance thresholds
    max_latency_ms: int = Field(default=50, env="MAX_LATENCY_MS")
    min_sharpe_ratio: float = Field(default=2.0, env="MIN_SHARPE_RATIO")


class Settings(BaseSettings):
    """Main settings aggregator"""
    # Sub-settings
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    exchange: ExchangeSettings = Field(default_factory=ExchangeSettings)
    data: DataSettings = Field(default_factory=DataSettings)
    model: ModelSettings = Field(default_factory=ModelSettings)
    strategy: StrategySettings = Field(default_factory=StrategySettings)
    backtest: BacktestSettings = Field(default_factory=BacktestSettings)
    monitoring: MonitoringSettings = Field(default_factory=MonitoringSettings)
    
    # General
    environment: str = Field(default="development", env="ENVIRONMENT")
    debug: bool = Field(default=False, env="DEBUG")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Global settings instance
settings = Settings()