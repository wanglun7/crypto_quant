"""
Base strategy classes and interfaces
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Union
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
import logging
from datetime import datetime
import asyncio
from decimal import Decimal

logger = logging.getLogger(__name__)


class StrategyState(Enum):
    """Strategy execution states"""
    INITIALIZED = "initialized"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPED = "stopped"
    ERROR = "error"


class SignalType(Enum):
    """Signal types for trading decisions"""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    STRONG_BUY = "strong_buy"
    STRONG_SELL = "strong_sell"


@dataclass
class StrategyConfig:
    """Configuration for strategy execution"""
    # Strategy identification
    name: str = "base_strategy"
    version: str = "1.0.0"
    
    # Execution parameters
    symbol: str = "BTC/USDT"
    timeframe: str = "1m"
    lookback_period: int = 100
    min_data_points: int = 200
    
    # Risk management
    max_position_size: float = 0.1  # 10% of capital
    max_drawdown: float = 0.15  # 15% max drawdown
    stop_loss_pct: float = 0.02  # 2% stop loss
    take_profit_pct: float = 0.04  # 4% take profit
    
    # Signal generation
    signal_threshold: float = 0.6
    signal_confirmation_bars: int = 1
    
    # Feature engineering
    use_technical_indicators: bool = True
    use_microstructure_features: bool = True
    use_ml_predictions: bool = True
    
    # Execution settings
    execution_delay: float = 0.1  # seconds
    max_slippage: float = 0.001  # 0.1%
    
    # Monitoring
    log_level: str = "INFO"
    save_signals: bool = True
    save_positions: bool = True
    
    # Custom parameters
    custom_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Signal:
    """Trading signal with metadata"""
    timestamp: datetime
    symbol: str
    signal_type: SignalType
    strength: float  # 0-1
    price: float
    confidence: float  # 0-1
    source: str  # Signal source (e.g., 'technical', 'ml', 'combined')
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        # Validate signal strength and confidence
        self.strength = max(0.0, min(1.0, self.strength))
        self.confidence = max(0.0, min(1.0, self.confidence))


@dataclass
class Position:
    """Position information"""
    symbol: str
    side: str  # 'long' or 'short'
    size: float
    entry_price: float
    current_price: float
    unrealized_pnl: float
    realized_pnl: float
    timestamp: datetime
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    
    @property
    def market_value(self) -> float:
        """Current market value of position"""
        return self.size * self.current_price
    
    @property
    def pnl_pct(self) -> float:
        """P&L percentage"""
        if self.side == 'long':
            return (self.current_price - self.entry_price) / self.entry_price
        else:
            return (self.entry_price - self.current_price) / self.entry_price


@dataclass
class StrategyMetrics:
    """Strategy performance metrics"""
    total_return: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0
    num_trades: int = 0
    
    # Additional metrics
    volatility: float = 0.0
    calmar_ratio: float = 0.0
    sortino_ratio: float = 0.0
    var_95: float = 0.0  # Value at Risk at 95%
    cvar_95: float = 0.0  # Conditional Value at Risk
    
    def update_metrics(self, returns: pd.Series):
        """Update metrics from returns series"""
        if len(returns) == 0:
            return
            
        # Basic metrics
        self.total_return = (1 + returns).prod() - 1
        self.volatility = returns.std() * np.sqrt(252 * 24 * 60)  # Annualized for crypto
        
        # Sharpe ratio
        if self.volatility > 0:
            self.sharpe_ratio = (returns.mean() * 252 * 24 * 60) / self.volatility
        
        # Max drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        self.max_drawdown = drawdown.min()
        
        # Win rate and trade metrics
        winning_trades = returns[returns > 0]
        losing_trades = returns[returns < 0]
        
        if len(returns) > 0:
            self.win_rate = len(winning_trades) / len(returns)
            self.num_trades = len(returns)
        
        if len(winning_trades) > 0:
            self.avg_win = winning_trades.mean()
        if len(losing_trades) > 0:
            self.avg_loss = abs(losing_trades.mean())
        
        # Profit factor
        if self.avg_loss > 0:
            self.profit_factor = (self.avg_win * len(winning_trades)) / (self.avg_loss * len(losing_trades))
        
        # Calmar ratio
        if abs(self.max_drawdown) > 0:
            self.calmar_ratio = (self.total_return / abs(self.max_drawdown))
        
        # Sortino ratio
        downside_returns = returns[returns < 0]
        if len(downside_returns) > 0:
            downside_deviation = downside_returns.std() * np.sqrt(252 * 24 * 60)
            if downside_deviation > 0:
                self.sortino_ratio = (returns.mean() * 252 * 24 * 60) / downside_deviation
        
        # VaR and CVaR
        if len(returns) > 0:
            self.var_95 = returns.quantile(0.05)
            self.cvar_95 = returns[returns <= self.var_95].mean()


class BaseStrategy(ABC):
    """Abstract base class for all trading strategies"""
    
    def __init__(self, config: StrategyConfig):
        self.config = config
        self.state = StrategyState.INITIALIZED
        self.logger = logging.getLogger(f"{__name__}.{config.name}")
        self.logger.setLevel(getattr(logging, config.log_level))
        
        # Strategy data
        self.data: pd.DataFrame = pd.DataFrame()
        self.features: pd.DataFrame = pd.DataFrame()
        self.signals: List[Signal] = []
        self.positions: Dict[str, Position] = {}
        self.metrics = StrategyMetrics()
        
        # Internal state
        self._last_signal_time: Optional[datetime] = None
        self._last_update_time: Optional[datetime] = None
        self._error_count: int = 0
        self._max_errors: int = 10
        
        # Hooks for extensibility
        self._on_init_callbacks: List[callable] = []
        self._on_signal_callbacks: List[callable] = []
        self._on_trade_callbacks: List[callable] = []
        self._on_error_callbacks: List[callable] = []
        
        logger.info(f"Strategy {config.name} initialized")
    
    @abstractmethod
    def generate_signal(self, data: pd.DataFrame) -> Optional[Signal]:
        """Generate trading signal from data"""
        pass
    
    @abstractmethod
    def calculate_position_size(self, signal: Signal, current_price: float, 
                              account_balance: float) -> float:
        """Calculate position size for given signal"""
        pass
    
    def initialize(self, initial_data: pd.DataFrame) -> bool:
        """Initialize strategy with historical data"""
        try:
            self.logger.info("Initializing strategy...")
            
            # Validate data
            if not self._validate_data(initial_data):
                self.logger.error("Data validation failed")
                return False
            
            # Set initial data
            self.data = initial_data.copy()
            
            # Initialize features
            self._initialize_features()
            
            # Run initialization callbacks
            for callback in self._on_init_callbacks:
                callback(self)
            
            self.state = StrategyState.INITIALIZED
            self.logger.info("Strategy initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Strategy initialization failed: {e}")
            self.state = StrategyState.ERROR
            return False
    
    def update(self, new_data: pd.DataFrame) -> Optional[Signal]:
        """Update strategy with new data and generate signal"""
        try:
            if self.state not in [StrategyState.INITIALIZED, StrategyState.RUNNING]:
                self.logger.warning(f"Strategy not ready for updates. State: {self.state}")
                return None
            
            # Update data
            self.data = pd.concat([self.data, new_data]).tail(
                self.config.lookback_period + 100
            )
            
            # Update features
            self._update_features()
            
            # Generate signal
            signal = self.generate_signal(self.data)
            
            if signal:
                self.signals.append(signal)
                self._last_signal_time = signal.timestamp
                
                # Run signal callbacks
                for callback in self._on_signal_callbacks:
                    callback(self, signal)
                
                self.logger.info(f"Signal generated: {signal.signal_type.value} "
                               f"(strength: {signal.strength:.2f}, "
                               f"confidence: {signal.confidence:.2f})")
            
            self._last_update_time = datetime.now()
            self.state = StrategyState.RUNNING
            
            return signal
            
        except Exception as e:
            self.logger.error(f"Strategy update failed: {e}")
            self._handle_error(e)
            return None
    
    def pause(self):
        """Pause strategy execution"""
        self.state = StrategyState.PAUSED
        self.logger.info("Strategy paused")
    
    def resume(self):
        """Resume strategy execution"""
        if self.state == StrategyState.PAUSED:
            self.state = StrategyState.RUNNING
            self.logger.info("Strategy resumed")
    
    def stop(self):
        """Stop strategy execution"""
        self.state = StrategyState.STOPPED
        self.logger.info("Strategy stopped")
    
    def get_current_positions(self) -> Dict[str, Position]:
        """Get current positions"""
        return self.positions.copy()
    
    def get_signals(self, limit: Optional[int] = None) -> List[Signal]:
        """Get recent signals"""
        signals = self.signals
        if limit:
            signals = signals[-limit:]
        return signals
    
    def get_metrics(self) -> StrategyMetrics:
        """Get strategy performance metrics"""
        return self.metrics
    
    def validate_signal(self, signal: Signal) -> bool:
        """Validate signal before processing"""
        if signal.strength < self.config.signal_threshold:
            self.logger.debug(f"Signal strength {signal.strength} below threshold "
                            f"{self.config.signal_threshold}")
            return False
        
        # Check signal timing
        if (self._last_signal_time and 
            signal.timestamp <= self._last_signal_time):
            self.logger.debug("Signal timestamp not newer than last signal")
            return False
        
        return True
    
    def add_callback(self, event_type: str, callback: callable):
        """Add callback for strategy events"""
        if event_type == 'init':
            self._on_init_callbacks.append(callback)
        elif event_type == 'signal':
            self._on_signal_callbacks.append(callback)
        elif event_type == 'trade':
            self._on_trade_callbacks.append(callback)
        elif event_type == 'error':
            self._on_error_callbacks.append(callback)
        else:
            raise ValueError(f"Unknown event type: {event_type}")
    
    def _validate_data(self, data: pd.DataFrame) -> bool:
        """Validate input data"""
        if data.empty:
            self.logger.error("Data is empty")
            return False
        
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = set(required_columns) - set(data.columns)
        if missing_columns:
            self.logger.error(f"Missing required columns: {missing_columns}")
            return False
        
        if len(data) < self.config.min_data_points:
            self.logger.error(f"Insufficient data points: {len(data)} < "
                            f"{self.config.min_data_points}")
            return False
        
        return True
    
    def _initialize_features(self):
        """Initialize features from data"""
        # This will be implemented by subclasses or feature engineering modules
        pass
    
    def _update_features(self):
        """Update features with new data"""
        # This will be implemented by subclasses or feature engineering modules
        pass
    
    def _handle_error(self, error: Exception):
        """Handle strategy errors"""
        self._error_count += 1
        
        # Run error callbacks
        for callback in self._on_error_callbacks:
            callback(self, error)
        
        if self._error_count >= self._max_errors:
            self.logger.error(f"Maximum errors reached ({self._max_errors}). "
                            f"Stopping strategy.")
            self.state = StrategyState.ERROR
        
        self.logger.error(f"Strategy error #{self._error_count}: {error}")
    
    def __str__(self) -> str:
        return f"{self.config.name} (State: {self.state.value})"
    
    def __repr__(self) -> str:
        return (f"BaseStrategy(name='{self.config.name}', "
                f"state='{self.state.value}', "
                f"signals={len(self.signals)})")


class StrategyManager:
    """Manage multiple strategies"""
    
    def __init__(self):
        self.strategies: Dict[str, BaseStrategy] = {}
        self.logger = logging.getLogger(__name__)
    
    def add_strategy(self, strategy: BaseStrategy):
        """Add strategy to manager"""
        self.strategies[strategy.config.name] = strategy
        self.logger.info(f"Added strategy: {strategy.config.name}")
    
    def remove_strategy(self, name: str):
        """Remove strategy from manager"""
        if name in self.strategies:
            del self.strategies[name]
            self.logger.info(f"Removed strategy: {name}")
    
    def get_strategy(self, name: str) -> Optional[BaseStrategy]:
        """Get strategy by name"""
        return self.strategies.get(name)
    
    def get_all_strategies(self) -> Dict[str, BaseStrategy]:
        """Get all strategies"""
        return self.strategies.copy()
    
    def update_all(self, data: pd.DataFrame) -> Dict[str, Optional[Signal]]:
        """Update all strategies with new data"""
        results = {}
        
        for name, strategy in self.strategies.items():
            try:
                signal = strategy.update(data)
                results[name] = signal
            except Exception as e:
                self.logger.error(f"Error updating strategy {name}: {e}")
                results[name] = None
        
        return results
    
    def get_combined_signals(self) -> List[Signal]:
        """Get all signals from all strategies"""
        all_signals = []
        
        for strategy in self.strategies.values():
            all_signals.extend(strategy.get_signals())
        
        # Sort by timestamp
        all_signals.sort(key=lambda x: x.timestamp)
        
        return all_signals
    
    def get_aggregate_metrics(self) -> Dict[str, StrategyMetrics]:
        """Get metrics for all strategies"""
        return {name: strategy.get_metrics() 
                for name, strategy in self.strategies.items()}
    
    def pause_all(self):
        """Pause all strategies"""
        for strategy in self.strategies.values():
            strategy.pause()
    
    def resume_all(self):
        """Resume all strategies"""
        for strategy in self.strategies.values():
            strategy.resume()
    
    def stop_all(self):
        """Stop all strategies"""
        for strategy in self.strategies.values():
            strategy.stop()