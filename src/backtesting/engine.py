"""
回测引擎核心 - 事件驱动的回测框架

该模块实现了高性能的事件驱动回测引擎，支持tick级别的精确回测。
核心特性：
- 事件驱动架构，避免前视偏差
- 支持多数据源和多策略
- 精确的交易成本和滑点模拟
- 实时性能监控和风险控制
- 内存高效的数据处理
"""

import logging
from datetime import datetime, timedelta
from decimal import Decimal, getcontext
from typing import Dict, List, Optional, Any, Callable, Iterator
import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import json
from concurrent.futures import ThreadPoolExecutor
import threading
from dataclasses import dataclass, field
from enum import Enum

from .events import (
    Event, EventType, EventQueue, EventDispatcher, EventHandler,
    MarketDataEvent, SignalEvent, OrderEvent, FillEvent, RiskEvent,
    PortfolioEvent, HeartbeatEvent, OrderType, OrderSide, OrderStatus,
    SignalType
)
from .costs import TradingCostModel
from .metrics import PerformanceMetrics
from ..data_pipeline.models import OHLCV, MarketTick, OrderBookSnapshot
from ..data_pipeline.database import DatabaseManager
from config.settings import settings

# 设置Decimal精度
getcontext().prec = 28

logger = logging.getLogger(__name__)


class BacktestMode(Enum):
    """回测模式"""
    TICK = "tick"           # Tick级别
    MINUTE = "minute"       # 分钟级别
    HOUR = "hour"          # 小时级别
    DAILY = "daily"        # 日级别


@dataclass
class BacktestConfig:
    """回测配置"""
    # 时间范围
    start_date: datetime
    end_date: datetime
    
    # 数据配置
    symbols: List[str] = field(default_factory=lambda: ["BTCUSDT"])
    mode: BacktestMode = BacktestMode.MINUTE
    
    # 资金配置
    initial_capital: Decimal = Decimal('100000')
    base_currency: str = "USDT"
    
    # 交易配置
    commission_rate: Decimal = Decimal('0.0004')  # 0.04%
    slippage_bps: int = 2  # 2个基点
    
    # 风险配置
    max_leverage: int = 10
    max_position_size: Decimal = Decimal('0.3')  # 30%
    
    # 数据源配置
    data_source: str = "database"
    cache_data: bool = True
    
    # 输出配置
    output_dir: Path = Path("./backtest_results")
    save_trades: bool = True
    save_positions: bool = True
    
    def __post_init__(self):
        """验证配置参数"""
        if self.end_date <= self.start_date:
            raise ValueError("end_date must be after start_date")
        if self.initial_capital <= 0:
            raise ValueError("initial_capital must be positive")
        if not self.symbols:
            raise ValueError("symbols cannot be empty")
        
        # 创建输出目录
        self.output_dir.mkdir(parents=True, exist_ok=True)


class Position:
    """持仓管理"""
    
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.quantity = Decimal('0')
        self.avg_price = Decimal('0')
        self.unrealized_pnl = Decimal('0')
        self.realized_pnl = Decimal('0')
        self.last_price = Decimal('0')
        self.created_at = datetime.now()
        self.updated_at = datetime.now()
    
    def update_price(self, price: Decimal):
        """更新价格"""
        self.last_price = price
        if self.quantity != 0:
            self.unrealized_pnl = (price - self.avg_price) * self.quantity
        self.updated_at = datetime.now()
    
    def add_trade(self, side: OrderSide, quantity: Decimal, price: Decimal):
        """添加交易"""
        old_quantity = self.quantity
        
        if side == OrderSide.BUY:
            new_quantity = old_quantity + quantity
        else:
            new_quantity = old_quantity - quantity
        
        # 计算已实现盈亏
        if old_quantity * new_quantity < 0:  # 反向交易
            realized_quantity = min(abs(old_quantity), quantity)
            self.realized_pnl += (price - self.avg_price) * realized_quantity * (1 if old_quantity > 0 else -1)
        
        # 更新持仓
        if new_quantity == 0:
            self.avg_price = Decimal('0')
            self.unrealized_pnl = Decimal('0')
        elif old_quantity * new_quantity > 0:  # 同向交易
            total_value = old_quantity * self.avg_price + quantity * price * (1 if side == OrderSide.BUY else -1)
            self.avg_price = total_value / new_quantity
        else:  # 新仓位
            self.avg_price = price
        
        self.quantity = new_quantity
        self.last_price = price
        self.updated_at = datetime.now()
    
    @property
    def market_value(self) -> Decimal:
        """市值"""
        return self.quantity * self.last_price
    
    @property
    def is_long(self) -> bool:
        """是否多头"""
        return self.quantity > 0
    
    @property
    def is_short(self) -> bool:
        """是否空头"""
        return self.quantity < 0
    
    @property
    def is_flat(self) -> bool:
        """是否平仓"""
        return self.quantity == 0


class Portfolio:
    """投资组合管理"""
    
    def __init__(self, initial_capital: Decimal, base_currency: str = "USDT"):
        self.initial_capital = initial_capital
        self.base_currency = base_currency
        self.cash = initial_capital
        self.positions: Dict[str, Position] = {}
        self.trades: List[Dict] = []
        self.equity_curve: List[Dict] = []
        self.created_at = datetime.now()
        
        # 风险指标
        self.max_drawdown = Decimal('0')
        self.peak_equity = initial_capital
        
        # 性能指标
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_commission = Decimal('0')
        self.total_slippage = Decimal('0')
    
    def get_position(self, symbol: str) -> Position:
        """获取持仓"""
        if symbol not in self.positions:
            self.positions[symbol] = Position(symbol)
        return self.positions[symbol]
    
    def update_prices(self, prices: Dict[str, Decimal]):
        """更新价格"""
        for symbol, price in prices.items():
            if symbol in self.positions:
                self.positions[symbol].update_price(price)
    
    def execute_trade(self, symbol: str, side: OrderSide, quantity: Decimal, 
                     price: Decimal, commission: Decimal = Decimal('0'),
                     slippage: Decimal = Decimal('0'), timestamp: datetime = None):
        """执行交易"""
        if timestamp is None:
            timestamp = datetime.now()
        
        position = self.get_position(symbol)
        
        # 计算交易成本
        gross_amount = quantity * price
        if side == OrderSide.BUY:
            net_amount = gross_amount + commission + slippage
            self.cash -= net_amount
        else:
            net_amount = gross_amount - commission - slippage
            self.cash += net_amount
        
        # 更新持仓
        position.add_trade(side, quantity, price)
        
        # 记录交易
        trade = {
            'timestamp': timestamp,
            'symbol': symbol,
            'side': side.value,
            'quantity': float(quantity),
            'price': float(price),
            'gross_amount': float(gross_amount),
            'commission': float(commission),
            'slippage': float(slippage),
            'net_amount': float(net_amount),
            'cash_after': float(self.cash),
            'position_after': float(position.quantity)
        }
        self.trades.append(trade)
        
        # 更新统计
        self.total_trades += 1
        self.total_commission += commission
        self.total_slippage += slippage
        
        # 更新盈亏统计
        if position.quantity == 0 and position.realized_pnl != 0:
            if position.realized_pnl > 0:
                self.winning_trades += 1
            else:
                self.losing_trades += 1
        
        logger.debug(f"Trade executed: {trade}")
    
    def calculate_equity(self) -> Decimal:
        """计算净值"""
        total_unrealized = sum(pos.unrealized_pnl for pos in self.positions.values())
        equity = self.cash + total_unrealized
        
        # 更新最大回撤
        if equity > self.peak_equity:
            self.peak_equity = equity
        else:
            current_drawdown = (self.peak_equity - equity) / self.peak_equity
            if current_drawdown > self.max_drawdown:
                self.max_drawdown = current_drawdown
        
        return equity
    
    def get_portfolio_summary(self) -> Dict:
        """获取投资组合摘要"""
        equity = self.calculate_equity()
        total_pnl = sum(pos.realized_pnl + pos.unrealized_pnl for pos in self.positions.values())
        
        return {
            'equity': float(equity),
            'cash': float(self.cash),
            'total_pnl': float(total_pnl),
            'total_return': float(total_pnl / self.initial_capital),
            'max_drawdown': float(self.max_drawdown),
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': self.winning_trades / max(self.total_trades, 1),
            'total_commission': float(self.total_commission),
            'total_slippage': float(self.total_slippage),
            'positions': {
                symbol: {
                    'quantity': float(pos.quantity),
                    'avg_price': float(pos.avg_price),
                    'last_price': float(pos.last_price),
                    'unrealized_pnl': float(pos.unrealized_pnl),
                    'realized_pnl': float(pos.realized_pnl),
                    'market_value': float(pos.market_value)
                }
                for symbol, pos in self.positions.items()
                if not pos.is_flat
            }
        }


class DataHandler:
    """数据处理器"""
    
    def __init__(self, config: BacktestConfig):
        self.config = config
        self.db_manager = DatabaseManager()
        self.data_cache: Dict[str, pd.DataFrame] = {}
        self.current_data: Dict[str, Any] = {}
        
    def load_data(self) -> Iterator[MarketDataEvent]:
        """加载数据并生成市场数据事件"""
        logger.info(f"Loading data from {self.config.start_date} to {self.config.end_date}")
        
        for symbol in self.config.symbols:
            # 从数据库加载数据
            if self.config.mode == BacktestMode.TICK:
                data = self._load_tick_data(symbol)
            elif self.config.mode == BacktestMode.MINUTE:
                data = self._load_ohlcv_data(symbol, '1m')
            elif self.config.mode == BacktestMode.HOUR:
                data = self._load_ohlcv_data(symbol, '1h')
            else:  # DAILY
                data = self._load_ohlcv_data(symbol, '1d')
            
            if self.config.cache_data:
                self.data_cache[symbol] = data
            
            # 生成市场数据事件
            for _, row in data.iterrows():
                if self.config.mode == BacktestMode.TICK:
                    yield MarketDataEvent(
                        timestamp=row.name,
                        symbol=symbol,
                        open=Decimal(str(row['price'])),
                        high=Decimal(str(row['price'])),
                        low=Decimal(str(row['price'])),
                        close=Decimal(str(row['price'])),
                        volume=Decimal(str(row['volume']))
                    )
                else:
                    yield MarketDataEvent(
                        timestamp=row.name,
                        symbol=symbol,
                        open=Decimal(str(row['open'])),
                        high=Decimal(str(row['high'])),
                        low=Decimal(str(row['low'])),
                        close=Decimal(str(row['close'])),
                        volume=Decimal(str(row['volume']))
                    )
    
    def _load_tick_data(self, symbol: str) -> pd.DataFrame:
        """加载tick数据"""
        # 这里实现从数据库加载tick数据的逻辑
        # 现在使用模拟数据作为示例
        dates = pd.date_range(
            start=self.config.start_date,
            end=self.config.end_date,
            freq='1min'
        )
        
        # 模拟价格数据
        np.random.seed(42)
        prices = 50000 + np.random.randn(len(dates)) * 100
        volumes = np.random.uniform(0.1, 10, len(dates))
        
        return pd.DataFrame({
            'price': prices,
            'volume': volumes
        }, index=dates)
    
    def _load_ohlcv_data(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """加载OHLCV数据"""
        # 这里实现从数据库加载OHLCV数据的逻辑
        # 现在使用模拟数据作为示例
        freq_map = {'1m': '1min', '1h': '1H', '1d': '1D'}
        freq = freq_map.get(timeframe, '1min')
        
        dates = pd.date_range(
            start=self.config.start_date,
            end=self.config.end_date,
            freq=freq
        )
        
        # 模拟OHLCV数据
        np.random.seed(42)
        base_price = 50000
        data = []
        
        for i, date in enumerate(dates):
            if i == 0:
                open_price = base_price
            else:
                open_price = data[-1]['close']
            
            high_price = open_price * (1 + np.random.uniform(0, 0.02))
            low_price = open_price * (1 - np.random.uniform(0, 0.02))
            close_price = open_price + np.random.uniform(-0.01, 0.01) * open_price
            volume = np.random.uniform(100, 1000)
            
            data.append({
                'open': open_price,
                'high': high_price,
                'low': low_price,
                'close': close_price,
                'volume': volume
            })
        
        return pd.DataFrame(data, index=dates)


class ExecutionHandler(EventHandler):
    """执行处理器 - 模拟订单执行"""
    
    def __init__(self, portfolio: Portfolio, cost_model: TradingCostModel):
        self.portfolio = portfolio
        self.cost_model = cost_model
        self.pending_orders: Dict[str, OrderEvent] = {}
        self.current_prices: Dict[str, Decimal] = {}
        
    def handle_event(self, event: Event) -> Optional[List[Event]]:
        """处理事件"""
        if isinstance(event, MarketDataEvent):
            return self._handle_market_data(event)
        elif isinstance(event, OrderEvent):
            return self._handle_order(event)
        return None
    
    def _handle_market_data(self, event: MarketDataEvent) -> List[Event]:
        """处理市场数据"""
        self.current_prices[event.symbol] = event.close
        
        # 检查待执行订单
        fills = []
        executed_orders = []
        
        for order_id, order in self.pending_orders.items():
            if order.symbol == event.symbol:
                fill = self._execute_order(order, event)
                if fill:
                    fills.append(fill)
                    executed_orders.append(order_id)
        
        # 移除已执行订单
        for order_id in executed_orders:
            del self.pending_orders[order_id]
        
        return fills
    
    def _handle_order(self, event: OrderEvent) -> List[Event]:
        """处理订单"""
        if event.order_type == OrderType.MARKET:
            # 市价单立即执行
            if event.symbol in self.current_prices:
                fill = self._execute_market_order(event)
                return [fill] if fill else []
            else:
                logger.warning(f"No current price for {event.symbol}, queuing order")
                self.pending_orders[event.order_id] = event
                return []
        else:
            # 限价单和止损单加入待执行队列
            self.pending_orders[event.order_id] = event
            return []
    
    def _execute_market_order(self, order: OrderEvent) -> Optional[FillEvent]:
        """执行市价单"""
        current_price = self.current_prices.get(order.symbol)
        if not current_price:
            return None
        
        # 计算成本
        costs = self.cost_model.calculate_costs(
            symbol=order.symbol,
            side=order.side,
            quantity=order.quantity,
            price=current_price,
            order_type=order.order_type
        )
        
        # 执行交易
        self.portfolio.execute_trade(
            symbol=order.symbol,
            side=order.side,
            quantity=order.quantity,
            price=current_price,
            commission=costs.commission,
            slippage=costs.slippage,
            timestamp=order.timestamp
        )
        
        return FillEvent(
            timestamp=order.timestamp,
            symbol=order.symbol,
            order_id=order.order_id,
            fill_id=f"fill_{order.order_id}",
            side=order.side,
            quantity=order.quantity,
            price=current_price,
            commission=costs.commission,
            slippage=costs.slippage
        )
    
    def _execute_order(self, order: OrderEvent, market_data: MarketDataEvent) -> Optional[FillEvent]:
        """执行限价单或止损单"""
        if order.order_type == OrderType.LIMIT:
            # 限价单执行逻辑
            if order.side == OrderSide.BUY and market_data.low <= order.price:
                execution_price = min(order.price, market_data.open)
            elif order.side == OrderSide.SELL and market_data.high >= order.price:
                execution_price = max(order.price, market_data.open)
            else:
                return None
        elif order.order_type == OrderType.STOP:
            # 止损单执行逻辑
            if order.side == OrderSide.BUY and market_data.high >= order.stop_price:
                execution_price = market_data.close
            elif order.side == OrderSide.SELL and market_data.low <= order.stop_price:
                execution_price = market_data.close
            else:
                return None
        else:
            return None
        
        # 计算成本
        costs = self.cost_model.calculate_costs(
            symbol=order.symbol,
            side=order.side,
            quantity=order.quantity,
            price=execution_price,
            order_type=order.order_type
        )
        
        # 执行交易
        self.portfolio.execute_trade(
            symbol=order.symbol,
            side=order.side,
            quantity=order.quantity,
            price=execution_price,
            commission=costs.commission,
            slippage=costs.slippage,
            timestamp=market_data.timestamp
        )
        
        return FillEvent(
            timestamp=market_data.timestamp,
            symbol=order.symbol,
            order_id=order.order_id,
            fill_id=f"fill_{order.order_id}",
            side=order.side,
            quantity=order.quantity,
            price=execution_price,
            commission=costs.commission,
            slippage=costs.slippage
        )


class RiskManager(EventHandler):
    """风险管理器"""
    
    def __init__(self, portfolio: Portfolio, config: BacktestConfig):
        self.portfolio = portfolio
        self.config = config
        self.risk_limits = {
            'max_position_size': config.max_position_size,
            'max_leverage': config.max_leverage,
            'max_drawdown': Decimal('0.2'),  # 20%
            'max_daily_loss': Decimal('0.05')  # 5%
        }
        self.daily_start_equity = portfolio.calculate_equity()
        self.last_date = None
        
    def handle_event(self, event: Event) -> Optional[List[Event]]:
        """处理事件"""
        if isinstance(event, OrderEvent):
            return self._check_order_risk(event)
        elif isinstance(event, MarketDataEvent):
            return self._check_portfolio_risk(event)
        return None
    
    def _check_order_risk(self, order: OrderEvent) -> List[Event]:
        """检查订单风险"""
        risks = []
        
        # 检查仓位限制
        position = self.portfolio.get_position(order.symbol)
        current_position = position.quantity
        
        if order.side == OrderSide.BUY:
            new_position = current_position + order.quantity
        else:
            new_position = current_position - order.quantity
        
        position_value = abs(new_position) * self.portfolio.positions[order.symbol].last_price
        equity = self.portfolio.calculate_equity()
        position_ratio = position_value / equity
        
        if position_ratio > self.risk_limits['max_position_size']:
            risks.append(RiskEvent(
                timestamp=order.timestamp,
                symbol=order.symbol,
                risk_type="position_limit",
                severity="critical",
                message=f"Position size ({position_ratio:.2%}) exceeds limit ({self.risk_limits['max_position_size']:.2%})",
                current_value=position_ratio,
                limit_value=self.risk_limits['max_position_size'],
                suggested_action="reduce_order_size"
            ))
        
        return risks
    
    def _check_portfolio_risk(self, event: MarketDataEvent) -> List[Event]:
        """检查投资组合风险"""
        risks = []
        
        # 更新日开始净值
        current_date = event.timestamp.date()
        if self.last_date is None or current_date != self.last_date:
            self.daily_start_equity = self.portfolio.calculate_equity()
            self.last_date = current_date
        
        # 检查最大回撤
        if self.portfolio.max_drawdown > self.risk_limits['max_drawdown']:
            risks.append(RiskEvent(
                timestamp=event.timestamp,
                symbol=event.symbol,
                risk_type="max_drawdown",
                severity="emergency",
                message=f"Max drawdown ({self.portfolio.max_drawdown:.2%}) exceeds limit ({self.risk_limits['max_drawdown']:.2%})",
                current_value=self.portfolio.max_drawdown,
                limit_value=self.risk_limits['max_drawdown'],
                suggested_action="close_all_positions"
            ))
        
        # 检查日内损失
        current_equity = self.portfolio.calculate_equity()
        daily_return = (current_equity - self.daily_start_equity) / self.daily_start_equity
        
        if daily_return < -self.risk_limits['max_daily_loss']:
            risks.append(RiskEvent(
                timestamp=event.timestamp,
                symbol=event.symbol,
                risk_type="daily_loss",
                severity="critical",
                message=f"Daily loss ({daily_return:.2%}) exceeds limit ({self.risk_limits['max_daily_loss']:.2%})",
                current_value=abs(daily_return),
                limit_value=self.risk_limits['max_daily_loss'],
                suggested_action="halt_trading"
            ))
        
        return risks


class PerformanceTracker(EventHandler):
    """性能跟踪器"""
    
    def __init__(self, portfolio: Portfolio, config: BacktestConfig):
        self.portfolio = portfolio
        self.config = config
        self.metrics = PerformanceMetrics()
        self.equity_history = []
        self.last_update = None
        
    def handle_event(self, event: Event) -> Optional[List[Event]]:
        """处理事件"""
        if isinstance(event, MarketDataEvent):
            return self._update_performance(event)
        elif isinstance(event, FillEvent):
            return self._record_trade(event)
        return None
    
    def _update_performance(self, event: MarketDataEvent) -> List[Event]:
        """更新性能指标"""
        # 更新投资组合价格
        self.portfolio.update_prices({event.symbol: event.close})
        
        # 记录净值曲线
        equity = self.portfolio.calculate_equity()
        self.equity_history.append({
            'timestamp': event.timestamp,
            'equity': float(equity),
            'cash': float(self.portfolio.cash),
            'unrealized_pnl': float(sum(pos.unrealized_pnl for pos in self.portfolio.positions.values()))
        })
        
        # 每小时生成投资组合事件
        if (self.last_update is None or 
            event.timestamp - self.last_update >= timedelta(hours=1)):
            
            self.last_update = event.timestamp
            
            return [PortfolioEvent(
                timestamp=event.timestamp,
                symbol=event.symbol,
                total_value=equity,
                cash=self.portfolio.cash,
                positions={
                    symbol: pos.quantity
                    for symbol, pos in self.portfolio.positions.items()
                    if not pos.is_flat
                },
                unrealized_pnl=sum(pos.unrealized_pnl for pos in self.portfolio.positions.values()),
                realized_pnl=sum(pos.realized_pnl for pos in self.portfolio.positions.values())
            )]
        
        return []
    
    def _record_trade(self, event: FillEvent) -> List[Event]:
        """记录交易"""
        # 这里可以添加交易记录逻辑
        return []
    
    def get_performance_report(self) -> Dict:
        """获取性能报告"""
        if not self.equity_history:
            return {}
        
        equity_series = pd.Series(
            [item['equity'] for item in self.equity_history],
            index=[item['timestamp'] for item in self.equity_history]
        )
        
        return self.metrics.calculate_metrics(
            equity_series,
            initial_capital=float(self.config.initial_capital),
            trades=self.portfolio.trades
        )


class BacktestEngine:
    """回测引擎主类"""
    
    def __init__(self, config: BacktestConfig):
        self.config = config
        self.event_queue = EventQueue()
        self.event_dispatcher = EventDispatcher()
        self.portfolio = Portfolio(config.initial_capital)
        self.data_handler = DataHandler(config)
        
        # 初始化处理器
        self.cost_model = TradingCostModel(
            commission_rate=config.commission_rate,
            slippage_bps=config.slippage_bps
        )
        
        self.execution_handler = ExecutionHandler(self.portfolio, self.cost_model)
        self.risk_manager = RiskManager(self.portfolio, config)
        self.performance_tracker = PerformanceTracker(self.portfolio, config)
        
        # 注册事件处理器
        self._register_handlers()
        
        # 状态管理
        self.running = False
        self.start_time = None
        self.end_time = None
        
        logger.info(f"Backtest engine initialized: {config.start_date} to {config.end_date}")
    
    def _register_handlers(self):
        """注册事件处理器"""
        self.event_dispatcher.register_handler(EventType.MARKET_DATA, self.execution_handler)
        self.event_dispatcher.register_handler(EventType.ORDER, self.execution_handler)
        self.event_dispatcher.register_handler(EventType.ORDER, self.risk_manager)
        self.event_dispatcher.register_handler(EventType.MARKET_DATA, self.risk_manager)
        self.event_dispatcher.register_handler(EventType.MARKET_DATA, self.performance_tracker)
        self.event_dispatcher.register_handler(EventType.FILL, self.performance_tracker)
    
    def add_strategy(self, strategy: EventHandler):
        """添加交易策略"""
        self.event_dispatcher.register_handler(EventType.MARKET_DATA, strategy)
        self.event_dispatcher.register_handler(EventType.FILL, strategy)
        logger.info(f"Strategy added: {strategy.__class__.__name__}")
    
    def run(self) -> Dict:
        """运行回测"""
        logger.info("Starting backtest...")
        self.start_time = datetime.now()
        self.running = True
        
        try:
            # 加载数据并处理事件
            total_events = 0
            
            for market_event in self.data_handler.load_data():
                if not self.running:
                    break
                
                # 添加市场数据事件
                self.event_queue.put(market_event)
                total_events += 1
                
                # 处理事件队列
                self._process_events()
                
                # 定期报告进度
                if total_events % 10000 == 0:
                    logger.info(f"Processed {total_events} events")
            
            # 处理剩余事件
            self._process_events()
            
            self.end_time = datetime.now()
            logger.info(f"Backtest completed in {self.end_time - self.start_time}")
            
            # 生成最终报告
            return self._generate_report()
            
        except Exception as e:
            logger.error(f"Backtest failed: {e}")
            raise
        finally:
            self.running = False
    
    def _process_events(self):
        """处理事件队列"""
        while not self.event_queue.empty():
            event = self.event_queue.get()
            if event is None:
                break
            
            # 分发事件
            output_events = self.event_dispatcher.dispatch(event)
            
            # 将输出事件加入队列
            for output_event in output_events:
                self.event_queue.put(output_event)
    
    def _generate_report(self) -> Dict:
        """生成回测报告"""
        portfolio_summary = self.portfolio.get_portfolio_summary()
        performance_report = self.performance_tracker.get_performance_report()
        
        report = {
            'config': {
                'start_date': self.config.start_date.isoformat(),
                'end_date': self.config.end_date.isoformat(),
                'initial_capital': float(self.config.initial_capital),
                'symbols': self.config.symbols,
                'mode': self.config.mode.value
            },
            'portfolio': portfolio_summary,
            'performance': performance_report,
            'execution': {
                'runtime': str(self.end_time - self.start_time),
                'total_events': len(self.performance_tracker.equity_history)
            }
        }
        
        # 保存报告
        if self.config.save_trades:
            self._save_results(report)
        
        return report
    
    def _save_results(self, report: Dict):
        """保存回测结果"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存主报告
        report_file = self.config.output_dir / f"backtest_report_{timestamp}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # 保存交易记录
        if self.portfolio.trades:
            trades_file = self.config.output_dir / f"trades_{timestamp}.json"
            with open(trades_file, 'w') as f:
                json.dump(self.portfolio.trades, f, indent=2, default=str)
        
        # 保存净值曲线
        if self.performance_tracker.equity_history:
            equity_file = self.config.output_dir / f"equity_curve_{timestamp}.json"
            with open(equity_file, 'w') as f:
                json.dump(self.performance_tracker.equity_history, f, indent=2, default=str)
        
        logger.info(f"Results saved to {self.config.output_dir}")
    
    def stop(self):
        """停止回测"""
        self.running = False
        logger.info("Backtest stopped")