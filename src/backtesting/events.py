"""
事件系统 - 回测引擎的核心事件框架

该模块实现了事件驱动回测系统的核心事件类型，确保无前视偏差的回测环境。
支持多种事件类型：市场数据、交易信号、订单执行、风险控制等。
"""
from abc import ABC, abstractmethod
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Dict, Any, Optional, List
import pandas as pd
from dataclasses import dataclass, field


class EventType(Enum):
    """事件类型枚举"""
    MARKET_DATA = "market_data"
    SIGNAL = "signal"
    ORDER = "order"
    FILL = "fill"
    RISK = "risk"
    PORTFOLIO = "portfolio"
    HEARTBEAT = "heartbeat"


class OrderType(Enum):
    """订单类型枚举"""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderSide(Enum):
    """订单方向枚举"""
    BUY = "buy"
    SELL = "sell"


class OrderStatus(Enum):
    """订单状态枚举"""
    SUBMITTED = "submitted"
    ACKNOWLEDGED = "acknowledged"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


class SignalType(Enum):
    """信号类型枚举"""
    LONG = "long"
    SHORT = "short"
    CLOSE = "close"
    NEUTRAL = "neutral"


@dataclass
class Event(ABC):
    """事件基类"""
    event_type: EventType
    timestamp: datetime
    symbol: str
    
    def __post_init__(self):
        """验证事件数据"""
        if not isinstance(self.timestamp, datetime):
            raise ValueError("timestamp must be a datetime object")
        if not isinstance(self.symbol, str):
            raise ValueError("symbol must be a string")


@dataclass
class MarketDataEvent(Event):
    """市场数据事件"""
    event_type: EventType = field(default=EventType.MARKET_DATA, init=False)
    
    # OHLCV数据
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal
    
    # 订单簿数据
    bid: Optional[Decimal] = None
    ask: Optional[Decimal] = None
    bid_size: Optional[Decimal] = None
    ask_size: Optional[Decimal] = None
    
    # 扩展市场数据
    funding_rate: Optional[Decimal] = None
    open_interest: Optional[Decimal] = None
    vwap: Optional[Decimal] = None
    
    # 数据质量标识
    data_quality: str = "good"
    
    def __post_init__(self):
        super().__post_init__()
        
        # 验证价格数据
        if self.high < self.low:
            raise ValueError("High price cannot be lower than low price")
        if self.close < 0 or self.open < 0:
            raise ValueError("Prices cannot be negative")
        if self.volume < 0:
            raise ValueError("Volume cannot be negative")
    
    @property
    def mid_price(self) -> Optional[Decimal]:
        """计算中间价"""
        if self.bid is not None and self.ask is not None:
            return (self.bid + self.ask) / 2
        return None
    
    @property
    def spread(self) -> Optional[Decimal]:
        """计算买卖价差"""
        if self.bid is not None and self.ask is not None:
            return self.ask - self.bid
        return None
    
    @property
    def spread_bps(self) -> Optional[Decimal]:
        """计算买卖价差（基点）"""
        if self.spread is not None and self.mid_price is not None:
            return (self.spread / self.mid_price) * 10000
        return None


@dataclass
class SignalEvent(Event):
    """交易信号事件"""
    event_type: EventType = field(default=EventType.SIGNAL, init=False)
    
    signal_type: SignalType
    strength: Decimal  # 信号强度 [-1, 1]
    confidence: Decimal  # 信号置信度 [0, 1]
    
    # 目标仓位信息
    target_position: Optional[Decimal] = None
    target_weight: Optional[Decimal] = None
    
    # 信号元数据
    strategy_name: str = "unknown"
    model_version: str = "1.0"
    features: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        super().__post_init__()
        
        # 验证信号参数
        if not -1 <= self.strength <= 1:
            raise ValueError("Signal strength must be between -1 and 1")
        if not 0 <= self.confidence <= 1:
            raise ValueError("Signal confidence must be between 0 and 1")


@dataclass
class OrderEvent(Event):
    """订单事件"""
    event_type: EventType = field(default=EventType.ORDER, init=False)
    
    order_id: str
    order_type: OrderType
    side: OrderSide
    quantity: Decimal
    
    # 价格信息
    price: Optional[Decimal] = None  # 限价单价格
    stop_price: Optional[Decimal] = None  # 止损价格
    
    # 订单参数
    time_in_force: str = "GTC"  # Good Till Cancelled
    reduce_only: bool = False
    post_only: bool = False
    
    # 策略信息
    strategy_name: str = "unknown"
    
    def __post_init__(self):
        super().__post_init__()
        
        # 验证订单参数
        if self.quantity <= 0:
            raise ValueError("Order quantity must be positive")
        if self.order_type == OrderType.LIMIT and self.price is None:
            raise ValueError("Limit orders must have a price")
        if self.order_type in [OrderType.STOP, OrderType.STOP_LIMIT] and self.stop_price is None:
            raise ValueError("Stop orders must have a stop price")


@dataclass
class FillEvent(Event):
    """成交事件"""
    event_type: EventType = field(default=EventType.FILL, init=False)
    
    order_id: str
    fill_id: str
    side: OrderSide
    quantity: Decimal
    price: Decimal
    
    # 成交信息
    commission: Decimal = Decimal('0')
    slippage: Decimal = Decimal('0')
    
    # 订单状态
    order_status: OrderStatus = OrderStatus.FILLED
    remaining_quantity: Decimal = Decimal('0')
    
    def __post_init__(self):
        super().__post_init__()
        
        # 验证成交参数
        if self.quantity <= 0:
            raise ValueError("Fill quantity must be positive")
        if self.price <= 0:
            raise ValueError("Fill price must be positive")
        if self.commission < 0:
            raise ValueError("Commission cannot be negative")
    
    @property
    def gross_amount(self) -> Decimal:
        """计算成交总额（不含手续费）"""
        return self.quantity * self.price
    
    @property
    def net_amount(self) -> Decimal:
        """计算成交净额（含手续费）"""
        if self.side == OrderSide.BUY:
            return self.gross_amount + self.commission
        else:
            return self.gross_amount - self.commission


@dataclass
class RiskEvent(Event):
    """风险控制事件"""
    event_type: EventType = field(default=EventType.RISK, init=False)
    
    risk_type: str  # "position_limit", "loss_limit", "volatility_limit", etc.
    severity: str  # "warning", "critical", "emergency"
    message: str
    
    # 风险指标
    current_value: Decimal
    limit_value: Decimal
    
    # 建议动作
    suggested_action: str = "reduce_position"
    
    def __post_init__(self):
        super().__post_init__()
        
        # 验证风险参数
        if self.severity not in ["warning", "critical", "emergency"]:
            raise ValueError("Severity must be 'warning', 'critical', or 'emergency'")


@dataclass
class PortfolioEvent(Event):
    """投资组合事件"""
    event_type: EventType = field(default=EventType.PORTFOLIO, init=False)
    
    # 资产组合状态
    total_value: Decimal
    cash: Decimal
    positions: Dict[str, Decimal]
    
    # 风险指标
    var_95: Optional[Decimal] = None
    var_99: Optional[Decimal] = None
    max_drawdown: Optional[Decimal] = None
    
    # 绩效指标
    unrealized_pnl: Decimal = Decimal('0')
    realized_pnl: Decimal = Decimal('0')
    
    def __post_init__(self):
        super().__post_init__()
        
        # 验证组合参数
        if self.total_value < 0:
            raise ValueError("Total portfolio value cannot be negative")
    
    @property
    def net_liquidation_value(self) -> Decimal:
        """计算净清算价值"""
        return self.total_value
    
    @property
    def total_pnl(self) -> Decimal:
        """计算总盈亏"""
        return self.unrealized_pnl + self.realized_pnl


@dataclass
class HeartbeatEvent(Event):
    """心跳事件 - 用于时间管理"""
    event_type: EventType = field(default=EventType.HEARTBEAT, init=False)
    
    def __post_init__(self):
        super().__post_init__()


class EventQueue:
    """事件队列 - 管理事件的优先级处理"""
    
    def __init__(self):
        self._events: List[Event] = []
        self._event_counts: Dict[EventType, int] = {}
    
    def put(self, event: Event):
        """添加事件到队列"""
        self._events.append(event)
        self._event_counts[event.event_type] = self._event_counts.get(event.event_type, 0) + 1
    
    def get(self) -> Optional[Event]:
        """获取下一个事件"""
        if not self._events:
            return None
        
        # 按时间戳排序获取最早的事件
        self._events.sort(key=lambda x: x.timestamp)
        event = self._events.pop(0)
        self._event_counts[event.event_type] -= 1
        
        return event
    
    def peek(self) -> Optional[Event]:
        """查看下一个事件但不移除"""
        if not self._events:
            return None
        
        self._events.sort(key=lambda x: x.timestamp)
        return self._events[0]
    
    def size(self) -> int:
        """获取队列大小"""
        return len(self._events)
    
    def empty(self) -> bool:
        """检查队列是否为空"""
        return len(self._events) == 0
    
    def clear(self):
        """清空队列"""
        self._events.clear()
        self._event_counts.clear()
    
    def get_event_counts(self) -> Dict[EventType, int]:
        """获取各类型事件计数"""
        return self._event_counts.copy()


class EventHandler(ABC):
    """事件处理器基类"""
    
    @abstractmethod
    def handle_event(self, event: Event) -> Optional[List[Event]]:
        """
        处理事件
        
        Args:
            event: 输入事件
            
        Returns:
            可选的输出事件列表
        """
        pass
    
    def can_handle(self, event: Event) -> bool:
        """检查是否可以处理该事件"""
        return True


class EventDispatcher:
    """事件分发器 - 将事件分发给相应的处理器"""
    
    def __init__(self):
        self._handlers: Dict[EventType, List[EventHandler]] = {}
    
    def register_handler(self, event_type: EventType, handler: EventHandler):
        """注册事件处理器"""
        if event_type not in self._handlers:
            self._handlers[event_type] = []
        self._handlers[event_type].append(handler)
    
    def unregister_handler(self, event_type: EventType, handler: EventHandler):
        """注销事件处理器"""
        if event_type in self._handlers:
            self._handlers[event_type].remove(handler)
    
    def dispatch(self, event: Event) -> List[Event]:
        """分发事件给所有注册的处理器"""
        output_events = []
        
        if event.event_type in self._handlers:
            for handler in self._handlers[event.event_type]:
                if handler.can_handle(event):
                    result = handler.handle_event(event)
                    if result:
                        output_events.extend(result)
        
        return output_events
    
    def clear_handlers(self):
        """清空所有处理器"""
        self._handlers.clear()