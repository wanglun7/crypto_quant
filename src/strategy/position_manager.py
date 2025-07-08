"""
仓位管理器 - 专业的仓位管理和资金配置

该模块实现了专业的仓位管理系统，包括：
- Kelly准则动态仓位计算
- 风险平价分配
- 动态杠杆调整
- 多层级止损止盈
- 相关性控制
- 流动性管理
- 最大回撤控制
"""

import logging
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from decimal import Decimal, getcontext
from typing import Dict, List, Optional, Tuple, Any, Union
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from enum import Enum
import warnings
from scipy.optimize import minimize
import math

from ..backtesting.events import EventHandler, Event, SignalEvent, OrderEvent, OrderType, OrderSide

# 设置Decimal精度
getcontext().prec = 28

logger = logging.getLogger(__name__)


class PositionSizingMethod(Enum):
    """仓位计算方法"""
    FIXED = "fixed"                    # 固定仓位
    KELLY = "kelly"                    # Kelly准则
    RISK_PARITY = "risk_parity"        # 风险平价
    VOLATILITY_TARGET = "vol_target"   # 波动率目标
    SHARPE_OPTIMAL = "sharpe_optimal"  # 夏普最优
    EQUAL_WEIGHT = "equal_weight"      # 等权重
    MOMENTUM_BASED = "momentum_based"  # 基于动量
    MEAN_REVERSION = "mean_reversion"  # 基于均值回归


class RiskLevel(Enum):
    """风险等级"""
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"
    EXTREME = "extreme"


@dataclass
class PositionLimits:
    """仓位限制"""
    max_position_size: float = 0.2        # 最大单仓位 (20%)
    max_total_exposure: float = 1.0       # 最大总敞口 (100%)
    max_leverage: float = 3.0             # 最大杠杆
    max_correlation_exposure: float = 0.5  # 最大相关性敞口
    
    # 止损设置
    max_individual_loss: float = 0.02     # 单笔最大损失 (2%)
    max_daily_loss: float = 0.05          # 日最大损失 (5%)
    max_drawdown: float = 0.15            # 最大回撤 (15%)
    
    # 流动性限制
    max_volume_participation: float = 0.1  # 最大成交量参与度 (10%)
    min_daily_volume: float = 1000000     # 最小日成交量 (USD)
    
    # 时间限制
    max_holding_period: timedelta = timedelta(days=30)
    min_holding_period: timedelta = timedelta(minutes=5)


@dataclass
class PositionInfo:
    """持仓信息"""
    symbol: str
    side: OrderSide
    size: Decimal
    entry_price: Decimal
    current_price: Decimal
    entry_time: datetime
    
    # 损益信息
    unrealized_pnl: Decimal = Decimal('0')
    realized_pnl: Decimal = Decimal('0')
    
    # 风险控制
    stop_loss: Optional[Decimal] = None
    take_profit: Optional[Decimal] = None
    trailing_stop: Optional[Decimal] = None
    
    # 元数据
    strategy_name: str = "unknown"
    signal_strength: float = 0.0
    risk_contribution: float = 0.0
    
    @property
    def market_value(self) -> Decimal:
        """市值"""
        return self.size * self.current_price
    
    @property
    def pnl_pct(self) -> float:
        """收益率百分比"""
        if self.entry_price == 0:
            return 0.0
        
        if self.side == OrderSide.BUY:
            return float((self.current_price - self.entry_price) / self.entry_price)
        else:
            return float((self.entry_price - self.current_price) / self.entry_price)
    
    @property
    def holding_period(self) -> timedelta:
        """持仓时间"""
        return datetime.now() - self.entry_time
    
    @property
    def is_profitable(self) -> bool:
        """是否盈利"""
        return self.pnl_pct > 0


@dataclass
class PositionSizingConfig:
    """仓位管理配置"""
    # 基本设置
    method: PositionSizingMethod = PositionSizingMethod.KELLY
    risk_level: RiskLevel = RiskLevel.MODERATE
    
    # Kelly准则参数
    kelly_lookback: int = 252              # 胜率和盈亏比计算窗口
    kelly_multiplier: float = 0.25         # Kelly系数乘数（保守调整）
    max_kelly_fraction: float = 0.2        # 最大Kelly比例
    
    # 风险平价参数
    risk_target: float = 0.15              # 目标风险（年化波动率）
    min_weight: float = 0.01               # 最小权重
    max_weight: float = 0.3                # 最大权重
    
    # 波动率目标参数
    volatility_target: float = 0.2         # 目标年化波动率
    volatility_lookback: int = 60          # 波动率计算窗口
    
    # 动态调整参数
    rebalance_frequency: timedelta = timedelta(hours=1)
    min_position_change: float = 0.05      # 最小仓位变化才重新平衡
    
    # 相关性控制
    correlation_lookback: int = 100        # 相关性计算窗口
    max_correlation_threshold: float = 0.7  # 最大相关性阈值
    
    # 流动性管理
    liquidity_buffer: float = 0.1          # 流动性缓冲
    emergency_exit_threshold: float = 0.1   # 紧急退出阈值
    
    # 仓位限制
    limits: PositionLimits = field(default_factory=PositionLimits)


class PositionManager(EventHandler):
    """仓位管理器"""
    
    def __init__(self, config: PositionSizingConfig, initial_capital: Decimal):
        self.config = config
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.logger = logging.getLogger(__name__)
        
        # 持仓管理
        self.positions: Dict[str, PositionInfo] = {}
        self.pending_orders: Dict[str, OrderEvent] = {}
        
        # 历史数据
        self.price_history: Dict[str, List[float]] = {}
        self.return_history: Dict[str, List[float]] = {}
        self.correlation_matrix: Optional[pd.DataFrame] = None
        
        # 性能跟踪
        self.equity_curve: List[Tuple[datetime, Decimal]] = []
        self.drawdown_history: List[float] = []
        self.trade_history: List[Dict] = []
        
        # 风险监控
        self.daily_pnl: Dict[str, Decimal] = {}  # 日期 -> PnL
        self.risk_metrics: Dict[str, float] = {}
        
        # 动态参数
        self.last_rebalance_time: Optional[datetime] = None
        self.current_volatility: float = 0.0
        self.current_sharpe: float = 0.0
        
        logger.info(f"Position manager initialized with {initial_capital} capital")
    
    def handle_event(self, event: Event) -> Optional[List[Event]]:
        """处理事件"""
        if isinstance(event, SignalEvent):
            return self._handle_signal(event)
        return None
    
    def _handle_signal(self, signal: SignalEvent) -> List[Event]:
        """处理交易信号"""
        try:
            # 计算建议仓位大小
            position_size = self._calculate_position_size(signal)
            
            if position_size == 0:
                self.logger.debug(f"Zero position size calculated for {signal.symbol}")
                return []
            
            # 风险检查
            if not self._risk_check(signal, position_size):
                self.logger.warning(f"Risk check failed for {signal.symbol}")
                return []
            
            # 流动性检查
            if not self._liquidity_check(signal, position_size):
                self.logger.warning(f"Liquidity check failed for {signal.symbol}")
                return []
            
            # 生成订单
            orders = self._generate_orders(signal, position_size)
            
            return orders
            
        except Exception as e:
            self.logger.error(f"Error handling signal: {e}")
            return []
    
    def _calculate_position_size(self, signal: SignalEvent) -> Decimal:
        """计算仓位大小"""
        if self.config.method == PositionSizingMethod.KELLY:
            return self._calculate_kelly_size(signal)
        elif self.config.method == PositionSizingMethod.RISK_PARITY:
            return self._calculate_risk_parity_size(signal)
        elif self.config.method == PositionSizingMethod.VOLATILITY_TARGET:
            return self._calculate_volatility_target_size(signal)
        elif self.config.method == PositionSizingMethod.FIXED:
            return self._calculate_fixed_size(signal)
        elif self.config.method == PositionSizingMethod.SHARPE_OPTIMAL:
            return self._calculate_sharpe_optimal_size(signal)
        else:
            return self._calculate_equal_weight_size(signal)
    
    def _calculate_kelly_size(self, signal: SignalEvent) -> Decimal:
        """基于Kelly准则计算仓位"""
        symbol = signal.symbol
        
        # 获取历史收益数据
        if symbol not in self.return_history or len(self.return_history[symbol]) < 20:
            # 如果历史数据不足，使用保守的固定仓位
            return self.current_capital * Decimal(str(self.config.limits.max_position_size * 0.5))
        
        returns = np.array(self.return_history[symbol][-self.config.kelly_lookback:])
        
        # 计算胜率和平均盈亏
        positive_returns = returns[returns > 0]
        negative_returns = returns[returns < 0]
        
        if len(positive_returns) == 0 or len(negative_returns) == 0:
            return Decimal('0')
        
        win_rate = len(positive_returns) / len(returns)
        avg_win = np.mean(positive_returns)
        avg_loss = abs(np.mean(negative_returns))
        
        # Kelly公式: f = (bp - q) / b
        # 其中 b = 平均盈利/平均亏损, p = 胜率, q = 败率
        if avg_loss == 0:
            return Decimal('0')
        
        b = avg_win / avg_loss
        p = win_rate
        q = 1 - win_rate
        
        kelly_fraction = (b * p - q) / b
        
        # 安全调整
        kelly_fraction = max(0, min(kelly_fraction, self.config.max_kelly_fraction))
        kelly_fraction *= self.config.kelly_multiplier
        
        # 信号强度调整
        kelly_fraction *= float(signal.strength)
        
        # 计算仓位大小
        position_value = self.current_capital * Decimal(str(kelly_fraction))
        
        # 应用限制
        max_position = self.current_capital * Decimal(str(self.config.limits.max_position_size))
        position_value = min(position_value, max_position)
        
        self.logger.debug(f"Kelly calculation for {symbol}: "
                         f"win_rate={win_rate:.3f}, avg_win={avg_win:.4f}, "
                         f"avg_loss={avg_loss:.4f}, kelly_fraction={kelly_fraction:.3f}")
        
        return position_value
    
    def _calculate_risk_parity_size(self, signal: SignalEvent) -> Decimal:
        """基于风险平价计算仓位"""
        symbol = signal.symbol
        
        # 计算个股波动率
        if symbol not in self.return_history or len(self.return_history[symbol]) < 20:
            return self.current_capital * Decimal('0.1')  # 默认10%
        
        returns = np.array(self.return_history[symbol][-60:])  # 使用60个周期
        volatility = np.std(returns) * np.sqrt(252 * 24)  # 年化波动率（假设每日24个数据点）
        
        if volatility == 0:
            return Decimal('0')
        
        # 目标风险贡献
        target_risk_contribution = self.config.risk_target / len(self.positions or [signal.symbol])
        
        # 计算所需仓位以达到目标风险
        position_weight = target_risk_contribution / volatility
        
        # 信号强度调整
        position_weight *= float(signal.strength)
        
        # 应用限制
        position_weight = max(self.config.min_weight, 
                            min(position_weight, self.config.max_weight))
        
        position_value = self.current_capital * Decimal(str(position_weight))
        
        self.logger.debug(f"Risk parity calculation for {symbol}: "
                         f"volatility={volatility:.4f}, "
                         f"position_weight={position_weight:.3f}")
        
        return position_value
    
    def _calculate_volatility_target_size(self, signal: SignalEvent) -> Decimal:
        """基于波动率目标计算仓位"""
        symbol = signal.symbol
        
        if symbol not in self.return_history or len(self.return_history[symbol]) < 20:
            return self.current_capital * Decimal('0.1')
        
        returns = np.array(self.return_history[symbol][-self.config.volatility_lookback:])
        current_vol = np.std(returns) * np.sqrt(252 * 24)  # 年化波动率
        
        if current_vol == 0:
            return Decimal('0')
        
        # 波动率缩放
        vol_scalar = self.config.volatility_target / current_vol
        
        # 基础权重
        base_weight = 0.2  # 20%基础权重
        
        # 应用波动率调整和信号强度
        adjusted_weight = base_weight * vol_scalar * float(signal.strength)
        
        # 应用限制
        adjusted_weight = max(0.01, min(adjusted_weight, self.config.limits.max_position_size))
        
        position_value = self.current_capital * Decimal(str(adjusted_weight))
        
        self.logger.debug(f"Volatility target calculation for {symbol}: "
                         f"current_vol={current_vol:.4f}, "
                         f"vol_scalar={vol_scalar:.3f}, "
                         f"adjusted_weight={adjusted_weight:.3f}")
        
        return position_value
    
    def _calculate_fixed_size(self, signal: SignalEvent) -> Decimal:
        """固定仓位大小"""
        base_size = self.config.limits.max_position_size * 0.5  # 使用最大仓位的50%作为基础
        adjusted_size = base_size * float(signal.strength)
        
        return self.current_capital * Decimal(str(adjusted_size))
    
    def _calculate_sharpe_optimal_size(self, signal: SignalEvent) -> Decimal:
        """基于夏普比率优化的仓位"""
        symbol = signal.symbol
        
        if symbol not in self.return_history or len(self.return_history[symbol]) < 50:
            return self.current_capital * Decimal('0.1')
        
        returns = np.array(self.return_history[symbol][-100:])
        
        # 计算夏普比率
        mean_return = np.mean(returns)
        vol_return = np.std(returns)
        
        if vol_return == 0:
            return Decimal('0')
        
        sharpe_ratio = mean_return / vol_return
        
        # 基于夏普比率调整仓位
        # 夏普比率越高，仓位越大
        if sharpe_ratio > 1:
            weight_multiplier = min(2.0, 1 + sharpe_ratio * 0.5)
        elif sharpe_ratio > 0:
            weight_multiplier = 0.5 + sharpe_ratio * 0.5
        else:
            weight_multiplier = 0.1  # 负夏普比率时使用很小的仓位
        
        base_weight = 0.15
        adjusted_weight = base_weight * weight_multiplier * float(signal.strength)
        adjusted_weight = min(adjusted_weight, self.config.limits.max_position_size)
        
        return self.current_capital * Decimal(str(adjusted_weight))
    
    def _calculate_equal_weight_size(self, signal: SignalEvent) -> Decimal:
        """等权重仓位"""
        num_positions = max(1, len(self.positions) + 1)
        weight = 1.0 / num_positions
        weight *= float(signal.strength)
        weight = min(weight, self.config.limits.max_position_size)
        
        return self.current_capital * Decimal(str(weight))
    
    def _risk_check(self, signal: SignalEvent, position_size: Decimal) -> bool:
        """风险检查"""
        symbol = signal.symbol
        
        # 检查单仓位限制
        position_pct = float(position_size / self.current_capital)
        if position_pct > self.config.limits.max_position_size:
            self.logger.warning(f"Position size {position_pct:.2%} exceeds limit "
                              f"{self.config.limits.max_position_size:.2%}")
            return False
        
        # 检查总敞口
        current_exposure = sum(float(pos.market_value) for pos in self.positions.values())
        new_exposure = current_exposure + float(position_size)
        exposure_pct = new_exposure / float(self.current_capital)
        
        if exposure_pct > self.config.limits.max_total_exposure:
            self.logger.warning(f"Total exposure {exposure_pct:.2%} exceeds limit "
                              f"{self.config.limits.max_total_exposure:.2%}")
            return False
        
        # 检查相关性敞口
        if not self._check_correlation_risk(signal, position_size):
            return False
        
        # 检查日内损失限制
        today_str = signal.timestamp.strftime('%Y-%m-%d')
        daily_pnl = self.daily_pnl.get(today_str, Decimal('0'))
        if float(daily_pnl / self.current_capital) < -self.config.limits.max_daily_loss:
            self.logger.warning(f"Daily loss limit exceeded: {float(daily_pnl / self.current_capital):.2%}")
            return False
        
        # 检查最大回撤
        if self._get_current_drawdown() > self.config.limits.max_drawdown:
            self.logger.warning(f"Maximum drawdown exceeded: {self._get_current_drawdown():.2%}")
            return False
        
        return True
    
    def _check_correlation_risk(self, signal: SignalEvent, position_size: Decimal) -> bool:
        """检查相关性风险"""
        if len(self.positions) == 0 or self.correlation_matrix is None:
            return True
        
        symbol = signal.symbol
        
        # 计算与现有仓位的相关性敞口
        correlated_exposure = Decimal('0')
        
        for pos_symbol, position in self.positions.items():
            if pos_symbol in self.correlation_matrix.index and symbol in self.correlation_matrix.columns:
                correlation = self.correlation_matrix.loc[pos_symbol, symbol]
                
                if abs(correlation) > self.config.correlation_threshold:
                    correlated_exposure += position.market_value * Decimal(str(abs(correlation)))
        
        # 加上新仓位的敞口
        correlated_exposure += position_size
        
        correlation_pct = float(correlated_exposure / self.current_capital)
        if correlation_pct > self.config.limits.max_correlation_exposure:
            self.logger.warning(f"Correlation exposure {correlation_pct:.2%} exceeds limit "
                              f"{self.config.limits.max_correlation_exposure:.2%}")
            return False
        
        return True
    
    def _liquidity_check(self, signal: SignalEvent, position_size: Decimal) -> bool:
        """流动性检查"""
        # 这里应该检查实际的市场流动性
        # 简化实现：假设所有信号都有足够流动性
        return True
    
    def _generate_orders(self, signal: SignalEvent, position_size: Decimal) -> List[OrderEvent]:
        """生成订单"""
        orders = []
        
        # 检查是否已有该品种的仓位
        existing_position = self.positions.get(signal.symbol)
        
        if existing_position:
            # 调整现有仓位
            orders.extend(self._adjust_existing_position(signal, existing_position, position_size))
        else:
            # 开新仓
            orders.extend(self._open_new_position(signal, position_size))
        
        return orders
    
    def _adjust_existing_position(self, signal: SignalEvent, existing_position: PositionInfo, target_size: Decimal) -> List[OrderEvent]:
        """调整现有仓位"""
        orders = []
        
        current_value = existing_position.market_value
        target_value = target_size
        
        # 计算需要调整的金额
        adjustment = target_value - current_value
        
        # 如果调整幅度很小，不执行
        if abs(float(adjustment / self.current_capital)) < self.config.min_position_change:
            return orders
        
        # 确定订单方向
        if adjustment > 0:
            # 需要增仓
            if (signal.signal_type == SignalType.LONG and existing_position.side == OrderSide.BUY) or \
               (signal.signal_type == SignalType.SHORT and existing_position.side == OrderSide.SELL):
                # 同向增仓
                order_side = existing_position.side
                order_size = adjustment / existing_position.current_price
            else:
                # 反向需要先平仓再开新仓
                return self._close_and_reopen_position(signal, existing_position, target_size)
        else:
            # 需要减仓
            order_side = OrderSide.SELL if existing_position.side == OrderSide.BUY else OrderSide.BUY
            order_size = abs(adjustment) / existing_position.current_price
        
        # 创建调整订单
        order = OrderEvent(
            timestamp=signal.timestamp,
            symbol=signal.symbol,
            order_id=f"adjust_{signal.symbol}_{signal.timestamp.strftime('%Y%m%d_%H%M%S')}",
            order_type=OrderType.MARKET,
            side=order_side,
            quantity=order_size,
            strategy_name=signal.strategy_name
        )
        
        orders.append(order)
        return orders
    
    def _open_new_position(self, signal: SignalEvent, position_size: Decimal) -> List[OrderEvent]:
        """开新仓位"""
        orders = []
        
        # 确定订单方向
        if signal.signal_type == SignalType.LONG:
            order_side = OrderSide.BUY
        elif signal.signal_type == SignalType.SHORT:
            order_side = OrderSide.SELL
        else:
            return orders  # NEUTRAL信号不开仓
        
        # 估算当前价格（这里简化处理）
        # 在实际系统中应该从最新市场数据获取
        estimated_price = Decimal('50000')  # 需要从市场数据获取实际价格
        
        # 计算订单数量
        order_quantity = position_size / estimated_price
        
        # 创建主订单
        main_order = OrderEvent(
            timestamp=signal.timestamp,
            symbol=signal.symbol,
            order_id=f"open_{signal.symbol}_{signal.timestamp.strftime('%Y%m%d_%H%M%S')}",
            order_type=OrderType.MARKET,
            side=order_side,
            quantity=order_quantity,
            strategy_name=signal.strategy_name
        )
        
        orders.append(main_order)
        
        # 创建止损订单
        stop_loss_price = self._calculate_stop_loss(estimated_price, order_side)
        if stop_loss_price:
            stop_order = OrderEvent(
                timestamp=signal.timestamp,
                symbol=signal.symbol,
                order_id=f"stop_{signal.symbol}_{signal.timestamp.strftime('%Y%m%d_%H%M%S')}",
                order_type=OrderType.STOP,
                side=OrderSide.SELL if order_side == OrderSide.BUY else OrderSide.BUY,
                quantity=order_quantity,
                stop_price=stop_loss_price,
                strategy_name=signal.strategy_name
            )
            orders.append(stop_order)
        
        # 创建止盈订单
        take_profit_price = self._calculate_take_profit(estimated_price, order_side)
        if take_profit_price:
            profit_order = OrderEvent(
                timestamp=signal.timestamp,
                symbol=signal.symbol,
                order_id=f"profit_{signal.symbol}_{signal.timestamp.strftime('%Y%m%d_%H%M%S')}",
                order_type=OrderType.LIMIT,
                side=OrderSide.SELL if order_side == OrderSide.BUY else OrderSide.BUY,
                quantity=order_quantity,
                price=take_profit_price,
                strategy_name=signal.strategy_name
            )
            orders.append(profit_order)
        
        return orders
    
    def _close_and_reopen_position(self, signal: SignalEvent, existing_position: PositionInfo, target_size: Decimal) -> List[OrderEvent]:
        """平仓并重新开仓"""
        orders = []
        
        # 先平仓
        close_order = OrderEvent(
            timestamp=signal.timestamp,
            symbol=signal.symbol,
            order_id=f"close_{signal.symbol}_{signal.timestamp.strftime('%Y%m%d_%H%M%S')}",
            order_type=OrderType.MARKET,
            side=OrderSide.SELL if existing_position.side == OrderSide.BUY else OrderSide.BUY,
            quantity=existing_position.size,
            strategy_name=signal.strategy_name
        )
        orders.append(close_order)
        
        # 再开新仓
        new_orders = self._open_new_position(signal, target_size)
        orders.extend(new_orders)
        
        return orders
    
    def _calculate_stop_loss(self, entry_price: Decimal, side: OrderSide) -> Optional[Decimal]:
        """计算止损价格"""
        stop_loss_pct = Decimal(str(self.config.limits.max_individual_loss))
        
        if side == OrderSide.BUY:
            # 多头止损：入场价 * (1 - 止损百分比)
            return entry_price * (Decimal('1') - stop_loss_pct)
        else:
            # 空头止损：入场价 * (1 + 止损百分比)
            return entry_price * (Decimal('1') + stop_loss_pct)
    
    def _calculate_take_profit(self, entry_price: Decimal, side: OrderSide) -> Optional[Decimal]:
        """计算止盈价格"""
        # 使用2倍的止损幅度作为止盈
        profit_pct = Decimal(str(self.config.limits.max_individual_loss * 2))
        
        if side == OrderSide.BUY:
            # 多头止盈：入场价 * (1 + 止盈百分比)
            return entry_price * (Decimal('1') + profit_pct)
        else:
            # 空头止盈：入场价 * (1 - 止盈百分比)
            return entry_price * (Decimal('1') - profit_pct)
    
    def _get_current_drawdown(self) -> float:
        """计算当前回撤"""
        if len(self.equity_curve) < 2:
            return 0.0
        
        # 计算最大净值和当前净值
        peak_equity = max(equity for _, equity in self.equity_curve)
        current_equity = self.equity_curve[-1][1]
        
        if peak_equity == 0:
            return 0.0
        
        drawdown = float((peak_equity - current_equity) / peak_equity)
        return max(0.0, drawdown)
    
    def update_positions(self, market_data: Dict[str, Decimal]):
        """更新持仓信息"""
        for symbol, position in self.positions.items():
            if symbol in market_data:
                position.current_price = market_data[symbol]
                
                # 更新未实现盈亏
                if position.side == OrderSide.BUY:
                    position.unrealized_pnl = (position.current_price - position.entry_price) * position.size
                else:
                    position.unrealized_pnl = (position.entry_price - position.current_price) * position.size
        
        # 更新净值曲线
        total_equity = self._calculate_total_equity()
        self.equity_curve.append((datetime.now(), total_equity))
        
        # 限制历史数据长度
        if len(self.equity_curve) > 10000:
            self.equity_curve = self.equity_curve[-5000:]
    
    def _calculate_total_equity(self) -> Decimal:
        """计算总权益"""
        total_unrealized = sum(pos.unrealized_pnl for pos in self.positions.values())
        return self.current_capital + total_unrealized
    
    def get_portfolio_summary(self) -> Dict[str, Any]:
        """获取投资组合摘要"""
        total_equity = self._calculate_total_equity()
        total_unrealized = sum(pos.unrealized_pnl for pos in self.positions.values())
        total_realized = sum(pos.realized_pnl for pos in self.positions.values())
        
        return {
            'total_equity': float(total_equity),
            'cash': float(self.current_capital),
            'unrealized_pnl': float(total_unrealized),
            'realized_pnl': float(total_realized),
            'total_return': float((total_equity - self.initial_capital) / self.initial_capital),
            'current_drawdown': self._get_current_drawdown(),
            'num_positions': len(self.positions),
            'largest_position': max([float(pos.market_value / total_equity) for pos in self.positions.values()], default=0.0),
            'leverage': float(sum(pos.market_value for pos in self.positions.values()) / total_equity) if total_equity > 0 else 0.0,
            'positions': {
                symbol: {
                    'side': pos.side.value,
                    'size': float(pos.size),
                    'entry_price': float(pos.entry_price),
                    'current_price': float(pos.current_price),
                    'unrealized_pnl': float(pos.unrealized_pnl),
                    'pnl_pct': pos.pnl_pct,
                    'market_value': float(pos.market_value),
                    'holding_period_hours': pos.holding_period.total_seconds() / 3600
                }
                for symbol, pos in self.positions.items()
            }
        }
    
    def should_rebalance(self) -> bool:
        """判断是否需要重新平衡"""
        if self.last_rebalance_time is None:
            return True
        
        time_since_rebalance = datetime.now() - self.last_rebalance_time
        return time_since_rebalance >= self.config.rebalance_frequency
    
    def update_correlation_matrix(self, symbols: List[str], returns_data: Dict[str, List[float]]):
        """更新相关性矩阵"""
        if len(symbols) < 2:
            return
        
        # 构建收益率DataFrame
        min_length = min(len(returns_data[symbol]) for symbol in symbols)
        if min_length < self.config.correlation_lookback:
            return
        
        df_data = {}
        for symbol in symbols:
            df_data[symbol] = returns_data[symbol][-min_length:]
        
        df = pd.DataFrame(df_data)
        self.correlation_matrix = df.corr()
        
        self.logger.debug(f"Updated correlation matrix for {len(symbols)} symbols")
    
    def add_position(self, symbol: str, side: OrderSide, size: Decimal, entry_price: Decimal, strategy_name: str = "unknown"):
        """添加新持仓"""
        position = PositionInfo(
            symbol=symbol,
            side=side,
            size=size,
            entry_price=entry_price,
            current_price=entry_price,
            entry_time=datetime.now(),
            strategy_name=strategy_name
        )
        
        self.positions[symbol] = position
        self.logger.info(f"Added position: {symbol} {side.value} {float(size)} @ {float(entry_price)}")
    
    def close_position(self, symbol: str, exit_price: Decimal) -> Optional[Decimal]:
        """平仓"""
        if symbol not in self.positions:
            return None
        
        position = self.positions[symbol]
        
        # 计算已实现盈亏
        if position.side == OrderSide.BUY:
            realized_pnl = (exit_price - position.entry_price) * position.size
        else:
            realized_pnl = (position.entry_price - exit_price) * position.size
        
        # 更新资本
        self.current_capital += realized_pnl
        
        # 记录交易
        trade_record = {
            'symbol': symbol,
            'side': position.side.value,
            'size': float(position.size),
            'entry_price': float(position.entry_price),
            'exit_price': float(exit_price),
            'entry_time': position.entry_time,
            'exit_time': datetime.now(),
            'holding_period': datetime.now() - position.entry_time,
            'realized_pnl': float(realized_pnl),
            'pnl_pct': float(realized_pnl / (position.entry_price * position.size)),
            'strategy_name': position.strategy_name
        }
        self.trade_history.append(trade_record)
        
        # 移除持仓
        del self.positions[symbol]
        
        self.logger.info(f"Closed position: {symbol} PnL: {float(realized_pnl):.2f}")
        
        return realized_pnl