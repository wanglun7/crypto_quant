"""
风险控制器 - 多层级风险管理系统

该模块实现了专业的风险控制系统，包括：
- 实时风险监控
- 多层级止损机制
- 敞口控制和限制
- 流动性风险管理
- 系统性风险检测
- 紧急停机机制
- 风险预警系统
"""

import logging
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from decimal import Decimal, getcontext
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from enum import Enum
import warnings
import asyncio
from threading import Lock
import json

from ..backtesting.events import EventHandler, Event, MarketDataEvent, OrderEvent, RiskEvent, OrderSide

# 设置Decimal精度
getcontext().prec = 28

logger = logging.getLogger(__name__)


class RiskLevel(Enum):
    """风险等级"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class RiskType(Enum):
    """风险类型"""
    POSITION_SIZE = "position_size"       # 仓位大小风险
    LEVERAGE = "leverage"                 # 杠杆风险
    CONCENTRATION = "concentration"       # 集中度风险
    CORRELATION = "correlation"           # 相关性风险
    LIQUIDITY = "liquidity"              # 流动性风险
    MARKET = "market"                    # 市场风险
    DRAWDOWN = "drawdown"                # 回撤风险
    VOLATILITY = "volatility"            # 波动率风险
    LOSS_LIMIT = "loss_limit"            # 损失限额风险
    SYSTEM = "system"                    # 系统风险
    OPERATIONAL = "operational"          # 操作风险


class RiskAction(Enum):
    """风险响应动作"""
    MONITOR = "monitor"                  # 监控
    WARN = "warn"                       # 警告
    REDUCE_POSITION = "reduce_position"  # 减仓
    CLOSE_POSITION = "close_position"   # 平仓
    HALT_TRADING = "halt_trading"       # 停止交易
    EMERGENCY_EXIT = "emergency_exit"   # 紧急退出
    SYSTEM_SHUTDOWN = "system_shutdown" # 系统停机


@dataclass
class RiskLimit:
    """风险限制"""
    name: str
    risk_type: RiskType
    threshold: float
    action: RiskAction
    enabled: bool = True
    
    # 时间相关限制
    lookback_period: timedelta = timedelta(hours=24)
    
    # 触发条件
    consecutive_breaches: int = 1  # 连续违规次数
    grace_period: timedelta = timedelta(minutes=5)  # 宽限期
    
    # 元数据
    description: str = ""
    last_breach_time: Optional[datetime] = None
    breach_count: int = 0
    
    def is_breached(self, current_value: float, timestamp: datetime) -> bool:
        """检查是否违规"""
        if not self.enabled:
            return False
        
        # 检查是否超过阈值
        if self.risk_type in [RiskType.DRAWDOWN, RiskType.LOSS_LIMIT]:
            # 对于损失类风险，负值表示损失
            breached = current_value < -abs(self.threshold)
        else:
            # 对于其他风险，超过阈值即违规
            breached = current_value > self.threshold
        
        if breached:
            # 更新违规统计
            if (self.last_breach_time is None or 
                timestamp - self.last_breach_time > self.grace_period):
                self.breach_count = 1
            else:
                self.breach_count += 1
            
            self.last_breach_time = timestamp
            
            # 检查是否达到连续违规次数
            return self.breach_count >= self.consecutive_breaches
        else:
            # 重置违规计数
            if (self.last_breach_time and 
                timestamp - self.last_breach_time > self.grace_period):
                self.breach_count = 0
            
            return False


@dataclass
class RiskMetrics:
    """风险指标"""
    timestamp: datetime
    
    # 基础风险指标
    total_exposure: float = 0.0
    leverage: float = 1.0
    concentration_risk: float = 0.0
    correlation_risk: float = 0.0
    
    # 损失指标
    unrealized_pnl: float = 0.0
    daily_pnl: float = 0.0
    max_drawdown: float = 0.0
    current_drawdown: float = 0.0
    
    # 波动率指标
    portfolio_volatility: float = 0.0
    var_95: float = 0.0
    var_99: float = 0.0
    expected_shortfall: float = 0.0
    
    # 流动性指标
    liquidity_ratio: float = 1.0
    largest_position_liquidity: float = 1.0
    
    # 系统指标
    system_latency: float = 0.0
    error_rate: float = 0.0
    
    # 预警标识
    warning_flags: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'timestamp': self.timestamp.isoformat(),
            'total_exposure': self.total_exposure,
            'leverage': self.leverage,
            'concentration_risk': self.concentration_risk,
            'correlation_risk': self.correlation_risk,
            'unrealized_pnl': self.unrealized_pnl,
            'daily_pnl': self.daily_pnl,
            'max_drawdown': self.max_drawdown,
            'current_drawdown': self.current_drawdown,
            'portfolio_volatility': self.portfolio_volatility,
            'var_95': self.var_95,
            'var_99': self.var_99,
            'expected_shortfall': self.expected_shortfall,
            'liquidity_ratio': self.liquidity_ratio,
            'largest_position_liquidity': self.largest_position_liquidity,
            'system_latency': self.system_latency,
            'error_rate': self.error_rate,
            'warning_flags': self.warning_flags
        }


@dataclass
class RiskConfig:
    """风险配置"""
    # 基础限制
    max_total_exposure: float = 1.0       # 最大总敞口 (100%)
    max_leverage: float = 3.0             # 最大杠杆
    max_position_size: float = 0.2        # 最大单仓位 (20%)
    max_concentration: float = 0.5        # 最大集中度 (50%)
    max_correlation_exposure: float = 0.6  # 最大相关性敞口 (60%)
    
    # 损失限制
    max_daily_loss: float = 0.05          # 最大日损失 (5%)
    max_position_loss: float = 0.02       # 最大单仓损失 (2%)
    max_drawdown: float = 0.15            # 最大回撤 (15%)
    stop_loss_multiplier: float = 1.5     # 止损乘数
    
    # 波动率控制
    max_portfolio_vol: float = 0.3        # 最大组合波动率 (30%)
    vol_lookback_days: int = 30           # 波动率计算窗口
    
    # 流动性要求
    min_liquidity_ratio: float = 0.1     # 最小流动性比率
    max_volume_participation: float = 0.1 # 最大成交量参与度
    
    # 系统限制
    max_system_latency: float = 1.0      # 最大系统延迟 (秒)
    max_error_rate: float = 0.01         # 最大错误率 (1%)
    
    # 监控频率
    risk_check_interval: timedelta = timedelta(seconds=30)
    metrics_update_interval: timedelta = timedelta(minutes=1)
    
    # 紧急设置
    emergency_exit_threshold: float = 0.1  # 紧急退出阈值
    circuit_breaker_threshold: float = 0.08 # 熔断阈值
    
    # 自定义限制
    custom_limits: List[RiskLimit] = field(default_factory=list)
    
    def get_default_limits(self) -> List[RiskLimit]:
        """获取默认风险限制"""
        return [
            # 仓位大小限制
            RiskLimit(
                name="max_position_size",
                risk_type=RiskType.POSITION_SIZE,
                threshold=self.max_position_size,
                action=RiskAction.REDUCE_POSITION,
                description="单仓位大小超限"
            ),
            
            # 杠杆限制
            RiskLimit(
                name="max_leverage",
                risk_type=RiskType.LEVERAGE,
                threshold=self.max_leverage,
                action=RiskAction.REDUCE_POSITION,
                description="杠杆倍数超限"
            ),
            
            # 集中度限制
            RiskLimit(
                name="max_concentration",
                risk_type=RiskType.CONCENTRATION,
                threshold=self.max_concentration,
                action=RiskAction.REDUCE_POSITION,
                description="仓位集中度过高"
            ),
            
            # 日损失限制
            RiskLimit(
                name="max_daily_loss",
                risk_type=RiskType.LOSS_LIMIT,
                threshold=self.max_daily_loss,
                action=RiskAction.HALT_TRADING,
                description="日损失超限",
                consecutive_breaches=2
            ),
            
            # 最大回撤限制
            RiskLimit(
                name="max_drawdown",
                risk_type=RiskType.DRAWDOWN,
                threshold=self.max_drawdown,
                action=RiskAction.EMERGENCY_EXIT,
                description="最大回撤超限",
                consecutive_breaches=3
            ),
            
            # 波动率限制
            RiskLimit(
                name="max_portfolio_vol",
                risk_type=RiskType.VOLATILITY,
                threshold=self.max_portfolio_vol,
                action=RiskAction.REDUCE_POSITION,
                description="组合波动率过高"
            ),
            
            # 流动性限制
            RiskLimit(
                name="min_liquidity",
                risk_type=RiskType.LIQUIDITY,
                threshold=self.min_liquidity_ratio,
                action=RiskAction.CLOSE_POSITION,
                description="流动性不足"
            ),
            
            # 熔断机制
            RiskLimit(
                name="circuit_breaker",
                risk_type=RiskType.LOSS_LIMIT,
                threshold=self.circuit_breaker_threshold,
                action=RiskAction.SYSTEM_SHUTDOWN,
                description="触发熔断机制",
                consecutive_breaches=1,
                grace_period=timedelta(seconds=0)
            )
        ]


class RiskController(EventHandler):
    """风险控制器"""
    
    def __init__(self, config: RiskConfig, initial_capital: Decimal):
        self.config = config
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.logger = logging.getLogger(__name__)
        
        # 风险限制
        self.risk_limits = config.get_default_limits() + config.custom_limits
        self.enabled_limits = {limit.name: limit for limit in self.risk_limits if limit.enabled}
        
        # 状态管理
        self.is_active = True
        self.emergency_mode = False
        self.trading_halted = False
        self.last_risk_check = datetime.now()
        
        # 历史数据
        self.equity_history: List[Tuple[datetime, Decimal]] = []
        self.pnl_history: List[Tuple[datetime, Decimal]] = []
        self.risk_metrics_history: List[RiskMetrics] = []
        
        # 当前风险状态
        self.current_metrics: Optional[RiskMetrics] = None
        self.active_warnings: Dict[str, RiskEvent] = {}
        self.breach_history: List[Dict] = []
        
        # 线程安全
        self._lock = Lock()
        
        # 回调函数
        self.risk_callbacks: Dict[RiskLevel, List[Callable]] = {
            level: [] for level in RiskLevel
        }
        
        # 统计数据
        self.total_risk_events = 0
        self.actions_taken = {action: 0 for action in RiskAction}
        
        self.logger.info(f"Risk controller initialized with {len(self.risk_limits)} limits")
    
    def handle_event(self, event: Event) -> Optional[List[Event]]:
        """处理事件"""
        if isinstance(event, MarketDataEvent):
            return self._handle_market_data(event)
        elif isinstance(event, OrderEvent):
            return self._handle_order_event(event)
        return None
    
    def _handle_market_data(self, event: MarketDataEvent) -> List[Event]:
        """处理市场数据事件"""
        # 检查是否需要进行风险检查
        if datetime.now() - self.last_risk_check < self.config.risk_check_interval:
            return []
        
        # 更新风险指标
        self._update_risk_metrics(event.timestamp)
        
        # 执行风险检查
        risk_events = self._perform_risk_check(event.timestamp)
        
        self.last_risk_check = datetime.now()
        
        return risk_events
    
    def _handle_order_event(self, event: OrderEvent) -> List[Event]:
        """处理订单事件"""
        # 检查订单是否符合风险限制
        if not self._validate_order(event):
            # 创建风险事件阻止订单
            risk_event = RiskEvent(
                timestamp=event.timestamp,
                symbol=event.symbol,
                risk_type="order_validation",
                severity="critical",
                message=f"Order {event.order_id} blocked by risk control",
                current_value=Decimal('0'),
                limit_value=Decimal('0'),
                suggested_action="cancel_order"
            )
            return [risk_event]
        
        return []
    
    def _update_risk_metrics(self, timestamp: datetime):
        """更新风险指标"""
        with self._lock:
            try:
                metrics = RiskMetrics(timestamp=timestamp)
                
                # 计算基础风险指标
                metrics.total_exposure = self._calculate_total_exposure()
                metrics.leverage = self._calculate_leverage()
                metrics.concentration_risk = self._calculate_concentration_risk()
                metrics.correlation_risk = self._calculate_correlation_risk()
                
                # 计算损失指标
                metrics.unrealized_pnl = self._calculate_unrealized_pnl()
                metrics.daily_pnl = self._calculate_daily_pnl(timestamp)
                metrics.max_drawdown = self._calculate_max_drawdown()
                metrics.current_drawdown = self._calculate_current_drawdown()
                
                # 计算波动率指标
                metrics.portfolio_volatility = self._calculate_portfolio_volatility()
                metrics.var_95, metrics.var_99 = self._calculate_var()
                metrics.expected_shortfall = self._calculate_expected_shortfall()
                
                # 计算流动性指标
                metrics.liquidity_ratio = self._calculate_liquidity_ratio()
                metrics.largest_position_liquidity = self._calculate_largest_position_liquidity()
                
                # 检查预警标识
                metrics.warning_flags = self._check_warning_flags()
                
                self.current_metrics = metrics
                self.risk_metrics_history.append(metrics)
                
                # 限制历史数据长度
                if len(self.risk_metrics_history) > 10000:
                    self.risk_metrics_history = self.risk_metrics_history[-5000:]
                
            except Exception as e:
                self.logger.error(f"Error updating risk metrics: {e}")
    
    def _perform_risk_check(self, timestamp: datetime) -> List[RiskEvent]:
        """执行风险检查"""
        if not self.current_metrics:
            return []
        
        risk_events = []
        
        for limit_name, limit in self.enabled_limits.items():
            try:
                # 获取当前风险值
                current_value = self._get_risk_value(limit.risk_type)
                
                # 检查是否违规
                if limit.is_breached(current_value, timestamp):
                    # 创建风险事件
                    risk_event = self._create_risk_event(limit, current_value, timestamp)
                    risk_events.append(risk_event)
                    
                    # 执行风险响应
                    self._execute_risk_action(limit, risk_event)
                    
                    # 记录违规历史
                    self._record_breach(limit, current_value, timestamp)
                
            except Exception as e:
                self.logger.error(f"Error checking limit {limit_name}: {e}")
        
        return risk_events
    
    def _get_risk_value(self, risk_type: RiskType) -> float:
        """获取指定类型的风险值"""
        if not self.current_metrics:
            return 0.0
        
        risk_value_map = {
            RiskType.POSITION_SIZE: self._get_largest_position_size(),
            RiskType.LEVERAGE: self.current_metrics.leverage,
            RiskType.CONCENTRATION: self.current_metrics.concentration_risk,
            RiskType.CORRELATION: self.current_metrics.correlation_risk,
            RiskType.LIQUIDITY: self.current_metrics.liquidity_ratio,
            RiskType.MARKET: self.current_metrics.var_95,
            RiskType.DRAWDOWN: self.current_metrics.current_drawdown,
            RiskType.VOLATILITY: self.current_metrics.portfolio_volatility,
            RiskType.LOSS_LIMIT: self.current_metrics.daily_pnl,
        }
        
        return risk_value_map.get(risk_type, 0.0)
    
    def _create_risk_event(self, limit: RiskLimit, current_value: float, timestamp: datetime) -> RiskEvent:
        """创建风险事件"""
        # 确定严重程度
        severity = self._determine_severity(limit, current_value)
        
        # 生成消息
        message = f"{limit.description}: 当前值 {current_value:.4f}, 限制 {limit.threshold:.4f}"
        
        # 建议动作
        suggested_action = limit.action.value
        
        risk_event = RiskEvent(
            timestamp=timestamp,
            symbol="PORTFOLIO",
            risk_type=limit.risk_type.value,
            severity=severity,
            message=message,
            current_value=Decimal(str(current_value)),
            limit_value=Decimal(str(limit.threshold)),
            suggested_action=suggested_action
        )
        
        self.total_risk_events += 1
        
        return risk_event
    
    def _determine_severity(self, limit: RiskLimit, current_value: float) -> str:
        """确定风险严重程度"""
        # 计算超出程度
        if limit.risk_type in [RiskType.DRAWDOWN, RiskType.LOSS_LIMIT]:
            excess_ratio = abs(current_value) / abs(limit.threshold) if limit.threshold != 0 else 1
        else:
            excess_ratio = current_value / limit.threshold if limit.threshold != 0 else 1
        
        if excess_ratio >= 2.0:
            return "emergency"
        elif excess_ratio >= 1.5:
            return "critical"
        elif excess_ratio >= 1.2:
            return "warning"
        else:
            return "info"
    
    def _execute_risk_action(self, limit: RiskLimit, risk_event: RiskEvent):
        """执行风险响应动作"""
        action = limit.action
        self.actions_taken[action] += 1
        
        if action == RiskAction.MONITOR:
            self.logger.info(f"Risk monitoring: {risk_event.message}")
            
        elif action == RiskAction.WARN:
            self.logger.warning(f"Risk warning: {risk_event.message}")
            self._send_warning(risk_event)
            
        elif action == RiskAction.REDUCE_POSITION:
            self.logger.warning(f"Risk action - Reduce position: {risk_event.message}")
            self._trigger_position_reduction()
            
        elif action == RiskAction.CLOSE_POSITION:
            self.logger.error(f"Risk action - Close position: {risk_event.message}")
            self._trigger_position_closure()
            
        elif action == RiskAction.HALT_TRADING:
            self.logger.error(f"Risk action - Halt trading: {risk_event.message}")
            self._halt_trading()
            
        elif action == RiskAction.EMERGENCY_EXIT:
            self.logger.critical(f"Risk action - Emergency exit: {risk_event.message}")
            self._trigger_emergency_exit()
            
        elif action == RiskAction.SYSTEM_SHUTDOWN:
            self.logger.critical(f"Risk action - System shutdown: {risk_event.message}")
            self._trigger_system_shutdown()
        
        # 执行回调函数
        risk_level = RiskLevel(risk_event.severity)
        for callback in self.risk_callbacks.get(risk_level, []):
            try:
                callback(risk_event)
            except Exception as e:
                self.logger.error(f"Error executing risk callback: {e}")
    
    def _send_warning(self, risk_event: RiskEvent):
        """发送警告"""
        self.active_warnings[risk_event.risk_type] = risk_event
        
    def _trigger_position_reduction(self):
        """触发仓位减少"""
        # 这里应该实现仓位减少逻辑
        # 可以通过事件系统通知仓位管理器
        self.logger.info("Position reduction triggered")
        
    def _trigger_position_closure(self):
        """触发仓位关闭"""
        # 这里应该实现仓位关闭逻辑
        self.logger.info("Position closure triggered")
        
    def _halt_trading(self):
        """停止交易"""
        self.trading_halted = True
        self.logger.error("Trading halted due to risk limit breach")
        
    def _trigger_emergency_exit(self):
        """触发紧急退出"""
        self.emergency_mode = True
        self.trading_halted = True
        self.logger.critical("Emergency exit triggered - All positions will be closed")
        
    def _trigger_system_shutdown(self):
        """触发系统停机"""
        self.is_active = False
        self.emergency_mode = True
        self.trading_halted = True
        self.logger.critical("System shutdown triggered due to critical risk breach")
        
    def _record_breach(self, limit: RiskLimit, current_value: float, timestamp: datetime):
        """记录违规历史"""
        breach_record = {
            'timestamp': timestamp.isoformat(),
            'limit_name': limit.name,
            'risk_type': limit.risk_type.value,
            'threshold': limit.threshold,
            'current_value': current_value,
            'action': limit.action.value,
            'breach_count': limit.breach_count
        }
        
        self.breach_history.append(breach_record)
        
        # 限制历史记录长度
        if len(self.breach_history) > 1000:
            self.breach_history = self.breach_history[-500:]
    
    def _validate_order(self, order: OrderEvent) -> bool:
        """验证订单是否符合风险限制"""
        if not self.is_active or self.trading_halted:
            self.logger.warning(f"Order {order.order_id} rejected - trading halted")
            return False
        
        # 检查仓位大小限制
        estimated_position_size = self._estimate_position_size_impact(order)
        if estimated_position_size > self.config.max_position_size:
            self.logger.warning(f"Order {order.order_id} rejected - position size too large")
            return False
        
        # 检查杠杆限制
        estimated_leverage = self._estimate_leverage_impact(order)
        if estimated_leverage > self.config.max_leverage:
            self.logger.warning(f"Order {order.order_id} rejected - leverage too high")
            return False
        
        return True
    
    def _estimate_position_size_impact(self, order: OrderEvent) -> float:
        """估算订单对仓位大小的影响"""
        # 简化实现：假设订单价值
        estimated_value = float(order.quantity) * 50000  # 假设价格
        return estimated_value / float(self.current_capital)
    
    def _estimate_leverage_impact(self, order: OrderEvent) -> float:
        """估算订单对杠杆的影响"""
        # 简化实现
        current_leverage = self.current_metrics.leverage if self.current_metrics else 1.0
        additional_leverage = self._estimate_position_size_impact(order)
        return current_leverage + additional_leverage
    
    # 风险指标计算方法
    def _calculate_total_exposure(self) -> float:
        """计算总敞口"""
        # 简化实现：返回模拟值
        return 0.8  # 80%敞口
    
    def _calculate_leverage(self) -> float:
        """计算杠杆倍数"""
        # 简化实现
        return 2.5
    
    def _calculate_concentration_risk(self) -> float:
        """计算集中度风险"""
        # 简化实现
        return 0.3  # 30%集中度
    
    def _calculate_correlation_risk(self) -> float:
        """计算相关性风险"""
        # 简化实现
        return 0.4  # 40%相关性风险
    
    def _calculate_unrealized_pnl(self) -> float:
        """计算未实现盈亏"""
        # 简化实现
        return 0.02  # 2%未实现盈利
    
    def _calculate_daily_pnl(self, timestamp: datetime) -> float:
        """计算日盈亏"""
        # 简化实现
        return 0.01  # 1%日盈利
    
    def _calculate_max_drawdown(self) -> float:
        """计算最大回撤"""
        if len(self.equity_history) < 2:
            return 0.0
        
        # 计算累计最大值和当前回撤
        peak = max(equity for _, equity in self.equity_history)
        current = self.equity_history[-1][1]
        
        if peak == 0:
            return 0.0
        
        drawdown = float((peak - current) / peak)
        return max(0.0, drawdown)
    
    def _calculate_current_drawdown(self) -> float:
        """计算当前回撤"""
        return self._calculate_max_drawdown()
    
    def _calculate_portfolio_volatility(self) -> float:
        """计算组合波动率"""
        if len(self.pnl_history) < 20:
            return 0.0
        
        # 提取最近的收益率
        recent_pnl = [float(pnl) for _, pnl in self.pnl_history[-60:]]
        returns = np.diff(recent_pnl) / (np.array(recent_pnl[:-1]) + 1e-8)
        
        # 年化波动率
        volatility = np.std(returns) * np.sqrt(365 * 24)  # 假设每小时一个数据点
        return volatility
    
    def _calculate_var(self) -> Tuple[float, float]:
        """计算VaR"""
        if len(self.pnl_history) < 20:
            return 0.0, 0.0
        
        recent_pnl = [float(pnl) for _, pnl in self.pnl_history[-100:]]
        returns = np.diff(recent_pnl) / (np.array(recent_pnl[:-1]) + 1e-8)
        
        var_95 = np.percentile(returns, 5)
        var_99 = np.percentile(returns, 1)
        
        return var_95, var_99
    
    def _calculate_expected_shortfall(self) -> float:
        """计算期望损失"""
        var_95, _ = self._calculate_var()
        
        if len(self.pnl_history) < 20:
            return 0.0
        
        recent_pnl = [float(pnl) for _, pnl in self.pnl_history[-100:]]
        returns = np.diff(recent_pnl) / (np.array(recent_pnl[:-1]) + 1e-8)
        
        # 计算超过VaR的平均损失
        tail_losses = returns[returns <= var_95]
        if len(tail_losses) > 0:
            return np.mean(tail_losses)
        
        return var_95
    
    def _calculate_liquidity_ratio(self) -> float:
        """计算流动性比率"""
        # 简化实现
        return 0.8  # 80%流动性
    
    def _calculate_largest_position_liquidity(self) -> float:
        """计算最大仓位流动性"""
        # 简化实现
        return 0.9  # 90%流动性
    
    def _get_largest_position_size(self) -> float:
        """获取最大仓位大小"""
        # 简化实现
        return 0.15  # 15%最大仓位
    
    def _check_warning_flags(self) -> List[str]:
        """检查预警标识"""
        flags = []
        
        if not self.current_metrics:
            return flags
        
        # 检查各种风险指标
        if self.current_metrics.leverage > self.config.max_leverage * 0.8:
            flags.append("high_leverage")
        
        if self.current_metrics.concentration_risk > self.config.max_concentration * 0.8:
            flags.append("high_concentration")
        
        if abs(self.current_metrics.daily_pnl) > self.config.max_daily_loss * 0.5:
            flags.append("significant_daily_loss")
        
        if self.current_metrics.portfolio_volatility > self.config.max_portfolio_vol * 0.8:
            flags.append("high_volatility")
        
        return flags
    
    # 公共接口方法
    def get_current_risk_status(self) -> Dict[str, Any]:
        """获取当前风险状态"""
        if not self.current_metrics:
            return {}
        
        return {
            'is_active': self.is_active,
            'trading_halted': self.trading_halted,
            'emergency_mode': self.emergency_mode,
            'current_metrics': self.current_metrics.to_dict(),
            'active_warnings': len(self.active_warnings),
            'total_risk_events': self.total_risk_events,
            'actions_taken': self.actions_taken.copy(),
            'enabled_limits': len(self.enabled_limits)
        }
    
    def add_risk_callback(self, risk_level: RiskLevel, callback: Callable):
        """添加风险回调函数"""
        self.risk_callbacks[risk_level].append(callback)
    
    def update_risk_limit(self, limit_name: str, new_threshold: float):
        """更新风险限制"""
        if limit_name in self.enabled_limits:
            self.enabled_limits[limit_name].threshold = new_threshold
            self.logger.info(f"Updated risk limit {limit_name} to {new_threshold}")
    
    def enable_limit(self, limit_name: str):
        """启用风险限制"""
        for limit in self.risk_limits:
            if limit.name == limit_name:
                limit.enabled = True
                self.enabled_limits[limit_name] = limit
                self.logger.info(f"Enabled risk limit {limit_name}")
                break
    
    def disable_limit(self, limit_name: str):
        """禁用风险限制"""
        for limit in self.risk_limits:
            if limit.name == limit_name:
                limit.enabled = False
                if limit_name in self.enabled_limits:
                    del self.enabled_limits[limit_name]
                self.logger.info(f"Disabled risk limit {limit_name}")
                break
    
    def reset_emergency_mode(self):
        """重置紧急模式"""
        if self.emergency_mode:
            self.emergency_mode = False
            self.trading_halted = False
            self.is_active = True
            self.logger.info("Emergency mode reset")
    
    def get_risk_report(self) -> str:
        """生成风险报告"""
        if not self.current_metrics:
            return "No risk metrics available"
        
        report = f"""
=== Risk Management Report ===
Timestamp: {self.current_metrics.timestamp}

System Status:
- Active: {self.is_active}
- Trading Halted: {self.trading_halted}
- Emergency Mode: {self.emergency_mode}

Current Risk Metrics:
- Total Exposure: {self.current_metrics.total_exposure:.2%}
- Leverage: {self.current_metrics.leverage:.2f}x
- Concentration Risk: {self.current_metrics.concentration_risk:.2%}
- Daily P&L: {self.current_metrics.daily_pnl:.2%}
- Current Drawdown: {self.current_metrics.current_drawdown:.2%}
- Portfolio Volatility: {self.current_metrics.portfolio_volatility:.2%}
- VaR 95%: {self.current_metrics.var_95:.2%}

Active Warnings: {len(self.active_warnings)}
Total Risk Events: {self.total_risk_events}

Warning Flags: {', '.join(self.current_metrics.warning_flags) if self.current_metrics.warning_flags else 'None'}

Recent Actions:
"""
        
        for action, count in self.actions_taken.items():
            if count > 0:
                report += f"- {action.value}: {count}\n"
        
        return report
    
    def export_risk_data(self) -> Dict[str, Any]:
        """导出风险数据"""
        return {
            'config': {
                'max_total_exposure': self.config.max_total_exposure,
                'max_leverage': self.config.max_leverage,
                'max_daily_loss': self.config.max_daily_loss,
                'max_drawdown': self.config.max_drawdown,
            },
            'current_status': self.get_current_risk_status(),
            'metrics_history': [metrics.to_dict() for metrics in self.risk_metrics_history[-100:]],
            'breach_history': self.breach_history[-50:],
            'enabled_limits': {
                name: {
                    'threshold': limit.threshold,
                    'action': limit.action.value,
                    'breach_count': limit.breach_count
                }
                for name, limit in self.enabled_limits.items()
            }
        }