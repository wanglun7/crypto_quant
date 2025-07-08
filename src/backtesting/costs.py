"""
交易成本模型 - 精确模拟实际交易成本

该模块实现了精确的交易成本计算，包括：
- 多层级手续费结构（基于VIP等级）
- 动态滑点模型（基于订单簿深度）
- 资金费率计算
- 市场影响成本
- 时间相关成本（如夜间交易）
"""

import logging
from decimal import Decimal, getcontext
from datetime import datetime, time
from typing import Dict, Optional, NamedTuple
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import pandas as pd

from .events import OrderType, OrderSide

# 设置Decimal精度
getcontext().prec = 28

logger = logging.getLogger(__name__)


class VIPLevel(Enum):
    """VIP等级枚举"""
    REGULAR = "regular"
    VIP1 = "vip1"
    VIP2 = "vip2"
    VIP3 = "vip3"
    VIP4 = "vip4"
    VIP5 = "vip5"
    VIP6 = "vip6"
    VIP7 = "vip7"
    VIP8 = "vip8"
    VIP9 = "vip9"


class TradingCosts(NamedTuple):
    """交易成本结构"""
    commission: Decimal      # 手续费
    slippage: Decimal       # 滑点成本
    funding: Decimal        # 资金费率
    market_impact: Decimal  # 市场影响成本
    total_cost: Decimal     # 总成本
    
    def __str__(self):
        return f"TradingCosts(commission={self.commission}, slippage={self.slippage}, funding={self.funding}, market_impact={self.market_impact}, total={self.total_cost})"


@dataclass
class ExchangeConfig:
    """交易所配置"""
    name: str = "binance"
    
    # 基础手续费率（VIP 0级别）
    maker_fee_rate: Decimal = Decimal('0.0002')  # 0.02% 挂单手续费
    taker_fee_rate: Decimal = Decimal('0.0004')  # 0.04% 吃单手续费
    
    # VIP等级手续费折扣
    vip_discounts: Dict[VIPLevel, Dict[str, Decimal]] = field(default_factory=lambda: {
        VIPLevel.REGULAR: {"maker": Decimal('1.0'), "taker": Decimal('1.0')},
        VIPLevel.VIP1: {"maker": Decimal('0.9'), "taker": Decimal('0.9')},
        VIPLevel.VIP2: {"maker": Decimal('0.8'), "taker": Decimal('0.8')},
        VIPLevel.VIP3: {"maker": Decimal('0.7'), "taker": Decimal('0.7')},
        VIPLevel.VIP4: {"maker": Decimal('0.7'), "taker": Decimal('0.65')},
        VIPLevel.VIP5: {"maker": Decimal('0.6'), "taker": Decimal('0.6')},
        VIPLevel.VIP6: {"maker": Decimal('0.5'), "taker": Decimal('0.55')},
        VIPLevel.VIP7: {"maker": Decimal('0.4'), "taker": Decimal('0.5')},
        VIPLevel.VIP8: {"maker": Decimal('0.3'), "taker": Decimal('0.45')},
        VIPLevel.VIP9: {"maker": Decimal('0.2'), "taker": Decimal('0.4')},
    })
    
    # 最小交易量
    min_quantity: Decimal = Decimal('0.001')
    
    # 价格精度
    price_precision: int = 2
    quantity_precision: int = 6
    
    # 资金费率相关
    funding_interval_hours: int = 8  # 8小时收取一次
    max_funding_rate: Decimal = Decimal('0.0075')  # 0.75%
    min_funding_rate: Decimal = Decimal('-0.0075')  # -0.75%


@dataclass
class SlippageConfig:
    """滑点配置"""
    # 基础滑点（基点）
    base_slippage_bps: int = 2  # 2个基点
    
    # 市场影响参数
    market_impact_coefficient: Decimal = Decimal('0.0001')  # 市场影响系数
    liquidity_adjustment: Decimal = Decimal('1.0')  # 流动性调整因子
    
    # 时间相关滑点
    time_based_adjustment: bool = True
    
    # 波动率调整
    volatility_adjustment: bool = True
    volatility_lookback_periods: int = 100  # 波动率回望期数
    
    # 订单簿深度相关
    use_orderbook_depth: bool = False  # 是否使用真实订单簿深度
    
    # 非线性滑点模型参数
    nonlinear_threshold: Decimal = Decimal('10000')  # 10000 USDT
    nonlinear_exponent: Decimal = Decimal('0.5')  # 平方根模型


class TradingCostModel:
    """交易成本模型"""
    
    def __init__(self, 
                 exchange_config: Optional[ExchangeConfig] = None,
                 slippage_config: Optional[SlippageConfig] = None,
                 vip_level: VIPLevel = VIPLevel.REGULAR):
        
        self.exchange_config = exchange_config or ExchangeConfig()
        self.slippage_config = slippage_config or SlippageConfig()
        self.vip_level = vip_level
        
        # 历史数据用于动态调整
        self.price_history: Dict[str, list] = {}
        self.volume_history: Dict[str, list] = {}
        self.spread_history: Dict[str, list] = {}
        
        # 资金费率历史
        self.funding_rate_history: Dict[str, list] = {}
        
        logger.info(f"Trading cost model initialized with VIP level: {vip_level.value}")
    
    def calculate_costs(self, 
                       symbol: str,
                       side: OrderSide,
                       quantity: Decimal,
                       price: Decimal,
                       order_type: OrderType,
                       timestamp: Optional[datetime] = None) -> TradingCosts:
        """
        计算交易成本
        
        Args:
            symbol: 交易对符号
            side: 交易方向
            quantity: 交易数量
            price: 交易价格
            order_type: 订单类型
            timestamp: 时间戳
            
        Returns:
            TradingCosts: 交易成本结构
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        # 计算手续费
        commission = self._calculate_commission(symbol, side, quantity, price, order_type)
        
        # 计算滑点
        slippage = self._calculate_slippage(symbol, side, quantity, price, order_type, timestamp)
        
        # 计算资金费率
        funding = self._calculate_funding_cost(symbol, side, quantity, price, timestamp)
        
        # 计算市场影响成本
        market_impact = self._calculate_market_impact(symbol, side, quantity, price, timestamp)
        
        # 计算总成本
        total_cost = commission + slippage + funding + market_impact
        
        costs = TradingCosts(
            commission=commission,
            slippage=slippage,
            funding=funding,
            market_impact=market_impact,
            total_cost=total_cost
        )
        
        logger.debug(f"Calculated costs for {symbol}: {costs}")
        return costs
    
    def _calculate_commission(self, 
                            symbol: str,
                            side: OrderSide,
                            quantity: Decimal,
                            price: Decimal,
                            order_type: OrderType) -> Decimal:
        """计算手续费"""
        trade_value = quantity * price
        
        # 确定是挂单还是吃单
        if order_type == OrderType.LIMIT:
            # 限价单通常是挂单（maker）
            base_rate = self.exchange_config.maker_fee_rate
            discount = self.exchange_config.vip_discounts[self.vip_level]["maker"]
        else:
            # 市价单通常是吃单（taker）
            base_rate = self.exchange_config.taker_fee_rate
            discount = self.exchange_config.vip_discounts[self.vip_level]["taker"]
        
        # 计算实际手续费率
        actual_rate = base_rate * discount
        
        # 计算手续费
        commission = trade_value * actual_rate
        
        return commission
    
    def _calculate_slippage(self, 
                          symbol: str,
                          side: OrderSide,
                          quantity: Decimal,
                          price: Decimal,
                          order_type: OrderType,
                          timestamp: datetime) -> Decimal:
        """计算滑点成本"""
        if order_type == OrderType.LIMIT:
            # 限价单理论上没有滑点
            return Decimal('0')
        
        # 基础滑点
        base_slippage = price * (Decimal(str(self.slippage_config.base_slippage_bps)) / Decimal('10000'))
        
        # 交易量调整
        trade_value = quantity * price
        volume_adjustment = self._calculate_volume_adjustment(trade_value)
        
        # 时间调整
        time_adjustment = self._calculate_time_adjustment(timestamp) if self.slippage_config.time_based_adjustment else Decimal('1.0')
        
        # 波动率调整
        volatility_adjustment = self._calculate_volatility_adjustment(symbol) if self.slippage_config.volatility_adjustment else Decimal('1.0')
        
        # 综合滑点
        total_slippage = base_slippage * volume_adjustment * time_adjustment * volatility_adjustment
        
        return total_slippage
    
    def _calculate_funding_cost(self, 
                              symbol: str,
                              side: OrderSide,
                              quantity: Decimal,
                              price: Decimal,
                              timestamp: datetime) -> Decimal:
        """计算资金费率成本"""
        # 检查是否需要支付资金费率
        if not self._is_funding_time(timestamp):
            return Decimal('0')
        
        # 获取当前资金费率
        current_funding_rate = self._get_current_funding_rate(symbol, timestamp)
        
        # 计算持仓价值
        position_value = quantity * price
        
        # 计算资金费率成本
        if side == OrderSide.BUY:
            # 多头支付正资金费率
            funding_cost = position_value * current_funding_rate
        else:
            # 空头支付负资金费率
            funding_cost = -position_value * current_funding_rate
        
        return funding_cost
    
    def _calculate_market_impact(self, 
                               symbol: str,
                               side: OrderSide,
                               quantity: Decimal,
                               price: Decimal,
                               timestamp: datetime) -> Decimal:
        """计算市场影响成本"""
        trade_value = quantity * price
        
        # 线性市场影响
        if trade_value <= self.slippage_config.nonlinear_threshold:
            market_impact = trade_value * self.slippage_config.market_impact_coefficient
        else:
            # 非线性市场影响
            base_impact = self.slippage_config.nonlinear_threshold * self.slippage_config.market_impact_coefficient
            excess_value = trade_value - self.slippage_config.nonlinear_threshold
            excess_impact = excess_value * self.slippage_config.market_impact_coefficient * ((excess_value / self.slippage_config.nonlinear_threshold) ** (self.slippage_config.nonlinear_exponent - 1))
            market_impact = base_impact + excess_impact
        
        # 流动性调整
        market_impact *= self.slippage_config.liquidity_adjustment
        
        return market_impact
    
    def _calculate_volume_adjustment(self, trade_value: Decimal) -> Decimal:
        """计算交易量调整因子"""
        # 平方根模型：交易量越大，滑点越高
        if trade_value <= Decimal('1000'):
            return Decimal('1.0')
        
        # 使用对数模型避免过度惩罚大单
        import math
        log_ratio = math.log(float(trade_value) / 1000)
        adjustment = Decimal(str(1 + log_ratio * 0.1))
        
        return adjustment
    
    def _calculate_time_adjustment(self, timestamp: datetime) -> Decimal:
        """计算时间调整因子"""
        # 市场开盘时间调整
        hour = timestamp.hour
        
        # 亚洲时段（0-8点UTC）流动性较低
        if 0 <= hour <= 8:
            return Decimal('1.2')
        # 欧洲时段（8-16点UTC）流动性中等
        elif 8 <= hour <= 16:
            return Decimal('1.0')
        # 美洲时段（16-24点UTC）流动性较高
        else:
            return Decimal('0.9')
    
    def _calculate_volatility_adjustment(self, symbol: str) -> Decimal:
        """计算波动率调整因子"""
        if symbol not in self.price_history or len(self.price_history[symbol]) < self.slippage_config.volatility_lookback_periods:
            return Decimal('1.0')
        
        # 计算历史波动率
        prices = self.price_history[symbol][-self.slippage_config.volatility_lookback_periods:]
        returns = [float(prices[i]) / float(prices[i-1]) - 1 for i in range(1, len(prices))]
        
        if not returns:
            return Decimal('1.0')
        
        volatility = np.std(returns) * np.sqrt(24)  # 日化波动率
        
        # 波动率调整：波动率越高，滑点越高
        adjustment = Decimal(str(1 + volatility * 0.5))
        
        return adjustment
    
    def _is_funding_time(self, timestamp: datetime) -> bool:
        """检查是否是资金费率收取时间"""
        # Binance永续合约每8小时收取一次：00:00, 08:00, 16:00 UTC
        hour = timestamp.hour
        return hour % self.exchange_config.funding_interval_hours == 0
    
    def _get_current_funding_rate(self, symbol: str, timestamp: datetime) -> Decimal:
        """获取当前资金费率"""
        # 这里可以实现真实的资金费率获取逻辑
        # 现在使用模拟数据
        
        # 基于时间的简单模拟
        import hashlib
        seed = int(hashlib.md5(f"{symbol}{timestamp.hour}".encode()).hexdigest(), 16) % 10000
        
        # 生成-0.1%到0.1%之间的资金费率
        rate = Decimal(str((seed - 5000) / 500000))
        
        # 确保在合理范围内
        rate = max(min(rate, self.exchange_config.max_funding_rate), self.exchange_config.min_funding_rate)
        
        return rate
    
    def update_market_data(self, symbol: str, price: Decimal, volume: Decimal, spread: Optional[Decimal] = None):
        """更新市场数据"""
        # 更新价格历史
        if symbol not in self.price_history:
            self.price_history[symbol] = []
        self.price_history[symbol].append(price)
        
        # 只保留最近的数据
        max_history = self.slippage_config.volatility_lookback_periods * 2
        if len(self.price_history[symbol]) > max_history:
            self.price_history[symbol] = self.price_history[symbol][-max_history:]
        
        # 更新成交量历史
        if symbol not in self.volume_history:
            self.volume_history[symbol] = []
        self.volume_history[symbol].append(volume)
        
        if len(self.volume_history[symbol]) > max_history:
            self.volume_history[symbol] = self.volume_history[symbol][-max_history:]
        
        # 更新价差历史
        if spread is not None:
            if symbol not in self.spread_history:
                self.spread_history[symbol] = []
            self.spread_history[symbol].append(spread)
            
            if len(self.spread_history[symbol]) > max_history:
                self.spread_history[symbol] = self.spread_history[symbol][-max_history:]
    
    def get_cost_breakdown(self, symbol: str) -> Dict[str, Decimal]:
        """获取成本分解"""
        return {
            'base_commission_rate': self.exchange_config.maker_fee_rate,
            'vip_discount': self.exchange_config.vip_discounts[self.vip_level]["maker"],
            'base_slippage_bps': Decimal(str(self.slippage_config.base_slippage_bps)),
            'market_impact_coefficient': self.slippage_config.market_impact_coefficient,
            'current_funding_rate': self._get_current_funding_rate(symbol, datetime.now())
        }
    
    def estimate_total_cost(self, symbol: str, trade_value: Decimal, order_type: OrderType = OrderType.MARKET) -> Decimal:
        """估算总交易成本"""
        # 简化的成本估算
        commission_rate = self.exchange_config.taker_fee_rate if order_type == OrderType.MARKET else self.exchange_config.maker_fee_rate
        vip_discount = self.exchange_config.vip_discounts[self.vip_level]["taker" if order_type == OrderType.MARKET else "maker"]
        
        commission = trade_value * commission_rate * vip_discount
        
        if order_type == OrderType.MARKET:
            slippage = trade_value * (Decimal(str(self.slippage_config.base_slippage_bps)) / Decimal('10000'))
            market_impact = trade_value * self.slippage_config.market_impact_coefficient
            total_cost = commission + slippage + market_impact
        else:
            total_cost = commission
        
        return total_cost


class AdvancedTradingCostModel(TradingCostModel):
    """高级交易成本模型 - 支持更复杂的成本计算"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # 订单簿深度数据
        self.orderbook_data: Dict[str, Dict] = {}
        
        # 交易量分析
        self.volume_profile: Dict[str, Dict] = {}
        
        # 宏观因子
        self.macro_factors: Dict[str, Decimal] = {
            'vix': Decimal('20'),  # 恐慌指数
            'btc_dominance': Decimal('0.45'),  # BTC市值占比
            'funding_rate_trend': Decimal('0'),  # 资金费率趋势
        }
    
    def update_orderbook(self, symbol: str, bids: list, asks: list):
        """更新订单簿数据"""
        self.orderbook_data[symbol] = {
            'bids': bids,
            'asks': asks,
            'bid_depth': sum(qty for price, qty in bids),
            'ask_depth': sum(qty for price, qty in asks),
            'spread': asks[0][0] - bids[0][0] if bids and asks else Decimal('0'),
            'mid_price': (bids[0][0] + asks[0][0]) / 2 if bids and asks else Decimal('0')
        }
    
    def _calculate_slippage(self, symbol: str, side: OrderSide, quantity: Decimal, price: Decimal, order_type: OrderType, timestamp: datetime) -> Decimal:
        """使用订单簿深度计算更精确的滑点"""
        if order_type == OrderType.LIMIT:
            return Decimal('0')
        
        # 如果有订单簿数据，使用精确计算
        if symbol in self.orderbook_data and self.slippage_config.use_orderbook_depth:
            return self._calculate_orderbook_slippage(symbol, side, quantity, price)
        
        # 否则使用基础模型
        return super()._calculate_slippage(symbol, side, quantity, price, order_type, timestamp)
    
    def _calculate_orderbook_slippage(self, symbol: str, side: OrderSide, quantity: Decimal, price: Decimal) -> Decimal:
        """基于订单簿深度计算滑点"""
        orderbook = self.orderbook_data[symbol]
        
        if side == OrderSide.BUY:
            # 买入时遍历卖单
            levels = orderbook['asks']
            total_cost = Decimal('0')
            remaining_qty = quantity
            
            for level_price, level_qty in levels:
                if remaining_qty <= 0:
                    break
                
                fill_qty = min(remaining_qty, level_qty)
                total_cost += fill_qty * level_price
                remaining_qty -= fill_qty
            
            if remaining_qty > 0:
                # 流动性不足，使用价格影响模型
                avg_price = total_cost / (quantity - remaining_qty) if quantity > remaining_qty else price
                price_impact = avg_price * Decimal('0.01')  # 1%价格影响
                total_cost += remaining_qty * (avg_price + price_impact)
            
            expected_cost = quantity * price
            slippage = total_cost - expected_cost
            
        else:
            # 卖出时遍历买单
            levels = orderbook['bids']
            total_value = Decimal('0')
            remaining_qty = quantity
            
            for level_price, level_qty in levels:
                if remaining_qty <= 0:
                    break
                
                fill_qty = min(remaining_qty, level_qty)
                total_value += fill_qty * level_price
                remaining_qty -= fill_qty
            
            if remaining_qty > 0:
                # 流动性不足
                avg_price = total_value / (quantity - remaining_qty) if quantity > remaining_qty else price
                price_impact = avg_price * Decimal('0.01')
                total_value += remaining_qty * (avg_price - price_impact)
            
            expected_value = quantity * price
            slippage = expected_value - total_value
        
        return max(slippage, Decimal('0'))
    
    def update_macro_factors(self, factors: Dict[str, Decimal]):
        """更新宏观因子"""
        self.macro_factors.update(factors)
    
    def get_cost_attribution(self, symbol: str, side: OrderSide, quantity: Decimal, price: Decimal, order_type: OrderType) -> Dict[str, Decimal]:
        """获取成本归因分析"""
        costs = self.calculate_costs(symbol, side, quantity, price, order_type)
        
        return {
            'commission_pct': costs.commission / (quantity * price),
            'slippage_pct': costs.slippage / (quantity * price),
            'market_impact_pct': costs.market_impact / (quantity * price),
            'funding_pct': costs.funding / (quantity * price),
            'total_cost_pct': costs.total_cost / (quantity * price),
            'total_cost_bps': costs.total_cost / (quantity * price) * 10000
        }