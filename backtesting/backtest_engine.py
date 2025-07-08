"""
事件驱动回测引擎
基于CRYPTO_QUANT_ARCHITECTURE.md设计
支持Walk-Forward验证
"""

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import logging
from decimal import Decimal, ROUND_HALF_UP
from dataclasses import dataclass
from enum import Enum
import zipfile

logger = logging.getLogger(__name__)


class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"


class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"


class OrderStatus(Enum):
    PENDING = "pending"
    FILLED = "filled"
    CANCELLED = "cancelled"


@dataclass
class MarketData:
    """市场数据结构"""
    timestamp: datetime
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    quote_volume: float
    count: int
    taker_buy_volume: float
    taker_buy_quote_volume: float


@dataclass
class Order:
    """订单结构"""
    order_id: str
    timestamp: datetime
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: Decimal
    price: Optional[Decimal] = None
    status: OrderStatus = OrderStatus.PENDING
    fill_price: Optional[Decimal] = None
    fill_timestamp: Optional[datetime] = None
    commission: Decimal = Decimal('0')


@dataclass
class Trade:
    """成交记录"""
    trade_id: str
    timestamp: datetime
    symbol: str
    side: OrderSide
    quantity: Decimal
    price: Decimal
    commission: Decimal
    pnl: Decimal = Decimal('0')


@dataclass
class Position:
    """持仓信息"""
    symbol: str
    quantity: Decimal
    avg_price: Decimal
    unrealized_pnl: Decimal = Decimal('0')
    realized_pnl: Decimal = Decimal('0')


class TradingCosts:
    """交易成本模型"""
    
    def __init__(self, 
                 maker_fee: float = 0.0001,  # 0.01% maker费率
                 taker_fee: float = 0.0001,  # 0.01% taker费率
                 slippage: float = 0.0005):  # 0.05% 滑点
        self.maker_fee = Decimal(str(maker_fee))
        self.taker_fee = Decimal(str(taker_fee))
        self.slippage = Decimal(str(slippage))
    
    def calculate_commission(self, quantity: Decimal, price: Decimal, is_maker: bool = False) -> Decimal:
        """计算手续费"""
        fee_rate = self.maker_fee if is_maker else self.taker_fee
        notional = quantity * price
        return (notional * fee_rate).quantize(Decimal('0.00000001'), rounding=ROUND_HALF_UP)
    
    def apply_slippage(self, price: Decimal, side: OrderSide) -> Decimal:
        """应用滑点"""
        if side == OrderSide.BUY:
            # 买入时价格上涨
            return price * (Decimal('1') + self.slippage)
        else:
            # 卖出时价格下跌
            return price * (Decimal('1') - self.slippage)


class DataProvider:
    """历史数据提供器"""
    
    def __init__(self, data_path: str):
        self.data_path = Path(data_path)
        self._data_cache: Dict[str, pd.DataFrame] = {}
    
    def load_data(self, symbol: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """加载指定时间范围的数据"""
        logger.info(f"Loading data for {symbol} from {start_date.date()} to {end_date.date()}")
        
        # 检查缓存
        cache_key = f"{symbol}_{start_date.date()}_{end_date.date()}"
        if cache_key in self._data_cache:
            logger.info(f"Using cached data for {cache_key}")
            return self._data_cache[cache_key]
        
        # 查找数据文件
        klines_path = self.data_path / "klines" / symbol / "1h"
        all_data = []
        
        current_date = start_date.date()
        end_date_only = end_date.date()
        
        while current_date <= end_date_only:
            file_path = klines_path / f"{symbol}-1h-{current_date}.zip"
            
            if file_path.exists():
                try:
                    df = self._parse_klines_file(file_path)
                    if not df.empty:
                        all_data.append(df)
                except Exception as e:
                    logger.warning(f"Failed to parse {file_path}: {e}")
            
            current_date += timedelta(days=1)
        
        if not all_data:
            logger.error(f"No data found for {symbol} in range {start_date.date()} to {end_date.date()}")
            return pd.DataFrame()
        
        # 合并所有数据
        combined_df = pd.concat(all_data, ignore_index=True)
        
        # 过滤时间范围
        mask = (combined_df['timestamp'] >= start_date) & (combined_df['timestamp'] <= end_date)
        filtered_df = combined_df[mask].copy()
        
        # 排序并重置索引
        filtered_df = filtered_df.sort_values('timestamp').reset_index(drop=True)
        
        # 缓存数据
        self._data_cache[cache_key] = filtered_df
        
        logger.info(f"Loaded {len(filtered_df)} records for {symbol}")
        return filtered_df
    
    def _parse_klines_file(self, file_path: Path) -> pd.DataFrame:
        """解析K线数据文件"""
        try:
            with zipfile.ZipFile(file_path, 'r') as zip_file:
                csv_files = [f for f in zip_file.namelist() if f.endswith('.csv')]
                if not csv_files:
                    return pd.DataFrame()
                
                with zip_file.open(csv_files[0]) as csv_file:
                    df = pd.read_csv(csv_file, header=None)
                    
                    # 设置列名
                    df.columns = [
                        'open_time', 'open', 'high', 'low', 'close', 'volume',
                        'close_time', 'quote_volume', 'count', 'taker_buy_volume',
                        'taker_buy_quote_volume', 'ignore'
                    ]
                    
                    # 转换时间戳 (Binance数据通常是毫秒级)
                    try:
                        # 先尝试毫秒级转换
                        df['timestamp'] = pd.to_datetime(df['open_time'], unit='ms')
                    except:
                        try:
                            # 再尝试微秒级转换
                            df['timestamp'] = pd.to_datetime(df['open_time'], unit='us')
                        except:
                            # 最后尝试秒级转换
                            df['timestamp'] = pd.to_datetime(df['open_time'], unit='s')
                    
                    # 转换数值类型
                    numeric_columns = ['open', 'high', 'low', 'close', 'volume', 
                                     'quote_volume', 'taker_buy_volume', 'taker_buy_quote_volume']
                    for col in numeric_columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                    
                    # 选择需要的列
                    df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume', 
                            'quote_volume', 'count', 'taker_buy_volume', 'taker_buy_quote_volume']]
                    
                    return df
                    
        except Exception as e:
            logger.error(f"Error parsing {file_path}: {e}")
            return pd.DataFrame()


class Portfolio:
    """组合管理"""
    
    def __init__(self, initial_capital: float = 100000.0):
        self.initial_capital = Decimal(str(initial_capital))
        self.cash = self.initial_capital
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.orders: List[Order] = []
        self.equity_curve = []
        
    def get_portfolio_value(self, current_prices: Dict[str, float]) -> Decimal:
        """计算组合总价值"""
        total_value = self.cash
        
        for symbol, position in self.positions.items():
            if symbol in current_prices and position.quantity != 0:
                current_price = Decimal(str(current_prices[symbol]))
                position_value = position.quantity * current_price
                total_value += position_value
        
        return total_value
    
    def update_unrealized_pnl(self, current_prices: Dict[str, float]):
        """更新未实现盈亏"""
        for symbol, position in self.positions.items():
            if symbol in current_prices and position.quantity != 0:
                current_price = Decimal(str(current_prices[symbol]))
                position.unrealized_pnl = (current_price - position.avg_price) * position.quantity
    
    def place_order(self, order: Order) -> bool:
        """下单"""
        self.orders.append(order)
        logger.debug(f"Order placed: {order.order_id} {order.side.value} {order.quantity} {order.symbol}")
        return True
    
    def fill_order(self, order: Order, fill_price: Decimal, timestamp: datetime, commission: Decimal):
        """订单成交"""
        order.status = OrderStatus.FILLED
        order.fill_price = fill_price
        order.fill_timestamp = timestamp
        order.commission = commission
        
        # 更新持仓
        self._update_position(order)
        
        # 记录交易
        trade = Trade(
            trade_id=f"trade_{len(self.trades) + 1}",
            timestamp=timestamp,
            symbol=order.symbol,
            side=order.side,
            quantity=order.quantity,
            price=fill_price,
            commission=commission
        )
        self.trades.append(trade)
        
        logger.debug(f"Order filled: {order.order_id} at {fill_price}")
    
    def _update_position(self, order: Order):
        """更新持仓"""
        symbol = order.symbol
        
        if symbol not in self.positions:
            self.positions[symbol] = Position(
                symbol=symbol,
                quantity=Decimal('0'),
                avg_price=Decimal('0')
            )
        
        position = self.positions[symbol]
        fill_value = order.quantity * order.fill_price
        
        if order.side == OrderSide.BUY:
            # 买入
            if position.quantity >= 0:
                # 增加多头仓位
                total_cost = position.quantity * position.avg_price + fill_value
                position.quantity += order.quantity
                position.avg_price = total_cost / position.quantity if position.quantity > 0 else Decimal('0')
            else:
                # 减少空头仓位
                position.quantity += order.quantity
                if position.quantity == 0:
                    position.avg_price = Decimal('0')
            
            self.cash -= fill_value + order.commission
            
        else:  # SELL
            # 卖出
            if position.quantity <= 0:
                # 增加空头仓位
                total_cost = abs(position.quantity) * position.avg_price + fill_value
                position.quantity -= order.quantity
                position.avg_price = total_cost / abs(position.quantity) if position.quantity < 0 else Decimal('0')
            else:
                # 减少多头仓位
                position.quantity -= order.quantity
                if position.quantity == 0:
                    position.avg_price = Decimal('0')
            
            self.cash += fill_value - order.commission


class BacktestEngine:
    """回测引擎"""
    
    def __init__(self, 
                 initial_capital: float = 100000.0,
                 data_path: str = "./data/sample_data"):
        self.portfolio = Portfolio(initial_capital)
        self.data_provider = DataProvider(data_path)
        self.trading_costs = TradingCosts()
        self.current_time: Optional[datetime] = None
        self.current_prices: Dict[str, float] = {}
        
        # 回测结果
        self.performance_metrics = {}
        self.trade_history = []
        
    def run_backtest(self, 
                     strategy,
                     symbol: str,
                     start_date: datetime,
                     end_date: datetime) -> Dict[str, Any]:
        """运行回测"""
        logger.info(f"Starting backtest for {symbol} from {start_date} to {end_date}")
        
        # 加载数据
        data = self.data_provider.load_data(symbol, start_date, end_date)
        if data.empty:
            raise ValueError(f"No data available for {symbol} in specified date range")
        
        # 初始化策略
        strategy.initialize(self)
        
        # 逐行回测
        for idx, row in data.iterrows():
            self.current_time = row['timestamp']
            
            # 更新当前价格
            market_data = MarketData(
                timestamp=row['timestamp'],
                symbol=symbol,
                open=row['open'],
                high=row['high'],
                low=row['low'],
                close=row['close'],
                volume=row['volume'],
                quote_volume=row['quote_volume'],
                count=row['count'],
                taker_buy_volume=row['taker_buy_volume'],
                taker_buy_quote_volume=row['taker_buy_quote_volume']
            )
            
            self.current_prices[symbol] = row['close']
            
            # 处理待执行订单
            self._process_orders(market_data)
            
            # 更新组合
            self.portfolio.update_unrealized_pnl(self.current_prices)
            
            # 策略逻辑
            strategy.on_data(market_data)
            
            # 记录权益曲线
            portfolio_value = self.portfolio.get_portfolio_value(self.current_prices)
            self.portfolio.equity_curve.append({
                'timestamp': self.current_time,
                'portfolio_value': float(portfolio_value),
                'cash': float(self.portfolio.cash),
                'unrealized_pnl': sum(float(pos.unrealized_pnl) for pos in self.portfolio.positions.values())
            })
            
            if idx % 100 == 0:
                logger.debug(f"Processed {idx+1}/{len(data)} records, Portfolio Value: ${portfolio_value:,.2f}")
        
        # 计算性能指标
        self._calculate_performance_metrics()
        
        logger.info(f"Backtest completed. Final portfolio value: ${self.portfolio.get_portfolio_value(self.current_prices):,.2f}")
        
        return {
            'portfolio_value': float(self.portfolio.get_portfolio_value(self.current_prices)),
            'total_return': float((self.portfolio.get_portfolio_value(self.current_prices) - self.portfolio.initial_capital) / self.portfolio.initial_capital),
            'total_trades': len(self.portfolio.trades),
            'equity_curve': self.portfolio.equity_curve,
            'trades': [self._trade_to_dict(trade) for trade in self.portfolio.trades],
            'performance_metrics': self.performance_metrics
        }
    
    def place_market_order(self, symbol: str, side: OrderSide, quantity: float) -> str:
        """下市价单"""
        order_id = f"order_{len(self.portfolio.orders) + 1}"
        order = Order(
            order_id=order_id,
            timestamp=self.current_time,
            symbol=symbol,
            side=side,
            order_type=OrderType.MARKET,
            quantity=Decimal(str(quantity))
        )
        
        self.portfolio.place_order(order)
        return order_id
    
    def _process_orders(self, market_data: MarketData):
        """处理待执行订单"""
        pending_orders = [order for order in self.portfolio.orders if order.status == OrderStatus.PENDING]
        
        for order in pending_orders:
            if order.symbol == market_data.symbol and order.order_type == OrderType.MARKET:
                # 市价单立即成交
                fill_price = Decimal(str(market_data.close))
                
                # 应用滑点
                fill_price = self.trading_costs.apply_slippage(fill_price, order.side)
                
                # 计算手续费
                commission = self.trading_costs.calculate_commission(order.quantity, fill_price, is_maker=False)
                
                # 成交订单
                self.portfolio.fill_order(order, fill_price, market_data.timestamp, commission)
    
    def _calculate_performance_metrics(self):
        """计算性能指标"""
        if not self.portfolio.equity_curve:
            return
        
        equity_df = pd.DataFrame(self.portfolio.equity_curve)
        equity_df.set_index('timestamp', inplace=True)
        
        # 计算收益率
        equity_df['returns'] = equity_df['portfolio_value'].pct_change().fillna(0)
        
        # 总收益率
        total_return = (equity_df['portfolio_value'].iloc[-1] - float(self.portfolio.initial_capital)) / float(self.portfolio.initial_capital)
        
        # 年化收益率 (假设1小时数据)
        total_hours = len(equity_df)
        total_years = total_hours / (365 * 24)
        annual_return = (1 + total_return) ** (1 / total_years) - 1 if total_years > 0 else 0
        
        # 夏普比率
        returns_std = equity_df['returns'].std()
        sharpe_ratio = (equity_df['returns'].mean() / returns_std * np.sqrt(365 * 24)) if returns_std > 0 else 0
        
        # 最大回撤
        peak = equity_df['portfolio_value'].expanding().max()
        drawdown = (equity_df['portfolio_value'] - peak) / peak
        max_drawdown = drawdown.min()
        
        # 胜率
        profitable_trades = sum(1 for trade in self.portfolio.trades if trade.pnl > 0)
        win_rate = profitable_trades / len(self.portfolio.trades) if self.portfolio.trades else 0
        
        self.performance_metrics = {
            'total_return': total_return,
            'annual_return': annual_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'total_trades': len(self.portfolio.trades),
            'total_hours': total_hours
        }
    
    def _trade_to_dict(self, trade: Trade) -> Dict[str, Any]:
        """将Trade对象转换为字典"""
        return {
            'trade_id': trade.trade_id,
            'timestamp': trade.timestamp.isoformat(),
            'symbol': trade.symbol,
            'side': trade.side.value,
            'quantity': float(trade.quantity),
            'price': float(trade.price),
            'commission': float(trade.commission),
            'pnl': float(trade.pnl)
        }


class BaseStrategy:
    """策略基类"""
    
    def __init__(self):
        self.engine: Optional[BacktestEngine] = None
    
    def initialize(self, engine: BacktestEngine):
        """初始化策略"""
        self.engine = engine
    
    def on_data(self, market_data: MarketData):
        """处理市场数据"""
        raise NotImplementedError
    
    def buy(self, symbol: str, quantity: float) -> str:
        """买入"""
        return self.engine.place_market_order(symbol, OrderSide.BUY, quantity)
    
    def sell(self, symbol: str, quantity: float) -> str:
        """卖出"""
        return self.engine.place_market_order(symbol, OrderSide.SELL, quantity)