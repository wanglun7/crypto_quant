"""
性能指标计算模块 - 专业的回测性能评估

该模块实现了全面的回测性能指标计算，包括：
- 传统风险调整收益指标（Sharpe, Sortino, Calmar等）
- 最大回撤分析（时间序列和水下曲线）
- 交易统计分析（胜率、盈亏比、持仓时间等）
- 风险度量（VaR, CVaR, Beta等）
- 绩效归因分析
- 滚动窗口分析
"""

import logging
from datetime import datetime, timedelta
from decimal import Decimal, getcontext
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from enum import Enum
import warnings

# 设置Decimal精度
getcontext().prec = 28

logger = logging.getLogger(__name__)


class MetricFrequency(Enum):
    """指标计算频率"""
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    YEARLY = "yearly"


@dataclass
class TradingStatistics:
    """交易统计数据"""
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    
    total_pnl: Decimal = Decimal('0')
    gross_profit: Decimal = Decimal('0')
    gross_loss: Decimal = Decimal('0')
    
    largest_win: Decimal = Decimal('0')
    largest_loss: Decimal = Decimal('0')
    
    avg_win: Decimal = Decimal('0')
    avg_loss: Decimal = Decimal('0')
    
    avg_trade_duration: timedelta = field(default_factory=lambda: timedelta(0))
    avg_winning_duration: timedelta = field(default_factory=lambda: timedelta(0))
    avg_losing_duration: timedelta = field(default_factory=lambda: timedelta(0))
    
    consecutive_wins: int = 0
    consecutive_losses: int = 0
    max_consecutive_wins: int = 0
    max_consecutive_losses: int = 0
    
    @property
    def win_rate(self) -> float:
        """胜率"""
        return self.winning_trades / max(self.total_trades, 1)
    
    @property
    def loss_rate(self) -> float:
        """败率"""
        return self.losing_trades / max(self.total_trades, 1)
    
    @property
    def profit_factor(self) -> float:
        """盈利因子"""
        return float(self.gross_profit / max(abs(self.gross_loss), Decimal('0.01')))
    
    @property
    def payoff_ratio(self) -> float:
        """盈亏比"""
        return float(self.avg_win / max(abs(self.avg_loss), Decimal('0.01')))
    
    @property
    def expectancy(self) -> float:
        """期望值"""
        return float((self.avg_win * self.win_rate) + (self.avg_loss * self.loss_rate))


@dataclass
class DrawdownPeriod:
    """回撤期间数据"""
    start_date: datetime
    end_date: Optional[datetime]
    valley_date: datetime
    
    peak_value: Decimal
    valley_value: Decimal
    recovery_value: Optional[Decimal]
    
    duration: timedelta
    recovery_duration: Optional[timedelta]
    
    max_drawdown: Decimal
    max_drawdown_pct: float
    
    is_recovered: bool = False
    
    @property
    def duration_days(self) -> int:
        """回撤持续天数"""
        return self.duration.days
    
    @property
    def recovery_duration_days(self) -> Optional[int]:
        """恢复持续天数"""
        return self.recovery_duration.days if self.recovery_duration else None


@dataclass
class RiskMetrics:
    """风险指标"""
    var_95: float = 0.0          # 95% VaR
    var_99: float = 0.0          # 99% VaR
    cvar_95: float = 0.0         # 95% CVaR (Expected Shortfall)
    cvar_99: float = 0.0         # 99% CVaR
    
    beta: float = 1.0            # Beta系数
    alpha: float = 0.0           # Alpha系数
    correlation: float = 0.0      # 与基准的相关性
    
    tracking_error: float = 0.0   # 跟踪误差
    information_ratio: float = 0.0  # 信息比率
    
    skewness: float = 0.0        # 偏度
    kurtosis: float = 0.0        # 峰度
    
    tail_ratio: float = 0.0      # 尾部比率
    
    # 下行风险指标
    downside_deviation: float = 0.0
    semi_variance: float = 0.0
    
    # 最大回撤相关
    max_drawdown: float = 0.0
    max_drawdown_duration: int = 0
    
    # 波动率相关
    volatility: float = 0.0
    downside_volatility: float = 0.0
    upside_volatility: float = 0.0


@dataclass
class PerformanceReport:
    """性能报告"""
    # 基本信息
    strategy_name: str
    start_date: datetime
    end_date: datetime
    duration: timedelta
    
    # 收益指标
    total_return: float
    annualized_return: float
    cumulative_return: float
    
    # 风险调整收益指标
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    sterling_ratio: float
    burke_ratio: float
    
    # 交易统计
    trading_stats: TradingStatistics
    
    # 风险指标
    risk_metrics: RiskMetrics
    
    # 回撤分析
    drawdown_periods: List[DrawdownPeriod]
    
    # 月度收益
    monthly_returns: pd.Series
    
    # 滚动指标
    rolling_sharpe: Optional[pd.Series] = None
    rolling_volatility: Optional[pd.Series] = None
    rolling_max_drawdown: Optional[pd.Series] = None
    
    # 基准比较
    benchmark_comparison: Optional[Dict[str, float]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'strategy_name': self.strategy_name,
            'start_date': self.start_date.isoformat(),
            'end_date': self.end_date.isoformat(),
            'duration_days': self.duration.days,
            'total_return': self.total_return,
            'annualized_return': self.annualized_return,
            'cumulative_return': self.cumulative_return,
            'sharpe_ratio': self.sharpe_ratio,
            'sortino_ratio': self.sortino_ratio,
            'calmar_ratio': self.calmar_ratio,
            'sterling_ratio': self.sterling_ratio,
            'burke_ratio': self.burke_ratio,
            'trading_stats': {
                'total_trades': self.trading_stats.total_trades,
                'win_rate': self.trading_stats.win_rate,
                'profit_factor': self.trading_stats.profit_factor,
                'payoff_ratio': self.trading_stats.payoff_ratio,
                'expectancy': self.trading_stats.expectancy,
                'max_consecutive_wins': self.trading_stats.max_consecutive_wins,
                'max_consecutive_losses': self.trading_stats.max_consecutive_losses,
            },
            'risk_metrics': {
                'max_drawdown': self.risk_metrics.max_drawdown,
                'volatility': self.risk_metrics.volatility,
                'var_95': self.risk_metrics.var_95,
                'var_99': self.risk_metrics.var_99,
                'cvar_95': self.risk_metrics.cvar_95,
                'cvar_99': self.risk_metrics.cvar_99,
                'skewness': self.risk_metrics.skewness,
                'kurtosis': self.risk_metrics.kurtosis,
            },
            'drawdown_analysis': {
                'max_drawdown_duration': max([dp.duration_days for dp in self.drawdown_periods] + [0]),
                'num_drawdown_periods': len(self.drawdown_periods),
                'avg_drawdown_duration': np.mean([dp.duration_days for dp in self.drawdown_periods]) if self.drawdown_periods else 0,
                'avg_recovery_duration': np.mean([dp.recovery_duration_days for dp in self.drawdown_periods if dp.recovery_duration_days]) if self.drawdown_periods else 0,
            },
            'monthly_returns': self.monthly_returns.to_dict() if self.monthly_returns is not None else {},
            'benchmark_comparison': self.benchmark_comparison or {}
        }


class PerformanceMetrics:
    """性能指标计算器"""
    
    def __init__(self, 
                 risk_free_rate: float = 0.02,  # 2%无风险利率
                 benchmark_returns: Optional[pd.Series] = None,
                 trading_days_per_year: int = 365):
        
        self.risk_free_rate = risk_free_rate
        self.benchmark_returns = benchmark_returns
        self.trading_days_per_year = trading_days_per_year
        
        logger.info(f"Performance metrics initialized with risk-free rate: {risk_free_rate}")
    
    def calculate_metrics(self, 
                         equity_curve: pd.Series,
                         initial_capital: float,
                         trades: List[Dict],
                         strategy_name: str = "Strategy") -> PerformanceReport:
        """
        计算完整的性能指标报告
        
        Args:
            equity_curve: 净值曲线
            initial_capital: 初始资本
            trades: 交易记录
            strategy_name: 策略名称
            
        Returns:
            PerformanceReport: 完整的性能报告
        """
        if equity_curve.empty:
            logger.warning("Empty equity curve provided")
            return self._create_empty_report(strategy_name)
        
        # 计算收益率序列
        returns = equity_curve.pct_change().dropna()
        
        # 基本信息
        start_date = equity_curve.index[0]
        end_date = equity_curve.index[-1]
        duration = end_date - start_date
        
        # 收益指标
        total_return = self._calculate_total_return(equity_curve, initial_capital)
        annualized_return = self._calculate_annualized_return(total_return, duration)
        cumulative_return = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1
        
        # 风险调整收益指标
        sharpe_ratio = self._calculate_sharpe_ratio(returns)
        sortino_ratio = self._calculate_sortino_ratio(returns)
        calmar_ratio = self._calculate_calmar_ratio(returns, equity_curve)
        sterling_ratio = self._calculate_sterling_ratio(returns, equity_curve)
        burke_ratio = self._calculate_burke_ratio(returns, equity_curve)
        
        # 交易统计
        trading_stats = self._calculate_trading_statistics(trades)
        
        # 风险指标
        risk_metrics = self._calculate_risk_metrics(returns, equity_curve)
        
        # 回撤分析
        drawdown_periods = self._analyze_drawdown_periods(equity_curve)
        
        # 月度收益
        monthly_returns = self._calculate_monthly_returns(returns)
        
        # 滚动指标
        rolling_sharpe = self._calculate_rolling_sharpe(returns, window=252)
        rolling_volatility = self._calculate_rolling_volatility(returns, window=252)
        rolling_max_drawdown = self._calculate_rolling_max_drawdown(equity_curve, window=252)
        
        # 基准比较
        benchmark_comparison = None
        if self.benchmark_returns is not None:
            benchmark_comparison = self._calculate_benchmark_comparison(returns, self.benchmark_returns)
        
        return PerformanceReport(
            strategy_name=strategy_name,
            start_date=start_date,
            end_date=end_date,
            duration=duration,
            total_return=total_return,
            annualized_return=annualized_return,
            cumulative_return=cumulative_return,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            calmar_ratio=calmar_ratio,
            sterling_ratio=sterling_ratio,
            burke_ratio=burke_ratio,
            trading_stats=trading_stats,
            risk_metrics=risk_metrics,
            drawdown_periods=drawdown_periods,
            monthly_returns=monthly_returns,
            rolling_sharpe=rolling_sharpe,
            rolling_volatility=rolling_volatility,
            rolling_max_drawdown=rolling_max_drawdown,
            benchmark_comparison=benchmark_comparison
        )
    
    def _calculate_total_return(self, equity_curve: pd.Series, initial_capital: float) -> float:
        """计算总收益率"""
        final_value = equity_curve.iloc[-1]
        return (final_value - initial_capital) / initial_capital
    
    def _calculate_annualized_return(self, total_return: float, duration: timedelta) -> float:
        """计算年化收益率"""
        years = duration.days / 365.25
        if years <= 0:
            return 0.0
        return (1 + total_return) ** (1 / years) - 1
    
    def _calculate_sharpe_ratio(self, returns: pd.Series) -> float:
        """计算夏普比率"""
        if returns.empty or returns.std() == 0:
            return 0.0
        
        excess_returns = returns - self.risk_free_rate / self.trading_days_per_year
        return float(excess_returns.mean() / returns.std() * np.sqrt(self.trading_days_per_year))
    
    def _calculate_sortino_ratio(self, returns: pd.Series) -> float:
        """计算索提诺比率"""
        if returns.empty:
            return 0.0
        
        excess_returns = returns - self.risk_free_rate / self.trading_days_per_year
        downside_returns = returns[returns < 0]
        
        if len(downside_returns) == 0:
            return float('inf')
        
        downside_deviation = downside_returns.std()
        if downside_deviation == 0:
            return 0.0
        
        return float(excess_returns.mean() / downside_deviation * np.sqrt(self.trading_days_per_year))
    
    def _calculate_calmar_ratio(self, returns: pd.Series, equity_curve: pd.Series) -> float:
        """计算卡玛比率"""
        if returns.empty:
            return 0.0
        
        annualized_return = returns.mean() * self.trading_days_per_year
        max_drawdown = self._calculate_max_drawdown(equity_curve)
        
        if max_drawdown == 0:
            return float('inf')
        
        return float(annualized_return / max_drawdown)
    
    def _calculate_sterling_ratio(self, returns: pd.Series, equity_curve: pd.Series) -> float:
        """计算斯特林比率"""
        if returns.empty:
            return 0.0
        
        annualized_return = returns.mean() * self.trading_days_per_year
        
        # 计算平均回撤
        drawdown_series = self._calculate_drawdown_series(equity_curve)
        avg_drawdown = abs(drawdown_series.mean())
        
        if avg_drawdown == 0:
            return float('inf')
        
        return float(annualized_return / avg_drawdown)
    
    def _calculate_burke_ratio(self, returns: pd.Series, equity_curve: pd.Series) -> float:
        """计算伯克比率"""
        if returns.empty:
            return 0.0
        
        annualized_return = returns.mean() * self.trading_days_per_year
        
        # 计算回撤平方和的平方根
        drawdown_series = self._calculate_drawdown_series(equity_curve)
        burke_denominator = np.sqrt(np.sum(drawdown_series**2))
        
        if burke_denominator == 0:
            return float('inf')
        
        return float(annualized_return / burke_denominator)
    
    def _calculate_trading_statistics(self, trades: List[Dict]) -> TradingStatistics:
        """计算交易统计"""
        if not trades:
            return TradingStatistics()
        
        stats = TradingStatistics()
        
        # 按交易对分组统计
        positions = {}
        
        for trade in trades:
            symbol = trade['symbol']
            side = trade['side']
            quantity = Decimal(str(trade['quantity']))
            price = Decimal(str(trade['price']))
            timestamp = trade['timestamp']
            
            if symbol not in positions:
                positions[symbol] = {'quantity': Decimal('0'), 'avg_price': Decimal('0'), 'trades': []}
            
            pos = positions[symbol]
            
            if side == 'buy':
                if pos['quantity'] < 0:  # 平空
                    close_quantity = min(abs(pos['quantity']), quantity)
                    pnl = (pos['avg_price'] - price) * close_quantity
                    stats.total_pnl += pnl
                    
                    if pnl > 0:
                        stats.winning_trades += 1
                        stats.gross_profit += pnl
                        stats.largest_win = max(stats.largest_win, pnl)
                        stats.consecutive_wins += 1
                        stats.consecutive_losses = 0
                        stats.max_consecutive_wins = max(stats.max_consecutive_wins, stats.consecutive_wins)
                    else:
                        stats.losing_trades += 1
                        stats.gross_loss += abs(pnl)
                        stats.largest_loss = max(stats.largest_loss, abs(pnl))
                        stats.consecutive_losses += 1
                        stats.consecutive_wins = 0
                        stats.max_consecutive_losses = max(stats.max_consecutive_losses, stats.consecutive_losses)
                
                # 更新持仓
                if pos['quantity'] <= 0:
                    new_quantity = pos['quantity'] + quantity
                    if new_quantity > 0:
                        pos['avg_price'] = price
                    pos['quantity'] = new_quantity
                else:
                    total_cost = pos['quantity'] * pos['avg_price'] + quantity * price
                    pos['quantity'] += quantity
                    pos['avg_price'] = total_cost / pos['quantity']
                    
            else:  # sell
                if pos['quantity'] > 0:  # 平多
                    close_quantity = min(pos['quantity'], quantity)
                    pnl = (price - pos['avg_price']) * close_quantity
                    stats.total_pnl += pnl
                    
                    if pnl > 0:
                        stats.winning_trades += 1
                        stats.gross_profit += pnl
                        stats.largest_win = max(stats.largest_win, pnl)
                        stats.consecutive_wins += 1
                        stats.consecutive_losses = 0
                        stats.max_consecutive_wins = max(stats.max_consecutive_wins, stats.consecutive_wins)
                    else:
                        stats.losing_trades += 1
                        stats.gross_loss += abs(pnl)
                        stats.largest_loss = max(stats.largest_loss, abs(pnl))
                        stats.consecutive_losses += 1
                        stats.consecutive_wins = 0
                        stats.max_consecutive_losses = max(stats.max_consecutive_losses, stats.consecutive_losses)
                
                # 更新持仓
                if pos['quantity'] >= 0:
                    new_quantity = pos['quantity'] - quantity
                    if new_quantity < 0:
                        pos['avg_price'] = price
                    pos['quantity'] = new_quantity
                else:
                    total_cost = abs(pos['quantity']) * pos['avg_price'] + quantity * price
                    pos['quantity'] -= quantity
                    pos['avg_price'] = total_cost / abs(pos['quantity'])
            
            pos['trades'].append({
                'timestamp': timestamp,
                'side': side,
                'quantity': quantity,
                'price': price
            })
        
        # 计算总交易数
        stats.total_trades = stats.winning_trades + stats.losing_trades
        
        # 计算平均盈亏
        if stats.winning_trades > 0:
            stats.avg_win = stats.gross_profit / stats.winning_trades
        if stats.losing_trades > 0:
            stats.avg_loss = stats.gross_loss / stats.losing_trades
        
        return stats
    
    def _calculate_risk_metrics(self, returns: pd.Series, equity_curve: pd.Series) -> RiskMetrics:
        """计算风险指标"""
        if returns.empty:
            return RiskMetrics()
        
        metrics = RiskMetrics()
        
        # 基本统计量
        metrics.volatility = float(returns.std() * np.sqrt(self.trading_days_per_year))
        metrics.skewness = float(returns.skew())
        metrics.kurtosis = float(returns.kurtosis())
        
        # VaR和CVaR
        metrics.var_95 = float(returns.quantile(0.05))
        metrics.var_99 = float(returns.quantile(0.01))
        
        # CVaR (Expected Shortfall)
        var_95_threshold = returns.quantile(0.05)
        var_99_threshold = returns.quantile(0.01)
        
        metrics.cvar_95 = float(returns[returns <= var_95_threshold].mean())
        metrics.cvar_99 = float(returns[returns <= var_99_threshold].mean())
        
        # 最大回撤
        metrics.max_drawdown = float(self._calculate_max_drawdown(equity_curve))
        
        # 下行风险指标
        downside_returns = returns[returns < 0]
        if len(downside_returns) > 0:
            metrics.downside_deviation = float(downside_returns.std())
            metrics.downside_volatility = float(downside_returns.std() * np.sqrt(self.trading_days_per_year))
            metrics.semi_variance = float(downside_returns.var())
        
        # 上行波动率
        upside_returns = returns[returns > 0]
        if len(upside_returns) > 0:
            metrics.upside_volatility = float(upside_returns.std() * np.sqrt(self.trading_days_per_year))
        
        # 尾部比率
        if len(upside_returns) > 0 and len(downside_returns) > 0:
            metrics.tail_ratio = float(upside_returns.quantile(0.95) / abs(downside_returns.quantile(0.05)))
        
        # 基准相关指标
        if self.benchmark_returns is not None:
            aligned_returns, aligned_benchmark = returns.align(self.benchmark_returns, join='inner')
            if len(aligned_returns) > 1:
                metrics.correlation = float(aligned_returns.corr(aligned_benchmark))
                
                # Beta和Alpha
                covariance = aligned_returns.cov(aligned_benchmark)
                benchmark_variance = aligned_benchmark.var()
                if benchmark_variance > 0:
                    metrics.beta = float(covariance / benchmark_variance)
                    metrics.alpha = float(aligned_returns.mean() - metrics.beta * aligned_benchmark.mean())
                
                # 跟踪误差
                tracking_difference = aligned_returns - aligned_benchmark
                metrics.tracking_error = float(tracking_difference.std() * np.sqrt(self.trading_days_per_year))
                
                # 信息比率
                if metrics.tracking_error > 0:
                    metrics.information_ratio = float(tracking_difference.mean() / tracking_difference.std() * np.sqrt(self.trading_days_per_year))
        
        return metrics
    
    def _calculate_max_drawdown(self, equity_curve: pd.Series) -> float:
        """计算最大回撤"""
        if equity_curve.empty:
            return 0.0
        
        cumulative_max = equity_curve.expanding().max()
        drawdown = (equity_curve - cumulative_max) / cumulative_max
        return float(abs(drawdown.min()))
    
    def _calculate_drawdown_series(self, equity_curve: pd.Series) -> pd.Series:
        """计算回撤序列"""
        if equity_curve.empty:
            return pd.Series()
        
        cumulative_max = equity_curve.expanding().max()
        drawdown = (equity_curve - cumulative_max) / cumulative_max
        return drawdown
    
    def _analyze_drawdown_periods(self, equity_curve: pd.Series) -> List[DrawdownPeriod]:
        """分析回撤期间"""
        if equity_curve.empty:
            return []
        
        drawdown_series = self._calculate_drawdown_series(equity_curve)
        periods = []
        
        in_drawdown = False
        start_date = None
        peak_value = None
        valley_value = None
        valley_date = None
        
        for date, dd in drawdown_series.items():
            if not in_drawdown and dd < 0:
                # 开始回撤
                in_drawdown = True
                start_date = date
                peak_value = equity_curve.loc[date] / (1 + dd)
                valley_value = equity_curve.loc[date]
                valley_date = date
            elif in_drawdown:
                if dd < 0:
                    # 继续回撤
                    if equity_curve.loc[date] < valley_value:
                        valley_value = equity_curve.loc[date]
                        valley_date = date
                else:
                    # 回撤结束
                    end_date = date
                    recovery_value = equity_curve.loc[date]
                    
                    duration = valley_date - start_date
                    recovery_duration = end_date - valley_date
                    max_drawdown = (peak_value - valley_value) / peak_value
                    
                    periods.append(DrawdownPeriod(
                        start_date=start_date,
                        end_date=end_date,
                        valley_date=valley_date,
                        peak_value=Decimal(str(peak_value)),
                        valley_value=Decimal(str(valley_value)),
                        recovery_value=Decimal(str(recovery_value)),
                        duration=duration,
                        recovery_duration=recovery_duration,
                        max_drawdown=Decimal(str(max_drawdown)),
                        max_drawdown_pct=float(max_drawdown),
                        is_recovered=True
                    ))
                    
                    in_drawdown = False
        
        # 如果最后还在回撤中
        if in_drawdown:
            duration = valley_date - start_date
            max_drawdown = (peak_value - valley_value) / peak_value
            
            periods.append(DrawdownPeriod(
                start_date=start_date,
                end_date=None,
                valley_date=valley_date,
                peak_value=Decimal(str(peak_value)),
                valley_value=Decimal(str(valley_value)),
                recovery_value=None,
                duration=duration,
                recovery_duration=None,
                max_drawdown=Decimal(str(max_drawdown)),
                max_drawdown_pct=float(max_drawdown),
                is_recovered=False
            ))
        
        return periods
    
    def _calculate_monthly_returns(self, returns: pd.Series) -> pd.Series:
        """计算月度收益"""
        if returns.empty:
            return pd.Series()
        
        try:
            monthly_returns = returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
            return monthly_returns
        except Exception as e:
            logger.warning(f"Error calculating monthly returns: {e}")
            return pd.Series()
    
    def _calculate_rolling_sharpe(self, returns: pd.Series, window: int = 252) -> pd.Series:
        """计算滚动夏普比率"""
        if len(returns) < window:
            return pd.Series()
        
        def rolling_sharpe(x):
            if len(x) < window or x.std() == 0:
                return np.nan
            excess_returns = x - self.risk_free_rate / self.trading_days_per_year
            return excess_returns.mean() / x.std() * np.sqrt(self.trading_days_per_year)
        
        return returns.rolling(window=window).apply(rolling_sharpe)
    
    def _calculate_rolling_volatility(self, returns: pd.Series, window: int = 252) -> pd.Series:
        """计算滚动波动率"""
        if len(returns) < window:
            return pd.Series()
        
        return returns.rolling(window=window).std() * np.sqrt(self.trading_days_per_year)
    
    def _calculate_rolling_max_drawdown(self, equity_curve: pd.Series, window: int = 252) -> pd.Series:
        """计算滚动最大回撤"""
        if len(equity_curve) < window:
            return pd.Series()
        
        def rolling_max_dd(x):
            if len(x) < window:
                return np.nan
            cumulative_max = x.expanding().max()
            drawdown = (x - cumulative_max) / cumulative_max
            return abs(drawdown.min())
        
        return equity_curve.rolling(window=window).apply(rolling_max_dd)
    
    def _calculate_benchmark_comparison(self, returns: pd.Series, benchmark_returns: pd.Series) -> Dict[str, float]:
        """计算与基准的比较"""
        aligned_returns, aligned_benchmark = returns.align(benchmark_returns, join='inner')
        
        if len(aligned_returns) == 0:
            return {}
        
        # 累积收益比较
        strategy_cumulative = (1 + aligned_returns).cumprod().iloc[-1] - 1
        benchmark_cumulative = (1 + aligned_benchmark).cumprod().iloc[-1] - 1
        
        # 年化收益比较
        days = len(aligned_returns)
        years = days / self.trading_days_per_year
        
        strategy_annual = (1 + strategy_cumulative) ** (1/years) - 1 if years > 0 else 0
        benchmark_annual = (1 + benchmark_cumulative) ** (1/years) - 1 if years > 0 else 0
        
        # 波动率比较
        strategy_vol = aligned_returns.std() * np.sqrt(self.trading_days_per_year)
        benchmark_vol = aligned_benchmark.std() * np.sqrt(self.trading_days_per_year)
        
        # 夏普比率比较
        strategy_sharpe = (strategy_annual - self.risk_free_rate) / strategy_vol if strategy_vol > 0 else 0
        benchmark_sharpe = (benchmark_annual - self.risk_free_rate) / benchmark_vol if benchmark_vol > 0 else 0
        
        return {
            'strategy_cumulative_return': float(strategy_cumulative),
            'benchmark_cumulative_return': float(benchmark_cumulative),
            'strategy_annual_return': float(strategy_annual),
            'benchmark_annual_return': float(benchmark_annual),
            'strategy_volatility': float(strategy_vol),
            'benchmark_volatility': float(benchmark_vol),
            'strategy_sharpe': float(strategy_sharpe),
            'benchmark_sharpe': float(benchmark_sharpe),
            'excess_return': float(strategy_annual - benchmark_annual),
            'correlation': float(aligned_returns.corr(aligned_benchmark))
        }
    
    def _create_empty_report(self, strategy_name: str) -> PerformanceReport:
        """创建空的性能报告"""
        return PerformanceReport(
            strategy_name=strategy_name,
            start_date=datetime.now(),
            end_date=datetime.now(),
            duration=timedelta(0),
            total_return=0.0,
            annualized_return=0.0,
            cumulative_return=0.0,
            sharpe_ratio=0.0,
            sortino_ratio=0.0,
            calmar_ratio=0.0,
            sterling_ratio=0.0,
            burke_ratio=0.0,
            trading_stats=TradingStatistics(),
            risk_metrics=RiskMetrics(),
            drawdown_periods=[],
            monthly_returns=pd.Series()
        )
    
    def compare_strategies(self, reports: List[PerformanceReport]) -> pd.DataFrame:
        """比较多个策略的性能"""
        if not reports:
            return pd.DataFrame()
        
        comparison_data = []
        
        for report in reports:
            comparison_data.append({
                'Strategy': report.strategy_name,
                'Total Return': f"{report.total_return:.2%}",
                'Annual Return': f"{report.annualized_return:.2%}",
                'Sharpe Ratio': f"{report.sharpe_ratio:.2f}",
                'Sortino Ratio': f"{report.sortino_ratio:.2f}",
                'Max Drawdown': f"{report.risk_metrics.max_drawdown:.2%}",
                'Volatility': f"{report.risk_metrics.volatility:.2%}",
                'Win Rate': f"{report.trading_stats.win_rate:.2%}",
                'Profit Factor': f"{report.trading_stats.profit_factor:.2f}",
                'Total Trades': report.trading_stats.total_trades,
                'VaR 95%': f"{report.risk_metrics.var_95:.2%}",
                'CVaR 95%': f"{report.risk_metrics.cvar_95:.2%}",
            })
        
        return pd.DataFrame(comparison_data)
    
    def generate_tear_sheet(self, report: PerformanceReport) -> str:
        """生成性能报告撕页"""
        tear_sheet = f"""
=== {report.strategy_name} Performance Report ===

Period: {report.start_date.strftime('%Y-%m-%d')} to {report.end_date.strftime('%Y-%m-%d')}
Duration: {report.duration.days} days

=== RETURNS ===
Total Return: {report.total_return:.2%}
Annualized Return: {report.annualized_return:.2%}
Cumulative Return: {report.cumulative_return:.2%}

=== RISK-ADJUSTED RETURNS ===
Sharpe Ratio: {report.sharpe_ratio:.2f}
Sortino Ratio: {report.sortino_ratio:.2f}
Calmar Ratio: {report.calmar_ratio:.2f}

=== RISK METRICS ===
Max Drawdown: {report.risk_metrics.max_drawdown:.2%}
Volatility: {report.risk_metrics.volatility:.2%}
VaR 95%: {report.risk_metrics.var_95:.2%}
CVaR 95%: {report.risk_metrics.cvar_95:.2%}
Skewness: {report.risk_metrics.skewness:.2f}
Kurtosis: {report.risk_metrics.kurtosis:.2f}

=== TRADING STATISTICS ===
Total Trades: {report.trading_stats.total_trades}
Win Rate: {report.trading_stats.win_rate:.2%}
Profit Factor: {report.trading_stats.profit_factor:.2f}
Payoff Ratio: {report.trading_stats.payoff_ratio:.2f}
Max Consecutive Wins: {report.trading_stats.max_consecutive_wins}
Max Consecutive Losses: {report.trading_stats.max_consecutive_losses}

=== DRAWDOWN ANALYSIS ===
Number of Drawdown Periods: {len(report.drawdown_periods)}
Max Drawdown Duration: {max([dp.duration_days for dp in report.drawdown_periods], default=0)} days
"""
        
        if report.benchmark_comparison:
            tear_sheet += f"""
=== BENCHMARK COMPARISON ===
Strategy Annual Return: {report.benchmark_comparison['strategy_annual_return']:.2%}
Benchmark Annual Return: {report.benchmark_comparison['benchmark_annual_return']:.2%}
Excess Return: {report.benchmark_comparison['excess_return']:.2%}
Correlation: {report.benchmark_comparison['correlation']:.2f}
"""
        
        return tear_sheet