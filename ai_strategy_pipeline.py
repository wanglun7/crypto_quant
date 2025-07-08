"""
AI-Driven Strategy Pipeline
核心集成文件：连接数据、特征工程、AI模型和策略执行
"""
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from decimal import Decimal
import asyncio
from pathlib import Path

# 导入已有的核心模块
from src.feature_engineering.aggregator import FeatureEngineer, FeatureConfig
from src.models.manager import ModelManager
from src.strategy.signal_generator import SignalGenerator, SignalConfig, FactorSignal, MarketState
from src.backtesting.events import MarketDataEvent, SignalEvent, SignalType
from src.strategy.risk_manager import RiskController, RiskConfig
from src.strategy.position_manager import PositionManager, PositionConfig
from config.settings import settings

logger = logging.getLogger(__name__)


class AIStrategyPipeline:
    """
    AI驱动的策略管道
    
    功能流程：
    1. 接收市场数据 (OHLCV)
    2. 生成330+特征 (FeatureEngineer)
    3. AI模型预测 (ModelManager)
    4. 生成交易信号 (SignalGenerator)
    5. 风险控制和仓位管理
    """
    
    def __init__(self, model_name: str = "crypto_hybrid_v1", 
                 initial_capital: Decimal = Decimal('100000')):
        """
        初始化AI策略管道
        
        Args:
            model_name: 使用的AI模型名称
            initial_capital: 初始资金
        """
        self.model_name = model_name
        self.initial_capital = initial_capital
        
        # 初始化组件
        self._initialize_components()
        
        # 状态跟踪
        self.last_prediction: Optional[np.ndarray] = None
        self.last_features: Optional[pd.DataFrame] = None
        self.prediction_history: List[Dict] = []
        self.signal_history: List[SignalEvent] = []
        
        logger.info(f"AI Strategy Pipeline initialized with model: {model_name}")
    
    def _initialize_components(self):
        """初始化所有组件"""
        # 特征工程
        feature_config = FeatureConfig(
            lookback_periods=[5, 10, 20, 50, 100, 200],
            min_data_points=200,
            parallel_processing=True,
            cache_features=True,
            normalize=True
        )
        self.feature_engineer = FeatureEngineer(feature_config)
        
        # AI模型管理器
        self.model_manager = ModelManager()
        
        # 信号生成器
        signal_config = SignalConfig(
            confidence_threshold=0.6,
            signal_decay_periods=5,
            max_signals_per_hour=10
        )
        self.signal_generator = SignalGenerator(signal_config)
        
        # 风险控制
        risk_config = RiskConfig(
            max_daily_loss=0.03,  # 3%
            max_drawdown=0.15,    # 15%
            max_leverage=3.0,
            max_position_size=0.3  # 30%
        )
        self.risk_controller = RiskController(risk_config, self.initial_capital)
        
        # 仓位管理
        position_config = PositionConfig(
            sizing_method="kelly",
            max_position_size=0.3,
            rebalance_threshold=0.1
        )
        self.position_manager = PositionManager(position_config, self.initial_capital)
        
        logger.info("All components initialized successfully")
    
    def process_market_data(self, market_data: pd.DataFrame) -> Optional[SignalEvent]:
        """
        处理市场数据并生成AI驱动的交易信号
        
        Args:
            market_data: 包含OHLCV的市场数据
            
        Returns:
            SignalEvent or None
        """
        try:
            # 1. 验证数据质量
            if not self._validate_market_data(market_data):
                return None
            
            # 2. 特征工程 - 生成330+特征
            logger.info("Generating features from market data...")
            features = self._generate_features(market_data)
            
            if features.empty:
                logger.warning("No features generated from market data")
                return None
            
            # 3. AI模型预测
            logger.info(f"Running AI prediction with model: {self.model_name}")
            prediction = self._get_ai_prediction(features)
            
            if prediction is None:
                logger.warning("No prediction from AI model")
                return None
            
            # 4. 解释预测结果
            market_signal = self._interpret_prediction(prediction, features)
            
            # 5. 生成交易信号
            signal_event = self._generate_trading_signal(market_signal, market_data.iloc[-1])
            
            # 6. 风险检查
            if signal_event and not self._validate_signal_risk(signal_event):
                logger.warning("Signal rejected by risk controller")
                return None
            
            # 7. 记录历史
            self._record_prediction_history(features, prediction, signal_event)
            
            return signal_event
            
        except Exception as e:
            logger.error(f"Error in AI strategy pipeline: {e}")
            return None
    
    def _validate_market_data(self, data: pd.DataFrame) -> bool:
        """验证市场数据质量"""
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        
        # 检查必需列
        if not all(col in data.columns for col in required_columns):
            logger.error(f"Missing required columns: {set(required_columns) - set(data.columns)}")
            return False
        
        # 检查数据量
        if len(data) < 200:
            logger.error(f"Insufficient data: {len(data)} < 200")
            return False
        
        # 检查价格合理性
        if (data[required_columns[:-1]] <= 0).any().any():
            logger.error("Invalid price data (zero or negative values)")
            return False
        
        return True
    
    def _generate_features(self, market_data: pd.DataFrame) -> pd.DataFrame:
        """使用特征工程生成所有特征"""
        try:
            # 添加时间戳如果没有的话
            if 'timestamp' not in market_data.columns:
                market_data = market_data.copy()
                market_data['timestamp'] = pd.to_datetime(market_data.index)
            
            # 添加微观结构数据 (如果没有的话，使用价格估算)
            if 'bid' not in market_data.columns:
                market_data['bid'] = market_data['close'] * 0.999  # 估算买价
                market_data['ask'] = market_data['close'] * 1.001  # 估算卖价
                market_data['bid_size'] = market_data['volume'] * 0.4
                market_data['ask_size'] = market_data['volume'] * 0.6
                market_data['price'] = market_data['close']
            
            # 使用特征工程器生成所有特征
            features = self.feature_engineer.fit_transform(market_data)
            
            # 存储特征用于分析
            self.last_features = features
            
            logger.info(f"Generated {len(features.columns)} features from market data")
            return features
            
        except Exception as e:
            logger.error(f"Feature generation failed: {e}")
            return pd.DataFrame()
    
    def _get_ai_prediction(self, features: pd.DataFrame) -> Optional[np.ndarray]:
        """使用AI模型进行预测"""
        try:
            # 检查模型是否存在
            if self.model_name not in self.model_manager.models:
                logger.warning(f"Model {self.model_name} not found, creating dummy model")
                return self._create_dummy_prediction(features)
            
            # 使用模型管理器进行预测
            prediction = self.model_manager.predict(
                features, 
                self.model_name, 
                return_probabilities=True
            )
            
            # 存储预测用于分析
            self.last_prediction = prediction
            
            logger.info(f"AI prediction completed: shape={prediction.shape}")
            return prediction
            
        except Exception as e:
            logger.error(f"AI prediction failed: {e}")
            return self._create_dummy_prediction(features)
    
    def _create_dummy_prediction(self, features: pd.DataFrame) -> np.ndarray:
        """创建虚拟预测（用于测试）"""
        logger.warning("Using dummy prediction - no trained model available")
        
        # 基于最近的价格动量创建简单预测
        if 'technical_returns' in features.columns:
            recent_returns = features['technical_returns'].iloc[-10:].mean()
            
            if recent_returns > 0.001:  # 1% threshold
                # 看涨
                prediction = np.array([[0.1, 0.2, 0.7]])  # [sell, hold, buy]
            elif recent_returns < -0.001:
                # 看跌
                prediction = np.array([[0.7, 0.2, 0.1]])  # [sell, hold, buy]
            else:
                # 中性
                prediction = np.array([[0.3, 0.4, 0.3]])  # [sell, hold, buy]
        else:
            # 随机预测
            prediction = np.random.dirichlet([1, 1, 1], 1)
            
        return prediction
    
    def _interpret_prediction(self, prediction: np.ndarray, features: pd.DataFrame) -> Dict[str, Any]:
        """解释AI预测结果"""
        if len(prediction.shape) == 1:
            prediction = prediction.reshape(1, -1)
        
        # 获取最新预测
        latest_pred = prediction[-1]
        
        # 解释预测 (假设是3分类：卖出/持有/买入)
        if len(latest_pred) == 3:
            sell_prob, hold_prob, buy_prob = latest_pred
            
            # 确定主要信号
            max_prob = max(latest_pred)
            predicted_action = np.argmax(latest_pred)
            
            # 计算信号强度
            signal_strength = max_prob - (1.0 / len(latest_pred))  # 超过随机的部分
            
            # 确定信号类型
            if predicted_action == 2:  # 买入
                signal_type = SignalType.LONG
                confidence = buy_prob
            elif predicted_action == 0:  # 卖出
                signal_type = SignalType.SHORT
                confidence = sell_prob
            else:  # 持有
                signal_type = None
                confidence = hold_prob
                
        else:
            # 回归预测 - 转换为分类
            pred_value = latest_pred[0] if len(latest_pred) == 1 else latest_pred
            
            if pred_value > 0.001:
                signal_type = SignalType.LONG
                confidence = min(abs(pred_value) * 10, 1.0)
            elif pred_value < -0.001:
                signal_type = SignalType.SHORT
                confidence = min(abs(pred_value) * 10, 1.0)
            else:
                signal_type = None
                confidence = 0.5
                
            signal_strength = abs(pred_value)
        
        # 构建市场状态
        market_state = self._assess_market_state(features)
        
        return {
            'signal_type': signal_type,
            'confidence': confidence,
            'strength': signal_strength,
            'probabilities': latest_pred.tolist(),
            'market_state': market_state
        }
    
    def _assess_market_state(self, features: pd.DataFrame) -> MarketState:
        """评估当前市场状态"""
        try:
            # 评估市场趋势
            if 'technical_sma_20' in features.columns and 'technical_sma_50' in features.columns:
                sma20 = features['technical_sma_20'].iloc[-1]
                sma50 = features['technical_sma_50'].iloc[-1]
                
                if sma20 > sma50 * 1.02:
                    trend = "bullish"
                elif sma20 < sma50 * 0.98:
                    trend = "bearish"
                else:
                    trend = "sideways"
            else:
                trend = "unknown"
            
            # 评估波动率制度
            if 'technical_volatility_20' in features.columns:
                vol = features['technical_volatility_20'].iloc[-1]
                vol_ma = features['technical_volatility_20'].iloc[-60:].mean()
                
                if vol > vol_ma * 1.5:
                    volatility_regime = "high"
                elif vol < vol_ma * 0.7:
                    volatility_regime = "low"
                else:
                    volatility_regime = "normal"
            else:
                volatility_regime = "unknown"
            
            # 评估流动性
            if 'microstructure_spread_pct' in features.columns:
                spread = features['microstructure_spread_pct'].iloc[-1]
                
                if spread < 0.05:
                    liquidity = "high"
                elif spread > 0.2:
                    liquidity = "low"
                else:
                    liquidity = "normal"
            else:
                liquidity = "unknown"
            
            return MarketState(
                regime=trend,
                volatility=volatility_regime,
                liquidity=liquidity,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Market state assessment failed: {e}")
            return MarketState(
                regime="unknown",
                volatility="unknown", 
                liquidity="unknown",
                timestamp=datetime.now()
            )
    
    def _generate_trading_signal(self, market_signal: Dict, latest_data: pd.Series) -> Optional[SignalEvent]:
        """根据AI预测生成交易信号"""
        signal_type = market_signal.get('signal_type')
        
        if signal_type is None:
            return None
        
        # 创建信号事件
        signal_event = SignalEvent(
            timestamp=datetime.now(),
            symbol="BTCUSDT",
            signal_type=signal_type,
            strength=Decimal(str(market_signal['strength'])),
            confidence=Decimal(str(market_signal['confidence'])),
            strategy_name="AI_Hybrid_Strategy",
            features={
                'ai_probabilities': market_signal['probabilities'],
                'market_state': market_signal['market_state'].__dict__,
                'price': float(latest_data['close']),
                'volume': float(latest_data['volume'])
            }
        )
        
        # 添加到历史记录
        self.signal_history.append(signal_event)
        
        logger.info(f"Generated signal: {signal_type.value} (confidence: {signal_event.confidence:.2f})")
        
        return signal_event
    
    def _validate_signal_risk(self, signal: SignalEvent) -> bool:
        """验证信号的风险"""
        try:
            # 检查信号强度
            if signal.strength < Decimal('0.5'):
                logger.warning(f"Signal strength too low: {signal.strength}")
                return False
            
            # 检查信号置信度
            if signal.confidence < Decimal('0.6'):
                logger.warning(f"Signal confidence too low: {signal.confidence}")
                return False
            
            # 风险控制器检查
            risk_status = self.risk_controller.get_current_risk_status()
            if not risk_status.get('is_active', True):
                logger.warning("Risk controller is not active")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Risk validation failed: {e}")
            return False
    
    def _record_prediction_history(self, features: pd.DataFrame, 
                                 prediction: np.ndarray, signal: Optional[SignalEvent]):
        """记录预测历史用于分析"""
        record = {
            'timestamp': datetime.now(),
            'feature_count': len(features.columns),
            'prediction': prediction.tolist(),
            'signal_generated': signal is not None,
            'signal_type': signal.signal_type.value if signal else None,
            'signal_strength': float(signal.strength) if signal else None
        }
        
        self.prediction_history.append(record)
        
        # 保持历史记录大小
        if len(self.prediction_history) > 1000:
            self.prediction_history = self.prediction_history[-500:]
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        """获取管道状态"""
        return {
            'model_name': self.model_name,
            'models_available': list(self.model_manager.models.keys()),
            'features_generated': len(self.last_features.columns) if self.last_features is not None else 0,
            'predictions_made': len(self.prediction_history),
            'signals_generated': len(self.signal_history),
            'last_prediction': self.last_prediction.tolist() if self.last_prediction is not None else None,
            'risk_status': self.risk_controller.get_current_risk_status()
        }
    
    async def process_realtime_data(self, market_data: pd.DataFrame) -> Optional[SignalEvent]:
        """异步处理实时数据"""
        loop = asyncio.get_event_loop()
        
        # 在线程池中运行处理
        signal = await loop.run_in_executor(
            None, self.process_market_data, market_data
        )
        
        return signal


# 全局实例
ai_pipeline = AIStrategyPipeline()


# 便捷函数
def create_ai_pipeline(model_name: str = "crypto_hybrid_v1", 
                      initial_capital: Decimal = Decimal('100000')) -> AIStrategyPipeline:
    """创建AI策略管道实例"""
    return AIStrategyPipeline(model_name, initial_capital)


def process_market_data_simple(ohlcv_data: pd.DataFrame, 
                              model_name: str = "crypto_hybrid_v1") -> Optional[Dict]:
    """简化的市场数据处理函数"""
    pipeline = create_ai_pipeline(model_name)
    signal = pipeline.process_market_data(ohlcv_data)
    
    if signal:
        return {
            'signal_type': signal.signal_type.value,
            'strength': float(signal.strength),
            'confidence': float(signal.confidence),
            'timestamp': signal.timestamp.isoformat(),
            'features': signal.features
        }
    
    return None


if __name__ == "__main__":
    # 测试AI策略管道
    logging.basicConfig(level=logging.INFO)
    
    # 创建测试数据
    dates = pd.date_range('2024-01-01', periods=500, freq='1H')
    test_data = pd.DataFrame({
        'open': 45000 + np.random.randn(500).cumsum() * 100,
        'high': 45000 + np.random.randn(500).cumsum() * 100 + 50,
        'low': 45000 + np.random.randn(500).cumsum() * 100 - 50,
        'close': 45000 + np.random.randn(500).cumsum() * 100,
        'volume': np.random.uniform(100, 1000, 500)
    }, index=dates)
    
    # 确保OHLC逻辑正确
    test_data['high'] = test_data[['open', 'close', 'high']].max(axis=1)
    test_data['low'] = test_data[['open', 'close', 'low']].min(axis=1)
    
    # 测试管道
    pipeline = create_ai_pipeline()
    signal = pipeline.process_market_data(test_data)
    
    if signal:
        print(f"✅ AI Strategy Pipeline Test Successful!")
        print(f"Signal: {signal.signal_type.value}")
        print(f"Strength: {signal.strength}")
        print(f"Confidence: {signal.confidence}")
        
        # 显示管道状态
        status = pipeline.get_pipeline_status()
        print(f"Features generated: {status['features_generated']}")
        print(f"Models available: {status['models_available']}")
    else:
        print("❌ No signal generated")