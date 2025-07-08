"""
On-chain metrics feature calculator
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging
from .base import BaseFeatureCalculator, FeatureConfig

logger = logging.getLogger(__name__)


class OnChainFeatureCalculator(BaseFeatureCalculator):
    """Calculate on-chain metrics features"""
    
    def __init__(self, config: FeatureConfig = None):
        super().__init__(config)
        self.required_columns = [
            'active_addresses',
            'transaction_count',
            'transaction_volume',
            'exchange_inflow',
            'exchange_outflow',
            'miner_revenue',
            'hash_rate',
            'difficulty'
        ]
        
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate on-chain features"""
        # For now, check if we have at least price data
        if 'close' not in data.columns:
            logger.warning("No price data available for on-chain features")
            return pd.DataFrame(index=data.index)
            
        features = pd.DataFrame(index=data.index)
        
        # If we have on-chain data, calculate features
        available_cols = set(data.columns) & set(self.required_columns)
        
        if 'active_addresses' in available_cols:
            features = pd.concat([
                features,
                self._calculate_address_features(data)
            ], axis=1)
            
        if 'exchange_inflow' in available_cols and 'exchange_outflow' in available_cols:
            features = pd.concat([
                features,
                self._calculate_exchange_flow_features(data)
            ], axis=1)
            
        if 'hash_rate' in available_cols:
            features = pd.concat([
                features,
                self._calculate_mining_features(data)
            ], axis=1)
            
        # Network value metrics (if we have price)
        if 'close' in data.columns:
            features = pd.concat([
                features,
                self._calculate_value_metrics(data)
            ], axis=1)
            
        return self.normalize_features(features)
    
    def _calculate_address_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate address-based features"""
        features = pd.DataFrame(index=data.index)
        
        # Active address trends
        features['active_addr_change'] = data['active_addresses'].pct_change()
        
        for period in self.config.lookback_periods:
            # Moving averages
            features[f'active_addr_ma_{period}'] = data['active_addresses'].rolling(period).mean()
            
            # Growth rates
            features[f'active_addr_growth_{period}'] = (
                data['active_addresses'] / data['active_addresses'].shift(period) - 1
            )
            
            # Volatility
            features[f'active_addr_vol_{period}'] = (
                data['active_addresses'].pct_change().rolling(period).std()
            )
        
        # Address momentum
        features['addr_momentum'] = (
            features['active_addr_ma_10'] - features['active_addr_ma_50']
        ) / features['active_addr_ma_50'].replace(0, 1)
        
        return features
    
    def _calculate_exchange_flow_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate exchange flow features"""
        features = pd.DataFrame(index=data.index)
        
        # Net flow
        features['exchange_netflow'] = data['exchange_inflow'] - data['exchange_outflow']
        features['exchange_flow_ratio'] = (
            data['exchange_inflow'] / data['exchange_outflow'].replace(0, 1)
        )
        
        # Flow trends
        for period in self.config.lookback_periods:
            # Net flow moving average
            features[f'netflow_ma_{period}'] = features['exchange_netflow'].rolling(period).mean()
            
            # Cumulative flows
            features[f'cum_netflow_{period}'] = features['exchange_netflow'].rolling(period).sum()
            
            # Flow volatility
            features[f'flow_volatility_{period}'] = (
                features['exchange_netflow'].rolling(period).std()
            )
        
        # Exchange pressure indicator
        features['exchange_pressure'] = features['exchange_netflow'] / (
            data['exchange_inflow'] + data['exchange_outflow']
        ).replace(0, 1)
        
        return features
    
    def _calculate_mining_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate mining-related features"""
        features = pd.DataFrame(index=data.index)
        
        # Hash rate trends
        features['hashrate_change'] = data['hash_rate'].pct_change()
        
        for period in self.config.lookback_periods:
            # Hash rate growth
            features[f'hashrate_growth_{period}'] = (
                data['hash_rate'] / data['hash_rate'].shift(period) - 1
            )
            
            # Difficulty adjustment impact
            if 'difficulty' in data.columns:
                features[f'diff_adjusted_hashrate_{period}'] = (
                    data['hash_rate'] / data['difficulty']
                ).rolling(period).mean()
        
        # Miner revenue per hash
        if 'miner_revenue' in data.columns:
            features['revenue_per_hash'] = (
                data['miner_revenue'] / data['hash_rate'].replace(0, 1)
            )
            
            # Miner profitability proxy
            for period in [7, 30]:
                features[f'miner_profit_ma_{period}'] = (
                    features['revenue_per_hash'].rolling(period).mean()
                )
        
        return features
    
    def _calculate_value_metrics(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate network value metrics"""
        features = pd.DataFrame(index=data.index)
        
        # NVT Ratio (Network Value to Transactions)
        if 'transaction_volume' in data.columns:
            # Approximate market cap (would need circulating supply in real implementation)
            approx_mcap = data['close'] * 19_000_000  # Approximate BTC supply
            
            features['nvt_ratio'] = approx_mcap / data['transaction_volume'].replace(0, 1)
            
            # NVT Signal (using 90-day MA of transaction volume)
            features['nvt_signal'] = approx_mcap / (
                data['transaction_volume'].rolling(90).mean().replace(0, 1)
            )
        
        # Price to on-chain activity ratios
        if 'active_addresses' in data.columns:
            features['price_to_addresses'] = (
                data['close'] / data['active_addresses'].replace(0, 1)
            )
            
            # Metcalfe's Law valuation
            features['metcalfe_ratio'] = data['close'] / (
                data['active_addresses'] ** 2 / 1e10
            ).replace(0, 1)
        
        # MVRV (Market Value to Realized Value) proxy
        # In real implementation, would need UTXO data
        features['mvrv_proxy'] = data['close'] / data['close'].rolling(365).mean()
        
        return features


class SentimentFeatureCalculator(BaseFeatureCalculator):
    """Calculate sentiment and social features"""
    
    def __init__(self, config: FeatureConfig = None):
        super().__init__(config)
        self.required_columns = [
            'fear_greed_index',
            'social_volume',
            'social_sentiment',
            'google_trends'
        ]
        
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate sentiment features"""
        features = pd.DataFrame(index=data.index)
        
        # Check available sentiment data
        available_cols = set(data.columns) & set(self.required_columns)
        
        if not available_cols:
            logger.warning("No sentiment data available")
            return features
            
        # Fear & Greed Index features
        if 'fear_greed_index' in available_cols:
            features['fgi'] = data['fear_greed_index']
            features['fgi_change'] = features['fgi'].diff()
            
            # Extreme sentiment indicators
            features['extreme_fear'] = (features['fgi'] < 20).astype(int)
            features['extreme_greed'] = (features['fgi'] > 80).astype(int)
            
            # Moving averages
            for period in [7, 14, 30]:
                features[f'fgi_ma_{period}'] = features['fgi'].rolling(period).mean()
            
            # Sentiment momentum
            features['fgi_momentum'] = features['fgi_ma_7'] - features['fgi_ma_30']
        
        # Social volume features
        if 'social_volume' in available_cols:
            features['social_vol_change'] = data['social_volume'].pct_change()
            
            for period in self.config.lookback_periods:
                features[f'social_vol_ma_{period}'] = (
                    data['social_volume'].rolling(period).mean()
                )
                
                # Volume spikes
                vol_std = data['social_volume'].rolling(period).std()
                vol_mean = data['social_volume'].rolling(period).mean()
                features[f'social_spike_{period}'] = (
                    (data['social_volume'] - vol_mean) / vol_std.replace(0, 1)
                )
        
        # Combined sentiment score
        if 'social_sentiment' in available_cols:
            features['sentiment_score'] = data['social_sentiment']
            
            # Sentiment divergence from price
            if 'close' in data.columns:
                price_trend = data['close'].pct_change().rolling(20).mean()
                sentiment_trend = features['sentiment_score'].rolling(20).mean()
                
                features['sentiment_divergence'] = (
                    sentiment_trend - price_trend * 100  # Scale adjustment
                )
        
        return self.normalize_features(features)