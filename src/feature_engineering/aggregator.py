"""
Feature aggregator to combine all feature types
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import asyncio
from datetime import datetime
import joblib

from .base import FeatureConfig, MarketMicrostructureCalculator, TechnicalIndicatorCalculator
from .onchain import OnChainFeatureCalculator, SentimentFeatureCalculator
from ..data_pipeline.cache import cache_manager

logger = logging.getLogger(__name__)


class FeatureAggregator:
    """Aggregate features from multiple calculators"""
    
    def __init__(self, config: FeatureConfig = None):
        self.config = config or FeatureConfig()
        
        # Initialize calculators
        self.calculators = {
            'microstructure': MarketMicrostructureCalculator(config),
            'technical': TechnicalIndicatorCalculator(config),
            'onchain': OnChainFeatureCalculator(config),
            'sentiment': SentimentFeatureCalculator(config)
        }
        
        self.feature_stats = {}
        self.feature_importance = {}
        
    def calculate_features(self, data: pd.DataFrame, 
                         feature_types: List[str] = None) -> pd.DataFrame:
        """Calculate all requested features"""
        if feature_types is None:
            feature_types = list(self.calculators.keys())
            
        # Validate data
        if data.empty:
            logger.error("Empty dataframe provided")
            return pd.DataFrame()
            
        # Sort by timestamp
        if 'timestamp' in data.columns:
            data = data.sort_values('timestamp')
            
        # Calculate features
        all_features = []
        
        for feature_type in feature_types:
            if feature_type not in self.calculators:
                logger.warning(f"Unknown feature type: {feature_type}")
                continue
                
            try:
                logger.info(f"Calculating {feature_type} features...")
                calculator = self.calculators[feature_type]
                features = calculator.calculate(data)
                
                if not features.empty:
                    # Add prefix to avoid naming conflicts
                    features = calculator.add_feature_prefix(features, feature_type)
                    all_features.append(features)
                    logger.info(f"Calculated {len(features.columns)} {feature_type} features")
                    
            except Exception as e:
                logger.error(f"Error calculating {feature_type} features: {e}")
                
        # Combine all features
        if all_features:
            combined_features = pd.concat(all_features, axis=1)
            
            # Add derived features
            combined_features = self._add_derived_features(combined_features, data)
            
            # Calculate feature statistics
            self._calculate_feature_stats(combined_features)
            
            return combined_features
        else:
            return pd.DataFrame(index=data.index)
    
    def _add_derived_features(self, features: pd.DataFrame, 
                             original_data: pd.DataFrame) -> pd.DataFrame:
        """Add cross-domain derived features"""
        derived = pd.DataFrame(index=features.index)
        
        # Price momentum with volume confirmation
        if 'technical_returns' in features.columns and 'volume' in original_data.columns:
            volume_ma = original_data['volume'].rolling(20).mean()
            derived['volume_confirmed_momentum'] = (
                features['technical_returns'].rolling(10).mean() * 
                (original_data['volume'] / volume_ma.replace(0, 1))
            )
        
        # Microstructure regime
        if 'microstructure_spread_pct' in features.columns:
            spread_percentile = features['microstructure_spread_pct'].rolling(100).rank(pct=True)
            derived['tight_spread_regime'] = (spread_percentile < 0.3).astype(int)
            derived['wide_spread_regime'] = (spread_percentile > 0.7).astype(int)
        
        # Volatility regime
        if 'technical_volatility_20' in features.columns:
            vol_ma = features['technical_volatility_20'].rolling(60).mean()
            vol_std = features['technical_volatility_20'].rolling(60).std()
            derived['high_vol_regime'] = (
                features['technical_volatility_20'] > vol_ma + vol_std
            ).astype(int)
        
        # Feature interactions
        feature_cols = features.columns.tolist()
        
        # Select top correlated features for interactions
        if len(feature_cols) > 10:
            # Calculate correlation with returns if available
            if 'technical_returns' in features.columns:
                correlations = features.corrwith(features['technical_returns']).abs()
                top_features = correlations.nlargest(10).index.tolist()
                
                # Create polynomial features for top correlates
                for i, feat1 in enumerate(top_features[:5]):
                    for feat2 in top_features[i+1:6]:
                        if feat1 != feat2:
                            derived[f'interact_{feat1}_{feat2}'] = (
                                features[feat1] * features[feat2]
                            )
        
        # Time-based features
        if 'timestamp' in original_data.columns:
            timestamps = pd.to_datetime(original_data['timestamp'], unit='us')
            derived['hour_of_day'] = timestamps.dt.hour
            derived['day_of_week'] = timestamps.dt.dayofweek
            
            # Encode cyclical time features
            derived['hour_sin'] = np.sin(2 * np.pi * derived['hour_of_day'] / 24)
            derived['hour_cos'] = np.cos(2 * np.pi * derived['hour_of_day'] / 24)
            derived['dow_sin'] = np.sin(2 * np.pi * derived['day_of_week'] / 7)
            derived['dow_cos'] = np.cos(2 * np.pi * derived['day_of_week'] / 7)
        
        # Combine with original features
        return pd.concat([features, derived], axis=1)
    
    def _calculate_feature_stats(self, features: pd.DataFrame):
        """Calculate statistics for feature monitoring"""
        self.feature_stats = {
            'mean': features.mean(),
            'std': features.std(),
            'min': features.min(),
            'max': features.max(),
            'missing_pct': features.isna().sum() / len(features) * 100
        }
        
        # Log features with high missing values
        high_missing = self.feature_stats['missing_pct'][
            self.feature_stats['missing_pct'] > 50
        ]
        if not high_missing.empty:
            logger.warning(f"Features with >50% missing values: {high_missing.to_dict()}")
    
    async def calculate_features_async(self, data: pd.DataFrame,
                                     feature_types: List[str] = None) -> pd.DataFrame:
        """Async version for real-time feature calculation"""
        # Check cache first
        cache_key = f"features:{data.index[-1]}"
        cached_features = cache_manager.get_cached_features(cache_key)
        
        if cached_features is not None:
            return pd.DataFrame(cached_features)
            
        # Calculate features
        features = self.calculate_features(data, feature_types)
        
        # Cache results
        if self.config.cache_features and not features.empty:
            cache_manager.cache_features(cache_key, features.to_dict())
            
        return features
    
    def get_feature_names(self, feature_types: List[str] = None) -> List[str]:
        """Get list of all feature names"""
        if feature_types is None:
            feature_types = list(self.calculators.keys())
            
        all_names = []
        for feature_type in feature_types:
            if feature_type in self.calculators:
                calculator = self.calculators[feature_type]
                # This would need to be implemented in each calculator
                # For now, return empty list
                pass
                
        return all_names
    
    def save_feature_pipeline(self, filepath: str):
        """Save feature pipeline configuration"""
        pipeline_config = {
            'config': self.config.__dict__,
            'feature_stats': self.feature_stats,
            'feature_importance': self.feature_importance,
            'timestamp': datetime.now().isoformat()
        }
        
        joblib.dump(pipeline_config, filepath)
        logger.info(f"Feature pipeline saved to {filepath}")
    
    def load_feature_pipeline(self, filepath: str):
        """Load feature pipeline configuration"""
        pipeline_config = joblib.load(filepath)
        
        # Restore config
        for key, value in pipeline_config['config'].items():
            setattr(self.config, key, value)
            
        self.feature_stats = pipeline_config.get('feature_stats', {})
        self.feature_importance = pipeline_config.get('feature_importance', {})
        
        logger.info(f"Feature pipeline loaded from {filepath}")


class FeatureEngineer:
    """High-level feature engineering interface"""
    
    def __init__(self, config: FeatureConfig = None):
        self.aggregator = FeatureAggregator(config)
        self.scaler = None
        self.feature_selector = None
        
    def fit_transform(self, data: pd.DataFrame, target: pd.Series = None) -> pd.DataFrame:
        """Fit feature pipeline and transform data"""
        # Calculate features
        features = self.aggregator.calculate_features(data)
        
        # Feature selection if target provided
        if target is not None and len(features.columns) > 100:
            features = self._select_features(features, target)
            
        # Handle infinities and extreme values
        features = self._handle_extreme_values(features)
        
        return features
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Transform new data using fitted pipeline"""
        features = self.aggregator.calculate_features(data)
        
        # Apply same feature selection
        if self.feature_selector is not None:
            selected_features = self.feature_selector
            features = features[selected_features]
            
        # Handle extreme values
        features = self._handle_extreme_values(features)
        
        return features
    
    def _select_features(self, features: pd.DataFrame, target: pd.Series,
                        n_features: int = 100) -> pd.DataFrame:
        """Select top features based on importance"""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.feature_selection import SelectKBest, f_classif
        
        # Align features and target
        common_idx = features.index.intersection(target.index)
        X = features.loc[common_idx].fillna(0)
        y = target.loc[common_idx]
        
        # Use Random Forest for feature importance
        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X, y)
        
        # Get feature importances
        importances = pd.Series(rf.feature_importances_, index=X.columns)
        top_features = importances.nlargest(n_features).index.tolist()
        
        # Store for later use
        self.feature_selector = top_features
        self.aggregator.feature_importance = importances.to_dict()
        
        logger.info(f"Selected {len(top_features)} features from {len(features.columns)}")
        
        return features[top_features]
    
    def _handle_extreme_values(self, features: pd.DataFrame) -> pd.DataFrame:
        """Handle infinities and extreme values"""
        # Replace infinities with NaN
        features = features.replace([np.inf, -np.inf], np.nan)
        
        # Clip extreme values (beyond 5 standard deviations)
        for col in features.columns:
            if features[col].dtype in [np.float64, np.float32]:
                mean = features[col].mean()
                std = features[col].std()
                
                if std > 0:
                    features[col] = features[col].clip(
                        lower=mean - 5*std,
                        upper=mean + 5*std
                    )
        
        # Fill remaining NaNs
        features = features.fillna(method='ffill').fillna(0)
        
        return features