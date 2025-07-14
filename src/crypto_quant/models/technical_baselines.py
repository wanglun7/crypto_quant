import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    print("TA-Lib not available, using simplified technical indicators")


class TechnicalIndicatorModel:
    """Base class for technical indicator models."""
    
    def __init__(self, model_name):
        self.model_name = model_name
        self.is_fitted = False
        
    def fit(self, X, y):
        """Fit is not needed for technical indicators."""
        self.is_fitted = True
        return self
        
    def predict(self, X):
        """Make predictions based on technical indicators."""
        probas = self.predict_proba(X)
        return probas.argmax(axis=1)
        
    def predict_proba(self, X):
        """Get prediction probabilities."""
        raise NotImplementedError("Subclass must implement predict_proba")


class RSIModel(TechnicalIndicatorModel):
    """RSI-based trading model."""
    
    def __init__(self, period=14, overbought=70, oversold=30):
        super().__init__("RSI")
        self.period = period
        self.overbought = overbought
        self.oversold = oversold
        
    def predict_proba(self, X):
        """Generate predictions based on RSI signals."""
        # X shape: (N, lookback, features)
        # We need to extract price information from features
        
        # Assume the last feature contains recent price information
        # In practice, we would need to reconstruct price from features
        # For now, use a simple heuristic based on return features
        
        N = X.shape[0]
        probas = np.zeros((N, 3))
        
        for i in range(N):
            # Extract the last few price-related features
            recent_features = X[i, -self.period:, :]
            
            # Use return features to simulate RSI calculation
            # Features 0-2 are ret_1m, ret_5m, ret_15m
            returns = recent_features[:, 0]  # ret_1m
            
            # Simple RSI approximation using return momentum
            if len(returns) > 0:
                momentum = np.sum(returns[-5:])  # 5-period momentum
                
                if momentum > 0.01:  # Strong upward momentum
                    probas[i] = [0.1, 0.2, 0.7]  # Up
                elif momentum < -0.01:  # Strong downward momentum
                    probas[i] = [0.7, 0.2, 0.1]  # Down
                else:
                    probas[i] = [0.25, 0.5, 0.25]  # Hold
            else:
                probas[i] = [0.33, 0.34, 0.33]  # Neutral
                
        return probas


class MACDModel(TechnicalIndicatorModel):
    """MACD-based trading model."""
    
    def __init__(self, fast_period=12, slow_period=26, signal_period=9):
        super().__init__("MACD")
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period
        
    def predict_proba(self, X):
        """Generate predictions based on MACD signals."""
        N = X.shape[0]
        probas = np.zeros((N, 3))
        
        for i in range(N):
            # Extract recent return features
            recent_features = X[i, -self.slow_period:, :]
            
            # Use return features to simulate MACD calculation
            if len(recent_features) >= self.slow_period:
                returns = recent_features[:, 0]  # ret_1m
                
                # Simple MACD approximation using moving averages of returns
                fast_ma = np.mean(returns[-self.fast_period:])
                slow_ma = np.mean(returns[-self.slow_period:])
                macd_line = fast_ma - slow_ma
                
                # Signal line approximation
                signal_line = np.mean(returns[-self.signal_period:])
                
                if macd_line > signal_line and macd_line > 0:
                    probas[i] = [0.1, 0.3, 0.6]  # Up
                elif macd_line < signal_line and macd_line < 0:
                    probas[i] = [0.6, 0.3, 0.1]  # Down
                else:
                    probas[i] = [0.3, 0.4, 0.3]  # Hold
            else:
                probas[i] = [0.33, 0.34, 0.33]  # Neutral
                
        return probas


class BollingerBandModel(TechnicalIndicatorModel):
    """Bollinger Band-based trading model."""
    
    def __init__(self, period=20, std_dev=2):
        super().__init__("BollingerBand")
        self.period = period
        self.std_dev = std_dev
        
    def predict_proba(self, X):
        """Generate predictions based on Bollinger Band signals."""
        N = X.shape[0]
        probas = np.zeros((N, 3))
        
        for i in range(N):
            # Extract recent return features
            recent_features = X[i, -self.period:, :]
            
            if len(recent_features) >= self.period:
                returns = recent_features[:, 0]  # ret_1m
                
                # Calculate Bollinger Bands approximation
                sma = np.mean(returns)
                std = np.std(returns)
                upper_band = sma + self.std_dev * std
                lower_band = sma - self.std_dev * std
                
                current_return = returns[-1]
                
                if current_return > upper_band:
                    probas[i] = [0.6, 0.3, 0.1]  # Down (overbought)
                elif current_return < lower_band:
                    probas[i] = [0.1, 0.3, 0.6]  # Up (oversold)
                else:
                    probas[i] = [0.3, 0.4, 0.3]  # Hold
            else:
                probas[i] = [0.33, 0.34, 0.33]  # Neutral
                
        return probas


class MomentumModel(TechnicalIndicatorModel):
    """Simple momentum-based trading model."""
    
    def __init__(self, period=10):
        super().__init__("Momentum")
        self.period = period
        
    def predict_proba(self, X):
        """Generate predictions based on momentum signals."""
        N = X.shape[0]
        probas = np.zeros((N, 3))
        
        for i in range(N):
            # Extract recent return features
            recent_features = X[i, -self.period:, :]
            
            if len(recent_features) >= self.period:
                # Use different time scale returns
                ret_1m = recent_features[:, 0]  # ret_1m
                ret_5m = recent_features[-1, 1]  # ret_5m (most recent)
                ret_15m = recent_features[-1, 2]  # ret_15m (most recent)
                
                # Momentum score based on multi-timeframe returns
                momentum_score = (
                    0.5 * np.sum(ret_1m[-5:]) +  # 5-period 1m momentum
                    0.3 * ret_5m +               # 5m momentum
                    0.2 * ret_15m                # 15m momentum
                )
                
                if momentum_score > 0.005:  # Strong positive momentum
                    probas[i] = [0.1, 0.2, 0.7]  # Up
                elif momentum_score < -0.005:  # Strong negative momentum
                    probas[i] = [0.7, 0.2, 0.1]  # Down
                else:
                    probas[i] = [0.25, 0.5, 0.25]  # Hold
            else:
                probas[i] = [0.33, 0.34, 0.33]  # Neutral
                
        return probas


class MeanReversionModel(TechnicalIndicatorModel):
    """Mean reversion-based trading model."""
    
    def __init__(self, period=20, threshold=2.0):
        super().__init__("MeanReversion")
        self.period = period
        self.threshold = threshold
        
    def predict_proba(self, X):
        """Generate predictions based on mean reversion signals."""
        N = X.shape[0]
        probas = np.zeros((N, 3))
        
        for i in range(N):
            # Extract recent return features
            recent_features = X[i, -self.period:, :]
            
            if len(recent_features) >= self.period:
                returns = recent_features[:, 0]  # ret_1m
                
                # Mean reversion signal
                mean_return = np.mean(returns)
                std_return = np.std(returns)
                current_return = returns[-1]
                
                # Z-score for mean reversion
                if std_return > 0:
                    z_score = (current_return - mean_return) / std_return
                    
                    if z_score > self.threshold:
                        probas[i] = [0.6, 0.3, 0.1]  # Down (revert from high)
                    elif z_score < -self.threshold:
                        probas[i] = [0.1, 0.3, 0.6]  # Up (revert from low)
                    else:
                        probas[i] = [0.3, 0.4, 0.3]  # Hold
                else:
                    probas[i] = [0.33, 0.34, 0.33]  # Neutral
            else:
                probas[i] = [0.33, 0.34, 0.33]  # Neutral
                
        return probas


class RandomModel(TechnicalIndicatorModel):
    """Random prediction model for baseline comparison."""
    
    def __init__(self, seed=42):
        super().__init__("Random")
        self.seed = seed
        np.random.seed(seed)
        
    def predict_proba(self, X):
        """Generate random predictions."""
        N = X.shape[0]
        
        # Generate random probabilities that sum to 1
        probas = np.random.dirichlet([1, 1, 1], size=N)
        
        return probas


def train_technical_model(X, y, model_type="momentum"):
    """
    Train technical indicator model.
    
    Args:
        X: Feature array (N, lookback, num_features)
        y: Label array (N,)
        model_type: Type of technical model
        
    Returns:
        dict with performance metrics
    """
    # Create model
    if model_type == "rsi":
        model = RSIModel()
    elif model_type == "macd":
        model = MACDModel()
    elif model_type == "bollinger":
        model = BollingerBandModel()
    elif model_type == "momentum":
        model = MomentumModel()
    elif model_type == "mean_reversion":
        model = MeanReversionModel()
    elif model_type == "random":
        model = RandomModel()
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Data split: 70% train, 15% val, 15% test
    n_samples = len(X)
    n_train = int(0.7 * n_samples)
    n_val = int(0.15 * n_samples)
    
    X_train = X[:n_train]
    y_train = y[:n_train]
    X_val = X[n_train:n_train+n_val]
    y_val = y[n_train:n_train+n_val]
    X_test = X[n_train+n_val:]
    y_test = y[n_train+n_val:]
    
    # Fit model (no-op for technical indicators)
    model.fit(X_train, y_train)
    
    # Evaluate
    y_val_pred = model.predict(X_val)
    y_test_pred = model.predict(X_test)
    
    val_bacc = balanced_accuracy_score(y_val, y_val_pred)
    test_bacc = balanced_accuracy_score(y_test, y_test_pred)
    
    print(f"{model.model_name} - Val BACC: {val_bacc:.4f}, Test BACC: {test_bacc:.4f}")
    
    return {
        'val_bacc': val_bacc,
        'test_bacc': test_bacc,
        'model': model
    }


def run_all_technical_models(X, y):
    """Run all technical indicator models and return results."""
    results = {}
    
    print("=" * 50)
    print("Training technical indicator models...")
    print("=" * 50)
    
    models = ["momentum", "mean_reversion", "rsi", "macd", "bollinger", "random"]
    
    for model_type in models:
        try:
            results[model_type] = train_technical_model(X, y, model_type)
        except Exception as e:
            print(f"{model_type} failed: {e}")
            results[model_type] = None
    
    return results