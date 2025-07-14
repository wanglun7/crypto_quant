import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import balanced_accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV
import pickle
import joblib
from pathlib import Path

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("XGBoost not available, skipping XGBoost model")


class SklearnModelWrapper:
    """Wrapper for sklearn models to match GRU interface."""
    
    def __init__(self, model, model_name):
        self.model = model
        self.model_name = model_name
        self.is_fitted = False
        self.feature_names = None
        
    def fit(self, X, y):
        """Fit the model on flattened features."""
        # Flatten the sequence data (N, lookback, features) -> (N, lookback*features)
        X_flat = X.reshape(X.shape[0], -1)
        
        # Store feature names for later use
        self.feature_names = [f"feat_{i}" for i in range(X_flat.shape[1])]
        
        print(f"Training {self.model_name} on {X_flat.shape[0]} samples with {X_flat.shape[1]} features")
        
        self.model.fit(X_flat, y)
        self.is_fitted = True
        
        return self
        
    def predict(self, X):
        """Make predictions."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
            
        X_flat = X.reshape(X.shape[0], -1)
        return self.model.predict(X_flat)
        
    def predict_proba(self, X):
        """Get prediction probabilities."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
            
        X_flat = X.reshape(X.shape[0], -1)
        return self.model.predict_proba(X_flat)
        
    def save(self, path):
        """Save the model."""
        joblib.dump({
            'model': self.model,
            'model_name': self.model_name,
            'feature_names': self.feature_names
        }, path)
        
    @classmethod
    def load(cls, path):
        """Load the model."""
        data = joblib.load(path)
        wrapper = cls(data['model'], data['model_name'])
        wrapper.is_fitted = True
        wrapper.feature_names = data['feature_names']
        return wrapper


def train_random_forest(X, y, **kwargs):
    """Train Random Forest model."""
    # Default parameters
    params = {
        'n_estimators': 100,
        'max_depth': 10,
        'min_samples_split': 5,
        'min_samples_leaf': 2,
        'random_state': 42,
        'n_jobs': -1,
        'class_weight': 'balanced'
    }
    params.update(kwargs)
    
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
    
    # Train model
    rf_model = RandomForestClassifier(**params)
    model = SklearnModelWrapper(rf_model, "RandomForest")
    model.fit(X_train, y_train)
    
    # Evaluate
    y_val_pred = model.predict(X_val)
    y_test_pred = model.predict(X_test)
    
    val_bacc = balanced_accuracy_score(y_val, y_val_pred)
    test_bacc = balanced_accuracy_score(y_test, y_test_pred)
    
    print(f"Random Forest - Val BACC: {val_bacc:.4f}, Test BACC: {test_bacc:.4f}")
    
    # Save model
    model.save("random_forest.pkl")
    
    return {
        'val_bacc': val_bacc,
        'test_bacc': test_bacc,
        'model_path': "random_forest.pkl"
    }


def train_logistic_regression(X, y, **kwargs):
    """Train Logistic Regression model."""
    params = {
        'C': 1.0,
        'max_iter': 1000,
        'random_state': 42,
        'class_weight': 'balanced',
        'solver': 'liblinear'
    }
    params.update(kwargs)
    
    # Data split
    n_samples = len(X)
    n_train = int(0.7 * n_samples)
    n_val = int(0.15 * n_samples)
    
    X_train = X[:n_train]
    y_train = y[:n_train]
    X_val = X[n_train:n_train+n_val]
    y_val = y[n_train:n_train+n_val]
    X_test = X[n_train+n_val:]
    y_test = y[n_train+n_val:]
    
    # Train model
    lr_model = LogisticRegression(**params)
    model = SklearnModelWrapper(lr_model, "LogisticRegression")
    model.fit(X_train, y_train)
    
    # Evaluate
    y_val_pred = model.predict(X_val)
    y_test_pred = model.predict(X_test)
    
    val_bacc = balanced_accuracy_score(y_val, y_val_pred)
    test_bacc = balanced_accuracy_score(y_test, y_test_pred)
    
    print(f"Logistic Regression - Val BACC: {val_bacc:.4f}, Test BACC: {test_bacc:.4f}")
    
    # Save model
    model.save("logistic_regression.pkl")
    
    return {
        'val_bacc': val_bacc,
        'test_bacc': test_bacc,
        'model_path': "logistic_regression.pkl"
    }


def train_xgboost(X, y, **kwargs):
    """Train XGBoost model."""
    if not XGBOOST_AVAILABLE:
        print("XGBoost not available, skipping")
        return None
        
    params = {
        'n_estimators': 100,
        'max_depth': 6,
        'learning_rate': 0.1,
        'random_state': 42,
        'n_jobs': -1,
        'eval_metric': 'mlogloss'
    }
    params.update(kwargs)
    
    # Data split
    n_samples = len(X)
    n_train = int(0.7 * n_samples)
    n_val = int(0.15 * n_samples)
    
    X_train = X[:n_train]
    y_train = y[:n_train]
    X_val = X[n_train:n_train+n_val]
    y_val = y[n_train:n_train+n_val]
    X_test = X[n_train+n_val:]
    y_test = y[n_train+n_val:]
    
    # Train model
    xgb_model = xgb.XGBClassifier(**params)
    model = SklearnModelWrapper(xgb_model, "XGBoost")
    model.fit(X_train, y_train)
    
    # Evaluate
    y_val_pred = model.predict(X_val)
    y_test_pred = model.predict(X_test)
    
    val_bacc = balanced_accuracy_score(y_val, y_val_pred)
    test_bacc = balanced_accuracy_score(y_test, y_test_pred)
    
    print(f"XGBoost - Val BACC: {val_bacc:.4f}, Test BACC: {test_bacc:.4f}")
    
    # Save model
    model.save("xgboost.pkl")
    
    return {
        'val_bacc': val_bacc,
        'test_bacc': test_bacc,
        'model_path': "xgboost.pkl"
    }


def train_svm(X, y, **kwargs):
    """Train SVM model."""
    params = {
        'C': 1.0,
        'kernel': 'rbf',
        'gamma': 'scale',
        'random_state': 42,
        'class_weight': 'balanced',
        'probability': True  # Enable probability estimates
    }
    params.update(kwargs)
    
    # Data split
    n_samples = len(X)
    n_train = int(0.7 * n_samples)
    n_val = int(0.15 * n_samples)
    
    X_train = X[:n_train]
    y_train = y[:n_train]
    X_val = X[n_train:n_train+n_val]
    y_val = y[n_train:n_train+n_val]
    X_test = X[n_train+n_val:]
    y_test = y[n_train+n_val:]
    
    # Use smaller subset for SVM due to computational cost
    if len(X_train) > 2000:
        print("Using subset of data for SVM training due to computational constraints")
        subset_idx = np.random.choice(len(X_train), 2000, replace=False)
        X_train = X_train[subset_idx]
        y_train = y_train[subset_idx]
    
    # Train model
    svm_model = SVC(**params)
    model = SklearnModelWrapper(svm_model, "SVM")
    model.fit(X_train, y_train)
    
    # Evaluate
    y_val_pred = model.predict(X_val)
    y_test_pred = model.predict(X_test)
    
    val_bacc = balanced_accuracy_score(y_val, y_val_pred)
    test_bacc = balanced_accuracy_score(y_test, y_test_pred)
    
    print(f"SVM - Val BACC: {val_bacc:.4f}, Test BACC: {test_bacc:.4f}")
    
    # Save model
    model.save("svm.pkl")
    
    return {
        'val_bacc': val_bacc,
        'test_bacc': test_bacc,
        'model_path': "svm.pkl"
    }


def load_sklearn_model(model_path):
    """Load a trained sklearn model."""
    return SklearnModelWrapper.load(model_path)


def run_all_sklearn_models(X, y):
    """Run all sklearn models and return results."""
    results = {}
    
    print("=" * 50)
    print("Training sklearn models...")
    print("=" * 50)
    
    # Random Forest
    try:
        results['random_forest'] = train_random_forest(X, y)
    except Exception as e:
        print(f"Random Forest failed: {e}")
        results['random_forest'] = None
    
    # Logistic Regression
    try:
        results['logistic_regression'] = train_logistic_regression(X, y)
    except Exception as e:
        print(f"Logistic Regression failed: {e}")
        results['logistic_regression'] = None
    
    # XGBoost
    try:
        results['xgboost'] = train_xgboost(X, y)
    except Exception as e:
        print(f"XGBoost failed: {e}")
        results['xgboost'] = None
    
    # SVM (commented out due to computational cost)
    # try:
    #     results['svm'] = train_svm(X, y)
    # except Exception as e:
    #     print(f"SVM failed: {e}")
    #     results['svm'] = None
    
    return results