"""
Model management system
"""
import pandas as pd
import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Any
import logging
from pathlib import Path
import joblib
from datetime import datetime
import asyncio
from concurrent.futures import ThreadPoolExecutor

from .trainer import ModelTrainer, ModelConfig, TrainingConfig
from .deep_learning import CryptoHybridModel
from ..feature_engineering import FeatureEngineer, FeatureConfig
from config.settings import settings

logger = logging.getLogger(__name__)


class ModelManager:
    """Manage multiple models and predictions"""
    
    def __init__(self):
        self.models: Dict[str, ModelTrainer] = {}
        self.feature_engineer = FeatureEngineer()
        self.model_metadata: Dict[str, Dict] = {}
        self.prediction_cache: Dict[str, Any] = {}
        
        # Model directory
        self.model_dir = Path(settings.model.model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # Load existing models
        self._load_existing_models()
        
    def _load_existing_models(self):
        """Load previously trained models"""
        for model_file in self.model_dir.glob("*.pth"):
            try:
                model_name = model_file.stem
                trainer = ModelTrainer(ModelConfig(), TrainingConfig())
                metadata = trainer.load_model(model_file)
                
                self.models[model_name] = trainer
                self.model_metadata[model_name] = metadata
                
                logger.info(f"Loaded model: {model_name}")
            except Exception as e:
                logger.error(f"Failed to load model {model_file}: {e}")
    
    def train_model(self, data: pd.DataFrame, model_name: str,
                   model_type: str = "hybrid", 
                   target_column: str = "next_return",
                   **kwargs) -> Dict[str, Any]:
        """Train a new model"""
        logger.info(f"Training model: {model_name}")
        
        # Prepare features
        logger.info("Engineering features...")
        features = self.feature_engineer.fit_transform(data)
        
        # Prepare target
        if target_column not in data.columns:
            # Create return target
            target = data['close'].pct_change().shift(-1)  # Next period return
            target = target.dropna()
        else:
            target = data[target_column].dropna()
        
        # Align features and target
        common_idx = features.index.intersection(target.index)
        features = features.loc[common_idx]
        target = target.loc[common_idx]
        
        logger.info(f"Training data shape: {features.shape}")
        logger.info(f"Target shape: {target.shape}")
        
        # Create configs
        model_config = ModelConfig(
            feature_dim=features.shape[1],
            **kwargs.get('model_config', {})
        )
        
        training_config = TrainingConfig(
            **kwargs.get('training_config', {})
        )
        
        # Create trainer
        trainer = ModelTrainer(model_config, training_config)
        
        # Train model
        results = trainer.train(features, target, model_type)
        
        # Save model
        model_path = self.model_dir / f"{model_name}.pth"
        metadata = {
            'created_at': datetime.now().isoformat(),
            'model_type': model_type,
            'feature_count': features.shape[1],
            'training_samples': len(features),
            'performance': results
        }
        trainer.save_model(model_path, metadata)
        
        # Store in memory
        self.models[model_name] = trainer
        self.model_metadata[model_name] = metadata
        
        logger.info(f"Model {model_name} trained and saved successfully")
        
        return results
    
    def predict(self, data: pd.DataFrame, model_name: str,
                return_probabilities: bool = False) -> np.ndarray:
        """Make predictions with a specific model"""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        # Get model
        trainer = self.models[model_name]
        model = trainer.model
        
        if model is None:
            raise ValueError(f"Model {model_name} not loaded")
        
        # Prepare features
        features = self.feature_engineer.transform(data)
        
        # Convert to sequences
        sequence_length = trainer.model_config.sequence_length
        
        if len(features) < sequence_length:
            raise ValueError(f"Insufficient data: need {sequence_length}, got {len(features)}")
        
        # Create sequences
        sequences = []
        for i in range(len(features) - sequence_length + 1):
            seq = features.iloc[i:i + sequence_length].values
            sequences.append(seq)
        
        # Convert to tensor
        X = torch.FloatTensor(sequences).to(trainer.device)
        
        # Predict
        model.eval()
        with torch.no_grad():
            if isinstance(model, CryptoHybridModel):
                outputs, attention_weights = model(X)
            else:
                outputs, attention_weights = model(X)
        
        # Process outputs
        if trainer.model_config.output_type == "classification":
            if return_probabilities:
                probabilities = torch.softmax(outputs, dim=1)
                predictions = probabilities.cpu().numpy()
            else:
                predictions = torch.argmax(outputs, dim=1).cpu().numpy()
        else:
            predictions = outputs.cpu().numpy().flatten()
        
        return predictions
    
    def predict_ensemble(self, data: pd.DataFrame, model_names: List[str],
                        weights: Optional[List[float]] = None) -> np.ndarray:
        """Make ensemble predictions"""
        if not model_names:
            raise ValueError("No models specified for ensemble")
        
        # Equal weights if not provided
        if weights is None:
            weights = [1.0 / len(model_names)] * len(model_names)
        
        if len(weights) != len(model_names):
            raise ValueError("Weights must match number of models")
        
        # Get predictions from each model
        all_predictions = []
        for model_name in model_names:
            pred = self.predict(data, model_name, return_probabilities=True)
            all_predictions.append(pred)
        
        # Weighted ensemble
        ensemble_pred = np.zeros_like(all_predictions[0])
        for pred, weight in zip(all_predictions, weights):
            ensemble_pred += weight * pred
        
        # Convert back to class predictions if classification
        if len(ensemble_pred.shape) > 1 and ensemble_pred.shape[1] > 1:
            final_predictions = np.argmax(ensemble_pred, axis=1)
        else:
            final_predictions = ensemble_pred
        
        return final_predictions
    
    async def predict_async(self, data: pd.DataFrame, model_name: str) -> np.ndarray:
        """Async prediction for real-time use"""
        loop = asyncio.get_event_loop()
        
        # Run prediction in thread pool to avoid blocking
        with ThreadPoolExecutor() as executor:
            future = loop.run_in_executor(
                executor, self.predict, data, model_name
            )
            predictions = await future
        
        return predictions
    
    def get_model_performance(self, model_name: str) -> Dict[str, Any]:
        """Get model performance metrics"""
        if model_name not in self.model_metadata:
            raise ValueError(f"Model {model_name} not found")
        
        return self.model_metadata[model_name].get('performance', {})
    
    def list_models(self) -> List[Dict[str, Any]]:
        """List all available models"""
        models_info = []
        
        for name, metadata in self.model_metadata.items():
            info = {
                'name': name,
                'type': metadata.get('model_type', 'unknown'),
                'created_at': metadata.get('created_at', 'unknown'),
                'feature_count': metadata.get('feature_count', 0),
                'training_samples': metadata.get('training_samples', 0)
            }
            
            # Add performance metrics
            performance = metadata.get('performance', {})
            if 'test_metrics' in performance:
                info.update(performance['test_metrics'])
            
            models_info.append(info)
        
        return models_info
    
    def delete_model(self, model_name: str):
        """Delete a model"""
        if model_name in self.models:
            del self.models[model_name]
        
        if model_name in self.model_metadata:
            del self.model_metadata[model_name]
        
        # Delete file
        model_path = self.model_dir / f"{model_name}.pth"
        if model_path.exists():
            model_path.unlink()
        
        logger.info(f"Model {model_name} deleted")
    
    def retrain_model(self, model_name: str, new_data: pd.DataFrame) -> Dict[str, Any]:
        """Retrain existing model with new data"""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        # Get existing metadata
        metadata = self.model_metadata[model_name]
        model_type = metadata.get('model_type', 'hybrid')
        
        # Retrain
        results = self.train_model(new_data, model_name, model_type)
        
        logger.info(f"Model {model_name} retrained successfully")
        
        return results
    
    def evaluate_model(self, model_name: str, test_data: pd.DataFrame,
                      test_target: pd.Series) -> Dict[str, float]:
        """Evaluate model on test data"""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        # Make predictions
        predictions = self.predict(test_data, model_name)
        
        # Calculate metrics
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        trainer = self.models[model_name]
        
        if trainer.model_config.output_type == "classification":
            # Align predictions with targets
            min_len = min(len(predictions), len(test_target))
            pred_aligned = predictions[:min_len]
            target_aligned = test_target.values[:min_len]
            
            # Convert target to classes if needed
            if not np.issubdtype(target_aligned.dtype, np.integer):
                target_aligned = np.where(target_aligned > 0.001, 2, 
                                        np.where(target_aligned < -0.001, 0, 1))
            
            metrics = {
                'accuracy': accuracy_score(target_aligned, pred_aligned),
                'precision': precision_score(target_aligned, pred_aligned, average='weighted', zero_division=0),
                'recall': recall_score(target_aligned, pred_aligned, average='weighted', zero_division=0),
                'f1': f1_score(target_aligned, pred_aligned, average='weighted', zero_division=0)
            }
        else:
            # Align predictions with targets
            min_len = min(len(predictions), len(test_target))
            pred_aligned = predictions[:min_len]
            target_aligned = test_target.values[:min_len]
            
            metrics = {
                'mse': mean_squared_error(target_aligned, pred_aligned),
                'mae': mean_absolute_error(target_aligned, pred_aligned),
                'r2': r2_score(target_aligned, pred_aligned)
            }
        
        return metrics
    
    def get_feature_importance(self, model_name: str) -> Dict[str, float]:
        """Get feature importance from model"""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        # This would require implementing attention-based feature importance
        # For now, return feature importance from feature engineer
        return self.feature_engineer.aggregator.feature_importance
    
    def export_model_config(self, model_name: str) -> Dict[str, Any]:
        """Export model configuration"""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        trainer = self.models[model_name]
        
        config = {
            'model_config': trainer.model_config.__dict__,
            'training_config': trainer.training_config.__dict__,
            'metadata': self.model_metadata[model_name]
        }
        
        return config


# Global model manager instance
model_manager = ModelManager()