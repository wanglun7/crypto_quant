"""
Model training pipeline
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import logging
from pathlib import Path
import json
from dataclasses import dataclass, asdict
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
import optuna
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from .deep_learning import (
    CryptoTransformer, CryptoLSTM, CryptoCNN, CryptoHybridModel, ModelConfig
)
from config.settings import settings

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Training configuration"""
    # Training parameters
    max_epochs: int = 100
    early_stopping_patience: int = 10
    validation_split: float = 0.2
    test_split: float = 0.1
    
    # Optimization
    optimizer: str = "adam"  # "adam", "sgd", "adamw"
    learning_rate: float = 0.001
    weight_decay: float = 1e-5
    lr_scheduler: str = "cosine"  # "cosine", "step", "plateau"
    
    # Regularization
    gradient_clip_norm: float = 1.0
    label_smoothing: float = 0.1
    
    # Hardware
    device: str = "auto"  # "auto", "cpu", "cuda"
    num_workers: int = 4
    pin_memory: bool = True
    
    # Logging
    log_interval: int = 100
    save_interval: int = 10
    
    # Hyperparameter tuning
    use_optuna: bool = False
    n_trials: int = 100


class CryptoDataset(Dataset):
    """Dataset for crypto prediction"""
    
    def __init__(self, features: np.ndarray, targets: np.ndarray,
                 sequence_length: int = 100):
        self.features = features
        self.targets = targets
        self.sequence_length = sequence_length
        
        # Validate data
        assert len(features) == len(targets), "Features and targets must have same length"
        assert len(features) >= sequence_length, "Not enough data for sequence length"
        
    def __len__(self) -> int:
        return len(self.features) - self.sequence_length + 1
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Get sequence
        feature_seq = self.features[idx:idx + self.sequence_length]
        target = self.targets[idx + self.sequence_length - 1]  # Predict at end of sequence
        
        return (
            torch.FloatTensor(feature_seq),
            torch.FloatTensor([target]) if isinstance(target, (int, float)) else torch.LongTensor([target])
        )


class ModelTrainer:
    """Comprehensive model trainer"""
    
    def __init__(self, model_config: ModelConfig, training_config: TrainingConfig):
        self.model_config = model_config
        self.training_config = training_config
        
        # Setup device
        if training_config.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(training_config.device)
        
        logger.info(f"Using device: {self.device}")
        
        # Initialize components
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        
        # Tracking
        self.train_losses = []
        self.val_losses = []
        self.train_metrics = []
        self.val_metrics = []
        self.best_val_loss = float('inf')
        self.best_model_state = None
        self.patience_counter = 0
        
    def create_model(self, model_type: str = "hybrid") -> nn.Module:
        """Create model based on type"""
        if model_type == "transformer":
            model = CryptoTransformer(self.model_config)
        elif model_type == "lstm":
            model = CryptoLSTM(self.model_config)
        elif model_type == "cnn":
            model = CryptoCNN(self.model_config)
        elif model_type == "hybrid":
            model = CryptoHybridModel(self.model_config)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        return model.to(self.device)
    
    def create_optimizer(self, model: nn.Module) -> optim.Optimizer:
        """Create optimizer"""
        if self.training_config.optimizer == "adam":
            optimizer = optim.Adam(
                model.parameters(),
                lr=self.training_config.learning_rate,
                weight_decay=self.training_config.weight_decay
            )
        elif self.training_config.optimizer == "adamw":
            optimizer = optim.AdamW(
                model.parameters(),
                lr=self.training_config.learning_rate,
                weight_decay=self.training_config.weight_decay
            )
        elif self.training_config.optimizer == "sgd":
            optimizer = optim.SGD(
                model.parameters(),
                lr=self.training_config.learning_rate,
                momentum=0.9,
                weight_decay=self.training_config.weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.training_config.optimizer}")
        
        return optimizer
    
    def create_scheduler(self, optimizer: optim.Optimizer, 
                        total_steps: int) -> Optional[optim.lr_scheduler._LRScheduler]:
        """Create learning rate scheduler"""
        if self.training_config.lr_scheduler == "cosine":
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=total_steps, eta_min=1e-6
            )
        elif self.training_config.lr_scheduler == "step":
            scheduler = optim.lr_scheduler.StepLR(
                optimizer, step_size=total_steps // 3, gamma=0.1
            )
        elif self.training_config.lr_scheduler == "plateau":
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=5, verbose=True
            )
        else:
            return None
        
        return scheduler
    
    def create_criterion(self) -> nn.Module:
        """Create loss function"""
        if self.model_config.output_type == "classification":
            if self.training_config.label_smoothing > 0:
                criterion = nn.CrossEntropyLoss(
                    label_smoothing=self.training_config.label_smoothing
                )
            else:
                criterion = nn.CrossEntropyLoss()
        else:
            criterion = nn.MSELoss()
        
        return criterion
    
    def prepare_data(self, features: pd.DataFrame, targets: pd.Series) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Prepare data loaders"""
        # Convert to numpy
        X = features.values.astype(np.float32)
        y = targets.values
        
        # Encode labels for classification
        if self.model_config.output_type == "classification":
            # Convert returns to classes: 0=sell, 1=hold, 2=buy
            y_classes = np.where(y > 0.001, 2, np.where(y < -0.001, 0, 1))
            y = y_classes.astype(np.int64)
        
        # Split data
        total_len = len(X)
        test_split_idx = int(total_len * (1 - self.training_config.test_split))
        val_split_idx = int(test_split_idx * (1 - self.training_config.validation_split))
        
        X_train, y_train = X[:val_split_idx], y[:val_split_idx]
        X_val, y_val = X[val_split_idx:test_split_idx], y[val_split_idx:test_split_idx]
        X_test, y_test = X[test_split_idx:], y[test_split_idx:]
        
        # Create datasets
        train_dataset = CryptoDataset(X_train, y_train, self.model_config.sequence_length)
        val_dataset = CryptoDataset(X_val, y_val, self.model_config.sequence_length)
        test_dataset = CryptoDataset(X_test, y_test, self.model_config.sequence_length)
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.model_config.batch_size,
            shuffle=True,
            num_workers=self.training_config.num_workers,
            pin_memory=self.training_config.pin_memory
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.model_config.batch_size,
            shuffle=False,
            num_workers=self.training_config.num_workers,
            pin_memory=self.training_config.pin_memory
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.model_config.batch_size,
            shuffle=False,
            num_workers=self.training_config.num_workers,
            pin_memory=self.training_config.pin_memory
        )
        
        return train_loader, val_loader, test_loader
    
    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, Dict[str, float]]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        
        for batch_idx, (features, targets) in enumerate(tqdm(train_loader, desc="Training")):
            features = features.to(self.device)
            targets = targets.to(self.device).squeeze()
            
            # Forward pass
            self.optimizer.zero_grad()
            
            if isinstance(self.model, CryptoHybridModel):
                outputs, _ = self.model(features)
            else:
                outputs, _ = self.model(features)
                
            loss = self.criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.training_config.gradient_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.training_config.gradient_clip_norm
                )
            
            self.optimizer.step()
            
            if self.scheduler and self.training_config.lr_scheduler != "plateau":
                self.scheduler.step()
            
            # Track metrics
            total_loss += loss.item()
            
            if self.model_config.output_type == "classification":
                predictions = torch.argmax(outputs, dim=1)
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
            
            # Log progress
            if batch_idx % self.training_config.log_interval == 0:
                logger.debug(f"Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.6f}")
        
        avg_loss = total_loss / len(train_loader)
        
        # Calculate metrics
        metrics = {}
        if self.model_config.output_type == "classification" and all_predictions:
            metrics['accuracy'] = accuracy_score(all_targets, all_predictions)
            metrics['precision'] = precision_score(all_targets, all_predictions, average='weighted', zero_division=0)
            metrics['recall'] = recall_score(all_targets, all_predictions, average='weighted', zero_division=0)
            metrics['f1'] = f1_score(all_targets, all_predictions, average='weighted', zero_division=0)
        
        return avg_loss, metrics
    
    def validate_epoch(self, val_loader: DataLoader) -> Tuple[float, Dict[str, float]]:
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for features, targets in tqdm(val_loader, desc="Validating"):
                features = features.to(self.device)
                targets = targets.to(self.device).squeeze()
                
                # Forward pass
                if isinstance(self.model, CryptoHybridModel):
                    outputs, _ = self.model(features)
                else:
                    outputs, _ = self.model(features)
                
                loss = self.criterion(outputs, targets)
                total_loss += loss.item()
                
                if self.model_config.output_type == "classification":
                    predictions = torch.argmax(outputs, dim=1)
                    all_predictions.extend(predictions.cpu().numpy())
                    all_targets.extend(targets.cpu().numpy())
        
        avg_loss = total_loss / len(val_loader)
        
        # Calculate metrics
        metrics = {}
        if self.model_config.output_type == "classification" and all_predictions:
            metrics['accuracy'] = accuracy_score(all_targets, all_predictions)
            metrics['precision'] = precision_score(all_targets, all_predictions, average='weighted', zero_division=0)
            metrics['recall'] = recall_score(all_targets, all_predictions, average='weighted', zero_division=0)
            metrics['f1'] = f1_score(all_targets, all_predictions, average='weighted', zero_division=0)
        
        return avg_loss, metrics
    
    def train(self, features: pd.DataFrame, targets: pd.Series, 
              model_type: str = "hybrid") -> Dict[str, Any]:
        """Main training loop"""
        logger.info("Starting model training...")
        
        # Prepare data
        train_loader, val_loader, test_loader = self.prepare_data(features, targets)
        
        # Create model
        self.model = self.create_model(model_type)
        self.optimizer = self.create_optimizer(self.model)
        self.criterion = self.create_criterion()
        
        # Calculate total steps for scheduler
        total_steps = len(train_loader) * self.training_config.max_epochs
        self.scheduler = self.create_scheduler(self.optimizer, total_steps)
        
        logger.info(f"Model: {model_type}, Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        logger.info(f"Training samples: {len(train_loader.dataset)}")
        logger.info(f"Validation samples: {len(val_loader.dataset)}")
        logger.info(f"Test samples: {len(test_loader.dataset)}")
        
        # Training loop
        for epoch in range(self.training_config.max_epochs):
            logger.info(f"Epoch {epoch + 1}/{self.training_config.max_epochs}")
            
            # Train
            train_loss, train_metrics = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)
            self.train_metrics.append(train_metrics)
            
            # Validate
            val_loss, val_metrics = self.validate_epoch(val_loader)
            self.val_losses.append(val_loss)
            self.val_metrics.append(val_metrics)
            
            # Scheduler step for plateau
            if self.scheduler and self.training_config.lr_scheduler == "plateau":
                self.scheduler.step(val_loss)
            
            # Log metrics
            logger.info(f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
            for metric, value in train_metrics.items():
                logger.info(f"Train {metric}: {value:.4f}")
            for metric, value in val_metrics.items():
                logger.info(f"Val {metric}: {value:.4f}")
            
            # Early stopping
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_model_state = self.model.state_dict().copy()
                self.patience_counter = 0
                logger.info("New best model saved!")
            else:
                self.patience_counter += 1
                
            if self.patience_counter >= self.training_config.early_stopping_patience:
                logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                break
        
        # Load best model for final evaluation
        if self.best_model_state:
            self.model.load_state_dict(self.best_model_state)
        
        # Final test evaluation
        test_loss, test_metrics = self.validate_epoch(test_loader)
        
        results = {
            'best_val_loss': self.best_val_loss,
            'test_loss': test_loss,
            'test_metrics': test_metrics,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_metrics': self.train_metrics,
            'val_metrics': self.val_metrics,
            'total_epochs': epoch + 1,
            'model_type': model_type
        }
        
        logger.info("Training completed!")
        logger.info(f"Best validation loss: {self.best_val_loss:.6f}")
        logger.info(f"Test loss: {test_loss:.6f}")
        for metric, value in test_metrics.items():
            logger.info(f"Test {metric}: {value:.4f}")
        
        return results
    
    def save_model(self, filepath: Path, metadata: Optional[Dict] = None):
        """Save model and training state"""
        save_dict = {
            'model_state_dict': self.model.state_dict(),
            'model_config': asdict(self.model_config),
            'training_config': asdict(self.training_config),
            'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_loss': self.best_val_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'metadata': metadata or {}
        }
        
        torch.save(save_dict, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: Path, model_type: str = "hybrid"):
        """Load model and training state"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        # Restore configs
        self.model_config = ModelConfig(**checkpoint['model_config'])
        self.training_config = TrainingConfig(**checkpoint['training_config'])
        
        # Create and load model
        self.model = self.create_model(model_type)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Restore training state
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        
        logger.info(f"Model loaded from {filepath}")
        
        return checkpoint.get('metadata', {})