"""CNN-LSTM hybrid model for cryptocurrency price prediction.

Implementation based on Alonso-Monsalve et al. 2020:
"Convolution on neural networks for high-frequency trend prediction 
of cryptocurrency exchange rates using technical indicators"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from typing import Tuple, Dict, Any, Optional
import structlog

logger = structlog.get_logger(__name__)


class CNN_LSTM(nn.Module):
    """CNN-LSTM hybrid model for cryptocurrency trend prediction.
    
    Architecture:
    1. 1D CNN layers for feature extraction from technical indicators
    2. LSTM layers for temporal sequence modeling
    3. Fully connected layers for binary classification
    """
    
    def __init__(
        self,
        n_features: int = 18,
        sequence_length: int = 60,
        cnn_filters: list = [32, 64, 128],
        cnn_kernel_sizes: list = [3, 5, 7],
        lstm_hidden_size: int = 128,
        lstm_num_layers: int = 2,
        dropout_rate: float = 0.3,
        fc_hidden_size: int = 64
    ):
        """Initialize CNN-LSTM model.
        
        Args:
            n_features: Number of input features (18 technical indicators)
            sequence_length: Length of input sequence (time steps)
            cnn_filters: List of filter sizes for CNN layers
            cnn_kernel_sizes: List of kernel sizes for CNN layers
            lstm_hidden_size: Hidden size of LSTM layers
            lstm_num_layers: Number of LSTM layers
            dropout_rate: Dropout rate for regularization
            fc_hidden_size: Hidden size of fully connected layer
        """
        super(CNN_LSTM, self).__init__()
        
        self.n_features = n_features
        self.sequence_length = sequence_length
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_num_layers = lstm_num_layers
        
        # CNN layers for feature extraction
        self.conv_layers = nn.ModuleList()
        in_channels = n_features
        
        for i, (filters, kernel_size) in enumerate(zip(cnn_filters, cnn_kernel_sizes)):
            self.conv_layers.append(nn.Sequential(
                nn.Conv1d(in_channels, filters, kernel_size, padding=kernel_size//2),
                nn.BatchNorm1d(filters),
                nn.ReLU(),
                nn.MaxPool1d(2),
                nn.Dropout(dropout_rate)
            ))
            in_channels = filters
        
        # Calculate the size after CNN layers
        cnn_output_length = sequence_length
        for _ in cnn_filters:
            cnn_output_length = cnn_output_length // 2  # MaxPool1d(2)
        
        # LSTM layers for temporal modeling
        self.lstm = nn.LSTM(
            input_size=cnn_filters[-1],
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
            dropout=dropout_rate if lstm_num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Fully connected layers for classification
        lstm_output_size = lstm_hidden_size * 2  # Bidirectional
        self.fc_layers = nn.Sequential(
            nn.Linear(lstm_output_size, fc_hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(fc_hidden_size, 1),
            nn.Sigmoid()
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LSTM):
                for name, param in module.named_parameters():
                    if 'weight' in name:
                        nn.init.xavier_normal_(param)
                    elif 'bias' in name:
                        nn.init.constant_(param, 0)
    
    def forward(self, x):
        """Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, n_features)
            
        Returns:
            Output probabilities of shape (batch_size, 1)
        """
        batch_size, seq_len, n_feat = x.shape
        
        # Transpose for CNN: (batch_size, n_features, sequence_length)
        x = x.transpose(1, 2)
        
        # CNN feature extraction
        for conv_layer in self.conv_layers:
            x = conv_layer(x)
        
        # Transpose back for LSTM: (batch_size, sequence_length, n_features)
        x = x.transpose(1, 2)
        
        # LSTM temporal modeling
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Use the last output of the LSTM
        last_output = lstm_out[:, -1, :]  # (batch_size, lstm_hidden_size * 2)
        
        # Fully connected classification
        output = self.fc_layers(last_output)
        
        return output


class CNN_LSTM_Trainer:
    """Trainer class for CNN-LSTM model."""
    
    def __init__(
        self,
        model: CNN_LSTM,
        device: str = 'auto',
        learning_rate: float = 0.001,
        weight_decay: float = 1e-5
    ):
        """Initialize trainer.
        
        Args:
            model: CNN-LSTM model instance
            device: Device to run on ('auto', 'cuda', 'mps', 'cpu')
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for L2 regularization
        """
        self.model = model
        
        # Auto-detect device
        if device == 'auto':
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            elif torch.backends.mps.is_available():
                self.device = torch.device('mps')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = torch.device(device)
        
        self.model.to(self.device)
        
        # Loss function and optimizer
        self.criterion = nn.BCELoss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=learning_rate, 
            weight_decay=weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=10, factor=0.5
        )
        
        logger.info("CNN-LSTM Trainer initialized", device=str(self.device))
    
    def prepare_data(
        self, 
        df: pd.DataFrame, 
        sequence_length: int = 60,
        test_size: float = 0.2,
        val_size: float = 0.1,
        normalize: bool = True
    ) -> Tuple[torch.utils.data.DataLoader, ...]:
        """Prepare data for training.
        
        Args:
            df: DataFrame with OHLCV data
            sequence_length: Length of input sequences
            test_size: Fraction of data for testing
            val_size: Fraction of data for validation
            normalize: Whether to normalize features
            
        Returns:
            Tuple of (train_loader, val_loader, test_loader, scaler)
        """
        from ..utils.indicators import prepare_cnn_lstm_data
        
        # Prepare features and targets
        X, y = prepare_cnn_lstm_data(df, sequence_length)
        
        logger.info(
            "Data prepared for CNN-LSTM",
            total_samples=len(X),
            sequence_length=sequence_length,
            n_features=X.shape[2],
            positive_samples=int(y.sum()),
            negative_samples=int(len(y) - y.sum())
        )
        
        # Split data
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, random_state=42, stratify=y_temp
        )
        
        # Normalize features if requested
        scaler = None
        if normalize:
            scaler = StandardScaler()
            # Reshape for scaling: (samples * timesteps, features)
            X_train_reshaped = X_train.reshape(-1, X_train.shape[-1])
            scaler.fit(X_train_reshaped)
            
            # Scale all datasets
            X_train = scaler.transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
            X_val = scaler.transform(X_val.reshape(-1, X_val.shape[-1])).reshape(X_val.shape)
            X_test = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)
        
        # Convert to PyTorch tensors
        X_train = torch.FloatTensor(X_train)
        X_val = torch.FloatTensor(X_val)
        X_test = torch.FloatTensor(X_test)
        y_train = torch.FloatTensor(y_train).unsqueeze(1)
        y_val = torch.FloatTensor(y_val).unsqueeze(1)
        y_test = torch.FloatTensor(y_test).unsqueeze(1)
        
        # Create data loaders
        train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
        val_dataset = torch.utils.data.TensorDataset(X_val, y_val)
        test_dataset = torch.utils.data.TensorDataset(X_test, y_test)
        
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=64, shuffle=True
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=64, shuffle=False
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=64, shuffle=False
        )
        
        return train_loader, val_loader, test_loader, scaler
    
    def train(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        epochs: int = 100,
        early_stopping_patience: int = 20,
        save_best_model: str = 'best_cnn_lstm_model.pth'
    ) -> Dict[str, list]:
        """Train the model.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of training epochs
            early_stopping_patience: Patience for early stopping
            save_best_model: Path to save best model
            
        Returns:
            Dictionary with training history
        """
        history = {
            'train_loss': [],
            'val_loss': [],
            'val_accuracy': []
        }
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        logger.info("Starting CNN-LSTM training", epochs=epochs)
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            
            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item()
            
            # Validation phase
            self.model.eval()
            val_loss = 0.0
            val_predictions = []
            val_targets = []
            
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X = batch_X.to(self.device)
                    batch_y = batch_y.to(self.device)
                    
                    outputs = self.model(batch_X)
                    loss = self.criterion(outputs, batch_y)
                    val_loss += loss.item()
                    
                    # Collect predictions for metrics
                    predictions = (outputs > 0.5).float()
                    val_predictions.extend(predictions.cpu().numpy())
                    val_targets.extend(batch_y.cpu().numpy())
            
            # Calculate metrics
            train_loss /= len(train_loader)
            val_loss /= len(val_loader)
            val_accuracy = accuracy_score(val_targets, val_predictions)
            
            # Update learning rate
            self.scheduler.step(val_loss)
            
            # Save history
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['val_accuracy'].append(val_accuracy)
            
            # Early stopping and model saving
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                if save_best_model:
                    torch.save(self.model.state_dict(), save_best_model)
            else:
                patience_counter += 1
            
            # Log progress
            if epoch % 10 == 0 or epoch == epochs - 1:
                logger.info(
                    "Training progress",
                    epoch=epoch,
                    train_loss=round(train_loss, 4),
                    val_loss=round(val_loss, 4),
                    val_accuracy=round(val_accuracy, 4),
                    lr=self.optimizer.param_groups[0]['lr']
                )
            
            # Early stopping
            if patience_counter >= early_stopping_patience:
                logger.info("Early stopping triggered", epoch=epoch)
                break
        
        logger.info("Training completed", best_val_loss=best_val_loss)
        return history
    
    def evaluate(self, test_loader: torch.utils.data.DataLoader) -> Dict[str, float]:
        """Evaluate model on test data.
        
        Args:
            test_loader: Test data loader
            
        Returns:
            Dictionary with evaluation metrics
        """
        self.model.eval()
        all_predictions = []
        all_targets = []
        all_probabilities = []
        
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                outputs = self.model(batch_X)
                predictions = (outputs > 0.5).float()
                
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(batch_y.cpu().numpy())
                all_probabilities.extend(outputs.cpu().numpy())
        
        # Calculate metrics
        accuracy = accuracy_score(all_targets, all_predictions)
        precision = precision_score(all_targets, all_predictions, zero_division=0)
        recall = recall_score(all_targets, all_predictions, zero_division=0)
        f1 = f1_score(all_targets, all_predictions, zero_division=0)
        auc = roc_auc_score(all_targets, all_probabilities)
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc_score': auc
        }
        
        logger.info("Model evaluation completed", **metrics)
        return metrics