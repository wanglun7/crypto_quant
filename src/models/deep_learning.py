"""
Deep learning models for crypto prediction
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Dict, List
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration for deep learning models"""
    # Data parameters
    sequence_length: int = 100
    feature_dim: int = 300
    prediction_horizon: int = 1
    
    # Transformer parameters
    d_model: int = 256
    nhead: int = 8
    num_encoder_layers: int = 4
    dim_feedforward: int = 512
    dropout: float = 0.1
    
    # LSTM parameters
    lstm_hidden_size: int = 256
    lstm_num_layers: int = 3
    lstm_bidirectional: bool = True
    
    # CNN parameters
    cnn_channels: List[int] = None
    cnn_kernel_sizes: List[int] = None
    
    # Output parameters
    num_classes: int = 3  # Buy, Hold, Sell
    output_type: str = "classification"  # "classification" or "regression"
    
    # Training parameters
    learning_rate: float = 0.001
    weight_decay: float = 1e-5
    batch_size: int = 64
    
    def __post_init__(self):
        if self.cnn_channels is None:
            self.cnn_channels = [128, 64, 32]
        if self.cnn_kernel_sizes is None:
            self.cnn_kernel_sizes = [3, 5, 7]


class PositionalEncoding(nn.Module):
    """Add positional encoding to transformer inputs"""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:x.size(0), :]


class MultiHeadAttentionBlock(nn.Module):
    """Custom multi-head attention with residual connections"""
    
    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, d_model * 4)
        self.linear2 = nn.Linear(d_model * 4, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, 
                src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-attention
        attn_output, _ = self.self_attn(x, x, x, attn_mask=src_mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed forward
        ff_output = self.linear2(F.relu(self.linear1(x)))
        x = self.norm2(x + self.dropout(ff_output))
        
        return x


class CryptoTransformer(nn.Module):
    """Transformer model for crypto price prediction"""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Input projection
        self.input_projection = nn.Linear(config.feature_dim, config.d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(config.d_model, config.sequence_length)
        
        # Transformer encoder layers
        self.transformer_layers = nn.ModuleList([
            MultiHeadAttentionBlock(config.d_model, config.nhead, config.dropout)
            for _ in range(config.num_encoder_layers)
        ])
        
        # Global attention pooling
        self.attention_pool = nn.MultiheadAttention(
            config.d_model, config.nhead, batch_first=True
        )
        
        # Output layers
        if config.output_type == "classification":
            self.classifier = nn.Sequential(
                nn.Linear(config.d_model, config.d_model // 2),
                nn.ReLU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.d_model // 2, config.num_classes)
            )
        else:
            self.regressor = nn.Sequential(
                nn.Linear(config.d_model, config.d_model // 2),
                nn.ReLU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.d_model // 2, 1)
            )
            
    def create_mask(self, seq_len: int) -> torch.Tensor:
        """Create causal mask for transformer"""
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass
        Args:
            x: (batch_size, sequence_length, feature_dim)
        Returns:
            output: (batch_size, num_classes) or (batch_size, 1)
            attention_weights: (batch_size, sequence_length)
        """
        batch_size, seq_len, _ = x.shape
        
        # Input projection
        x = self.input_projection(x)  # (batch, seq, d_model)
        
        # Add positional encoding
        x = x.transpose(0, 1)  # (seq, batch, d_model)
        x = self.pos_encoder(x)
        x = x.transpose(0, 1)  # (batch, seq, d_model)
        
        # Create attention mask
        mask = self.create_mask(seq_len).to(x.device)
        
        # Pass through transformer layers
        for layer in self.transformer_layers:
            x = layer(x, mask)
        
        # Global attention pooling
        # Use learnable query vector
        query = x.mean(dim=1, keepdim=True)  # (batch, 1, d_model)
        pooled_output, attention_weights = self.attention_pool(query, x, x)
        pooled_output = pooled_output.squeeze(1)  # (batch, d_model)
        
        # Classification or regression
        if self.config.output_type == "classification":
            output = self.classifier(pooled_output)
        else:
            output = self.regressor(pooled_output)
            
        return output, attention_weights.squeeze(1)


class CryptoLSTM(nn.Module):
    """LSTM model for crypto prediction"""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=config.feature_dim,
            hidden_size=config.lstm_hidden_size,
            num_layers=config.lstm_num_layers,
            dropout=config.dropout if config.lstm_num_layers > 1 else 0,
            bidirectional=config.lstm_bidirectional,
            batch_first=True
        )
        
        lstm_output_size = (config.lstm_hidden_size * 2 
                           if config.lstm_bidirectional 
                           else config.lstm_hidden_size)
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(lstm_output_size, lstm_output_size // 2),
            nn.Tanh(),
            nn.Linear(lstm_output_size // 2, 1)
        )
        
        # Output layers
        if config.output_type == "classification":
            self.classifier = nn.Sequential(
                nn.Linear(lstm_output_size, lstm_output_size // 2),
                nn.ReLU(),
                nn.Dropout(config.dropout),
                nn.Linear(lstm_output_size // 2, config.num_classes)
            )
        else:
            self.regressor = nn.Sequential(
                nn.Linear(lstm_output_size, lstm_output_size // 2),
                nn.ReLU(),
                nn.Dropout(config.dropout),
                nn.Linear(lstm_output_size // 2, 1)
            )
            
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass
        Args:
            x: (batch_size, sequence_length, feature_dim)
        Returns:
            output: (batch_size, num_classes) or (batch_size, 1)
            attention_weights: (batch_size, sequence_length)
        """
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)  # (batch, seq, hidden_size)
        
        # Attention mechanism
        attention_scores = self.attention(lstm_out)  # (batch, seq, 1)
        attention_weights = F.softmax(attention_scores, dim=1)  # (batch, seq, 1)
        
        # Weighted sum
        context_vector = torch.sum(lstm_out * attention_weights, dim=1)  # (batch, hidden_size)
        
        # Classification or regression
        if self.config.output_type == "classification":
            output = self.classifier(context_vector)
        else:
            output = self.regressor(context_vector)
            
        return output, attention_weights.squeeze(-1)


class CryptoCNN(nn.Module):
    """CNN model for pattern recognition in crypto data"""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Multiple CNN branches with different kernel sizes
        self.conv_branches = nn.ModuleList()
        
        for i, (channels, kernel_size) in enumerate(zip(config.cnn_channels, config.cnn_kernel_sizes)):
            branch = nn.Sequential(
                nn.Conv1d(config.feature_dim, channels, kernel_size, padding=kernel_size//2),
                nn.BatchNorm1d(channels),
                nn.ReLU(),
                nn.Conv1d(channels, channels, kernel_size, padding=kernel_size//2),
                nn.BatchNorm1d(channels),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1)
            )
            self.conv_branches.append(branch)
        
        # Combine branches
        total_channels = sum(config.cnn_channels)
        
        # Output layers
        if config.output_type == "classification":
            self.classifier = nn.Sequential(
                nn.Linear(total_channels, total_channels // 2),
                nn.ReLU(),
                nn.Dropout(config.dropout),
                nn.Linear(total_channels // 2, config.num_classes)
            )
        else:
            self.regressor = nn.Sequential(
                nn.Linear(total_channels, total_channels // 2),
                nn.ReLU(),
                nn.Dropout(config.dropout),
                nn.Linear(total_channels // 2, 1)
            )
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        Args:
            x: (batch_size, sequence_length, feature_dim)
        Returns:
            output: (batch_size, num_classes) or (batch_size, 1)
        """
        # Transpose for Conv1d: (batch, feature_dim, sequence_length)
        x = x.transpose(1, 2)
        
        # Apply each CNN branch
        branch_outputs = []
        for branch in self.conv_branches:
            branch_out = branch(x)  # (batch, channels, 1)
            branch_outputs.append(branch_out.squeeze(-1))  # (batch, channels)
        
        # Concatenate all branches
        combined = torch.cat(branch_outputs, dim=1)  # (batch, total_channels)
        
        # Classification or regression
        if self.config.output_type == "classification":
            output = self.classifier(combined)
        else:
            output = self.regressor(combined)
            
        return output


class CryptoHybridModel(nn.Module):
    """Hybrid model combining Transformer, LSTM, and CNN"""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Sub-models
        self.transformer = CryptoTransformer(config)
        self.lstm = CryptoLSTM(config)
        self.cnn = CryptoCNN(config)
        
        # Fusion layer
        if config.output_type == "classification":
            fusion_input_size = config.num_classes * 3  # 3 models
            self.fusion = nn.Sequential(
                nn.Linear(fusion_input_size, fusion_input_size // 2),
                nn.ReLU(),
                nn.Dropout(config.dropout),
                nn.Linear(fusion_input_size // 2, config.num_classes)
            )
        else:
            fusion_input_size = 3  # 3 models
            self.fusion = nn.Sequential(
                nn.Linear(fusion_input_size, 16),
                nn.ReLU(),
                nn.Dropout(config.dropout),
                nn.Linear(16, 1)
            )
        
        # Model weights (learnable)
        self.model_weights = nn.Parameter(torch.ones(3) / 3)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass
        Args:
            x: (batch_size, sequence_length, feature_dim)
        Returns:
            output: (batch_size, num_classes) or (batch_size, 1)
            attention_info: Dictionary with attention weights from each model
        """
        # Get predictions from each model
        transformer_out, transformer_attn = self.transformer(x)
        lstm_out, lstm_attn = self.lstm(x)
        cnn_out = self.cnn(x)
        
        # Weighted combination
        weights = F.softmax(self.model_weights, dim=0)
        
        if self.config.output_type == "classification":
            # Concatenate logits
            combined_logits = torch.cat([transformer_out, lstm_out, cnn_out], dim=1)
            output = self.fusion(combined_logits)
        else:
            # Weighted average
            weighted_outputs = (weights[0] * transformer_out + 
                              weights[1] * lstm_out + 
                              weights[2] * cnn_out)
            output = self.fusion(weighted_outputs)
        
        attention_info = {
            'transformer_attention': transformer_attn,
            'lstm_attention': lstm_attn,
            'model_weights': weights
        }
        
        return output, attention_info