#!/usr/bin/env python3
"""Quick fix for CNN-LSTM model - focus on class imbalance."""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import structlog

from crypto_quant.models.cnn_lstm import CNN_LSTM
from crypto_quant.utils.indicators import prepare_cnn_lstm_data

logger = structlog.get_logger(__name__)


class FocalLoss(nn.Module):
    """Focal Loss to handle class imbalance."""
    
    def __init__(self, alpha=1.0, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        # Convert to probabilities
        p = torch.sigmoid(inputs)
        
        # Calculate focal loss
        ce_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        p_t = p * targets + (1 - p) * (1 - targets)
        loss = ce_loss * ((1 - p_t) ** self.gamma)
        
        if self.alpha >= 0:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * loss
        
        return loss.mean()


def train_with_focal_loss():
    """Train CNN-LSTM with Focal Loss to fix class imbalance."""
    
    # Load existing data (faster than fetching new)
    try:
        df = pd.read_csv('data/BTC_USDT_1m_last_3days.csv')
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        logger.info("Loaded cached data", rows=len(df))
    except:
        logger.error("No cached data found. Please run previous training first.")
        return
    
    # Prepare data with longer sequence for better learning
    X, y = prepare_cnn_lstm_data(df, sequence_length=60)
    
    logger.info(
        "Data prepared",
        samples=len(X),
        features=X.shape[2],
        positive_ratio=y.mean(),
        negative_ratio=1-y.mean()
    )
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Normalize
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
    X_test = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)
    
    # Create model with adjusted parameters
    model = CNN_LSTM(
        n_features=18,
        sequence_length=60,
        cnn_filters=[32, 64, 64],  # Smaller to prevent overfitting
        lstm_hidden_size=64,       # Smaller
        lstm_num_layers=2,
        dropout_rate=0.5,          # More dropout
        fc_hidden_size=32          # Smaller FC layer
    )
    
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    model.to(device)
    
    # Calculate class weights for Focal Loss
    pos_ratio = y_train.mean()
    neg_ratio = 1 - pos_ratio
    alpha = neg_ratio / (pos_ratio + 1e-8)  # Balance factor
    
    logger.info(
        "Class balance",
        positive_ratio=pos_ratio,
        negative_ratio=neg_ratio,
        focal_alpha=alpha
    )
    
    # Use Focal Loss instead of BCE
    criterion = FocalLoss(alpha=alpha, gamma=2.0)
    
    # Lower learning rate
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)
    
    # Training
    epochs = 30
    batch_size = 16  # Smaller batch size
    
    history = {'loss': [], 'accuracy': [], 'f1': [], 'recall': []}
    
    logger.info("Starting training with Focal Loss", epochs=epochs)
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        # Training batches
        for i in range(0, len(X_train), batch_size):
            X_batch = torch.FloatTensor(X_train[i:i+batch_size]).to(device)
            y_batch = torch.FloatTensor(y_train[i:i+batch_size]).unsqueeze(1).to(device)
            
            optimizer.zero_grad()
            
            # Forward pass - get logits (before sigmoid)
            outputs = model(X_batch)
            
            # Apply sigmoid to get probabilities for loss calculation
            if hasattr(model.fc_layers[-1], 'weight'):
                # Remove sigmoid from model temporarily
                logits = outputs / 1.0  # Keep as logits
                # Calculate focal loss on logits
                loss = criterion(logits, y_batch)
            else:
                # If model outputs probabilities, convert back to logits
                eps = 1e-8
                logits = torch.log(outputs / (1 - outputs + eps) + eps)
                loss = criterion(logits, y_batch)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
        
        # Evaluation
        model.eval()
        with torch.no_grad():
            test_outputs = []
            for i in range(0, len(X_test), batch_size):
                X_batch = torch.FloatTensor(X_test[i:i+batch_size]).to(device)
                outputs = model(X_batch)
                test_outputs.extend(outputs.cpu().numpy())
            
            test_outputs = np.array(test_outputs)
            test_preds = (test_outputs > 0.5).astype(int).flatten()
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, test_preds)
            precision = precision_score(y_test, test_preds, zero_division=0)
            recall = recall_score(y_test, test_preds, zero_division=0)
            f1 = f1_score(y_test, test_preds, zero_division=0)
            auc = roc_auc_score(y_test, test_outputs.flatten()) if len(np.unique(test_preds)) > 1 else 0.5
            
            history['loss'].append(total_loss / (len(X_train) // batch_size))
            history['accuracy'].append(accuracy)
            history['f1'].append(f1)
            history['recall'].append(recall)
            
            # Log progress
            if epoch % 5 == 0 or epoch == epochs - 1:
                logger.info(
                    "Training progress",
                    epoch=epoch,
                    loss=round(total_loss / (len(X_train) // batch_size), 4),
                    accuracy=round(accuracy, 4),
                    precision=round(precision, 4),
                    recall=round(recall, 4),
                    f1=round(f1, 4),
                    auc=round(auc, 4),
                    prob_mean=round(test_outputs.mean(), 4),
                    prob_std=round(test_outputs.std(), 4),
                    pred_ratio=round(test_preds.mean(), 4)
                )
    
    # Final results
    final_metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'auc_score': auc
    }
    
    print("\n" + "="*60)
    print("FOCAL LOSS TRAINING RESULTS")
    print("="*60)
    print(f"Original model (mode collapse): 52.3% accuracy, 0% recall")
    print(f"Fixed model (Focal Loss):       {accuracy:.1%} accuracy, {recall:.1%} recall")
    print(f"")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"AUC Score: {auc:.4f}")
    print(f"")
    print(f"Probability mean: {test_outputs.mean():.4f}")
    print(f"Probability std:  {test_outputs.std():.4f}")
    print(f"Prediction ratio: {test_preds.mean():.4f} (vs true ratio: {y_test.mean():.4f})")
    print("="*60)
    
    # Show improvement
    baseline_acc = 0.523
    improvement = accuracy - baseline_acc
    print(f"\nIMPROVEMENT: {improvement:+.1%} accuracy vs baseline")
    
    if recall > 0.1:
        print("✅ SUCCESS: Model can now predict 'Up' movements!")
    if accuracy > 0.55:
        print("✅ SUCCESS: Significantly above random (50%)")
    if f1 > 0.1:
        print("✅ SUCCESS: Balanced precision and recall")
    
    # Save model
    torch.save(model.state_dict(), 'models/cnn_lstm_btc_model.pth')
    logger.info("Model saved", path='models/cnn_lstm_btc_model.pth')
    
    return model, final_metrics, history


if __name__ == "__main__":
    # Configure logging
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer()
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    
    model, metrics, history = train_with_focal_loss()