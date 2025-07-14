import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler
import numpy as np
from sklearn.metrics import balanced_accuracy_score
import os


class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size=256, num_classes=3, dropout=0.3):
        super(SimpleNN, self).__init__()
        
        # Flatten sequential input to use in feedforward network
        self.flatten_size = input_size  # This will be lookback * features
        
        # Multi-layer feedforward network
        self.network = nn.Sequential(
            nn.Linear(self.flatten_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 4, num_classes)
        )
        
    def forward(self, x):
        # x shape: (batch, seq_len, input_size)
        # Flatten to (batch, seq_len * input_size)
        x_flat = x.view(x.size(0), -1)
        
        # Forward pass
        out = self.network(x_flat)
        return out
    
    def predict_proba(self, x, temperature=1.0):
        """Return probability distribution over classes with optional temperature scaling."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            # Apply temperature scaling
            scaled_logits = logits / temperature
            proba = torch.softmax(scaled_logits, dim=1)
        return proba


class SimpleConvNet(nn.Module):
    def __init__(self, input_size, seq_len, num_classes=3, dropout=0.3):
        super(SimpleConvNet, self).__init__()
        
        # 1D Convolutional layers for time series
        self.conv1 = nn.Conv1d(input_size, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        
        self.pool = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(dropout)
        
        # Calculate the size after convolutions and pooling
        conv_out_size = 256 * (seq_len // 8)  # 3 pooling layers: 60 -> 30 -> 15 -> 7
        
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        # x shape: (batch, seq_len, input_size)
        # Conv1d expects (batch, channels, seq_len)
        x = x.transpose(1, 2)  # (batch, input_size, seq_len)
        
        # Convolutional layers
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected
        x = self.fc(x)
        return x
    
    def predict_proba(self, x, temperature=1.0):
        """Return probability distribution over classes with optional temperature scaling."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            # Apply temperature scaling
            scaled_logits = logits / temperature
            proba = torch.softmax(scaled_logits, dim=1)
        return proba


def train_simple_nn(X, y, epochs=10, batch_size=64, lr=1e-3, model_type="feedforward"):
    """
    Train Simple Neural Network model on the dataset.
    
    Args:
        X: Feature array (N, lookback, num_features)
        y: Label array (N,)
        epochs: Number of training epochs (max 10)
        batch_size: Batch size for training
        lr: Learning rate
        model_type: "feedforward" or "conv"
        
    Returns:
        dict with 'val_bacc', 'test_bacc', 'ckpt_path'
    """
    # Features are already normalized in dataset.py, skip additional normalization
    X_norm = X
    X_mean = np.zeros((1, 1, X.shape[2]))  # Dummy for compatibility
    X_std = np.ones((1, 1, X.shape[2]))    # Dummy for compatibility
    
    # Data split: 70% train, 15% val, 15% test (time series order)
    n_samples = len(X_norm)
    n_train = int(0.7 * n_samples)
    n_val = int(0.15 * n_samples)
    
    X_train = X_norm[:n_train]
    y_train = y[:n_train]
    X_val = X_norm[n_train:n_train+n_val]
    y_val = y[n_train:n_train+n_val]
    X_test = X_norm[n_train+n_val:]
    y_test = y[n_train+n_val:]
    
    # Convert to tensors
    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.LongTensor(y_train)
    X_val_t = torch.FloatTensor(X_val)
    y_val_t = torch.LongTensor(y_val)
    X_test_t = torch.FloatTensor(X_test)
    y_test_t = torch.LongTensor(y_test)
    
    # Compute class weights for imbalanced data
    unique_classes, counts = np.unique(y_train, return_counts=True)
    class_weights = 1.0 / counts  # Inverse frequency weighting
    class_weights = class_weights / class_weights.sum() * len(unique_classes)  # Normalize
    
    # Ensure we have weights for all 3 classes
    class_weights_full = np.ones(3)
    for i, cls in enumerate(unique_classes):
        class_weights_full[cls] = class_weights[i]
    
    print(f"\\nClass weights for loss function:")
    for i in range(3):
        print(f"  Class {i}: {class_weights_full[i]:.3f}")
    
    # Create sample weights for WeightedRandomSampler
    sample_weights = np.array([class_weights_full[label] for label in y_train])
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights))
    
    # Create data loaders
    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
    
    # Initialize model
    if model_type == "conv":
        model = SimpleConvNet(input_size=X.shape[2], seq_len=X.shape[1], num_classes=3, dropout=0.3)
        ckpt_name = "simple_conv_latest.pt"
    else:
        # Feedforward network
        flattened_size = X.shape[1] * X.shape[2]  # lookback * features
        model = SimpleNN(input_size=flattened_size, hidden_size=256, num_classes=3, dropout=0.3)
        ckpt_name = "simple_nn_latest.pt"
    
    # Loss with class weights for better handling of imbalanced classes
    criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor(class_weights_full))
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=2, factor=0.7)
    
    # Training loop with early stopping
    best_val_bacc = 0
    patience_counter = 0
    patience = 5  # Increased patience
    
    # Train for longer with more data
    max_epochs = max(epochs, 30)
    
    for epoch in range(max_epochs):
        # Training
        model.train()
        total_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_t)
            val_preds = torch.argmax(val_outputs, dim=1).numpy()
            val_bacc = balanced_accuracy_score(y_val, val_preds)
        
        # Learning rate scheduling
        scheduler.step(val_bacc)
        
        # Early stopping check
        if val_bacc > best_val_bacc:
            best_val_bacc = val_bacc
            patience_counter = 0
            # Save best model
            torch.save({
                'model_state_dict': model.state_dict(),
                'model_type': model_type,
                'input_size': X.shape[2],
                'seq_len': X.shape[1],
                'X_mean': X_mean,
                'X_std': X_std,
            }, ckpt_name)
        else:
            patience_counter += 1
            
        # Early stopping
        if patience_counter >= patience:
            break
    
    # Final evaluation on test set
    # Load best model
    checkpoint = torch.load(ckpt_name, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    with torch.no_grad():
        test_outputs = model(X_test_t)
        test_preds = torch.argmax(test_outputs, dim=1).numpy()
        test_bacc = balanced_accuracy_score(y_test, test_preds)
    
    return {
        'val_bacc': best_val_bacc,
        'test_bacc': test_bacc,
        'ckpt_path': ckpt_name
    }


def load_simple_nn(ckpt_path="simple_nn_latest.pt"):
    """Load trained Simple NN model from checkpoint."""
    checkpoint = torch.load(ckpt_path, weights_only=False)
    model_type = checkpoint.get('model_type', 'feedforward')
    
    if model_type == "conv":
        model = SimpleConvNet(
            input_size=checkpoint['input_size'], 
            seq_len=checkpoint['seq_len'],
            num_classes=3, 
            dropout=0.3
        )
    else:
        # Feedforward network
        flattened_size = checkpoint['seq_len'] * checkpoint['input_size']
        model = SimpleNN(
            input_size=flattened_size, 
            hidden_size=256, 
            num_classes=3, 
            dropout=0.3
        )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Store normalization parameters
    model.X_mean = checkpoint.get('X_mean', None)
    model.X_std = checkpoint.get('X_std', None)
    
    return model