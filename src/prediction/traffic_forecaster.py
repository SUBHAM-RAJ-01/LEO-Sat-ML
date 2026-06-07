"""
LSTM and GRU-based traffic prediction models for LEO satellite networks.
Predicts future congestion and traffic conditions for proactive routing.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


class TrafficDataset(Dataset):
    """PyTorch Dataset for time-series traffic prediction."""
    
    def __init__(self, sequences, targets):
        self.sequences = torch.FloatTensor(sequences)
        self.targets = torch.FloatTensor(targets)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]


class LSTMTrafficPredictor(nn.Module):
    """LSTM model for traffic congestion forecasting."""
    
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.2, prediction_horizon=10):
        super(LSTMTrafficPredictor, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.prediction_horizon = prediction_horizon
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, prediction_horizon)
        )
    
    def forward(self, x):
        # x shape: (batch, seq_len, input_size)
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Use the last hidden state for prediction
        last_hidden = lstm_out[:, -1, :]  # (batch, hidden_size)
        
        # Predict next prediction_horizon timesteps
        predictions = self.fc(last_hidden)  # (batch, prediction_horizon)
        
        return predictions


class GRUTrafficPredictor(nn.Module):
    """GRU model for traffic congestion forecasting."""
    
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.2, prediction_horizon=10):
        super(GRUTrafficPredictor, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.prediction_horizon = prediction_horizon
        
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, prediction_horizon)
        )
    
    def forward(self, x):
        # x shape: (batch, seq_len, input_size)
        gru_out, h_n = self.gru(x)
        
        # Use the last hidden state for prediction
        last_hidden = gru_out[:, -1, :]  # (batch, hidden_size)
        
        # Predict next prediction_horizon timesteps
        predictions = self.fc(last_hidden)  # (batch, prediction_horizon)
        
        return predictions


class TrafficForecaster:
    """
    Unified interface for traffic prediction using LSTM or GRU.
    Handles training, validation, inference, and model management.
    """
    
    def __init__(self, model_type='lstm', input_size=7, hidden_size=128, num_layers=2,
                 dropout=0.2, lookback_window=50, prediction_horizon=10, device='cpu'):
        """
        Args:
            model_type: 'lstm' or 'gru'
            input_size: Number of input features per timestep
            hidden_size: LSTM/GRU hidden dimension
            num_layers: Number of stacked LSTM/GRU layers
            dropout: Dropout probability
            lookback_window: Number of past timesteps used for prediction
            prediction_horizon: Number of future timesteps to predict
            device: 'cpu' or 'cuda'
        """
        self.model_type = model_type.lower()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.lookback_window = lookback_window
        self.prediction_horizon = prediction_horizon
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Initialize model
        if self.model_type == 'lstm':
            self.model = LSTMTrafficPredictor(
                input_size, hidden_size, num_layers, dropout, prediction_horizon
            ).to(self.device)
        elif self.model_type == 'gru':
            self.model = GRUTrafficPredictor(
                input_size, hidden_size, num_layers, dropout, prediction_horizon
            ).to(self.device)
        else:
            raise ValueError(f"Unknown model_type: {model_type}. Use 'lstm' or 'gru'.")
        
        self.scaler = StandardScaler()
        self.is_fitted = False
        
        self.train_losses = []
        self.val_losses = []
    
    def prepare_sequences(self, data, target_column_idx=0):
        """
        Create sliding window sequences for time-series prediction.
        
        Args:
            data: numpy array of shape (timesteps, features)
            target_column_idx: Index of the feature to predict
        
        Returns:
            X: Input sequences (num_samples, lookback_window, features)
            y: Target sequences (num_samples, prediction_horizon)
        """
        X, y = [], []
        
        for i in range(len(data) - self.lookback_window - self.prediction_horizon + 1):
            X.append(data[i:i + self.lookback_window, :])
            y.append(data[i + self.lookback_window:i + self.lookback_window + self.prediction_horizon, 
                         target_column_idx])
        
        return np.array(X), np.array(y)
    
    def fit(self, train_data, val_data=None, epochs=100, batch_size=32, lr=0.001, 
            target_column_idx=0, patience=10, verbose=True):
        """
        Train the forecasting model.
        
        Args:
            train_data: numpy array (timesteps, features)
            val_data: Optional validation data
            epochs: Number of training epochs
            batch_size: Batch size
            lr: Learning rate
            target_column_idx: Which feature column to predict
            patience: Early stopping patience
            verbose: Print training progress
        """
        # Normalize data
        self.scaler.fit(train_data)
        train_data_scaled = self.scaler.transform(train_data)
        
        # Prepare sequences
        X_train, y_train = self.prepare_sequences(train_data_scaled, target_column_idx)
        
        train_dataset = TrafficDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # Validation data
        val_loader = None
        if val_data is not None:
            val_data_scaled = self.scaler.transform(val_data)
            X_val, y_val = self.prepare_sequences(val_data_scaled, target_column_idx)
            val_dataset = TrafficDataset(X_val, y_val)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Training setup
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            
            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                optimizer.zero_grad()
                predictions = self.model(batch_X)
                loss = criterion(predictions, batch_y)
                loss.backward()
                
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                optimizer.step()
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            self.train_losses.append(train_loss)
            
            # Validation
            if val_loader is not None:
                self.model.eval()
                val_loss = 0.0
                
                with torch.no_grad():
                    for batch_X, batch_y in val_loader:
                        batch_X = batch_X.to(self.device)
                        batch_y = batch_y.to(self.device)
                        predictions = self.model(batch_X)
                        loss = criterion(predictions, batch_y)
                        val_loss += loss.item()
                
                val_loss /= len(val_loader)
                self.val_losses.append(val_loss)
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if patience_counter >= patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch+1}")
                    break
                
                if verbose and (epoch + 1) % 10 == 0:
                    print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
            else:
                if verbose and (epoch + 1) % 10 == 0:
                    print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {train_loss:.6f}")
        
        self.is_fitted = True
        if verbose:
            print(f"✓ {self.model_type.upper()} model training completed")
    
    def predict(self, data):
        """
        Predict future traffic conditions.
        
        Args:
            data: numpy array (timesteps, features) - last lookback_window steps
        
        Returns:
            predictions: numpy array (prediction_horizon,) - future values
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")
        
        self.model.eval()
        
        # Normalize
        data_scaled = self.scaler.transform(data)
        
        # Take last lookback_window timesteps
        if len(data_scaled) < self.lookback_window:
            # Pad if necessary
            padding = np.zeros((self.lookback_window - len(data_scaled), data_scaled.shape[1]))
            data_scaled = np.vstack([padding, data_scaled])
        
        sequence = data_scaled[-self.lookback_window:]
        sequence_tensor = torch.FloatTensor(sequence).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            predictions = self.model(sequence_tensor)
        
        return predictions.cpu().numpy().flatten()
    
    def evaluate(self, test_data, target_column_idx=0):
        """
        Evaluate model on test data.
        
        Returns:
            metrics: dict with MAE, RMSE, MAPE
        """
        test_data_scaled = self.scaler.transform(test_data)
        X_test, y_test = self.prepare_sequences(test_data_scaled, target_column_idx)
        
        test_dataset = TrafficDataset(X_test, y_test)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        self.model.eval()
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X = batch_X.to(self.device)
                predictions = self.model(batch_X)
                all_preds.append(predictions.cpu().numpy())
                all_targets.append(batch_y.numpy())
        
        all_preds = np.vstack(all_preds)
        all_targets = np.vstack(all_targets)
        
        # Calculate metrics
        mae = np.mean(np.abs(all_preds - all_targets))
        rmse = np.sqrt(np.mean((all_preds - all_targets) ** 2))
        
        # MAPE (avoid division by zero)
        mask = all_targets != 0
        mape = np.mean(np.abs((all_targets[mask] - all_preds[mask]) / all_targets[mask])) * 100
        
        return {
            'mae': mae,
            'rmse': rmse,
            'mape': mape,
            'predictions': all_preds,
            'targets': all_targets
        }
    
    def save(self, filepath):
        """Save model checkpoint."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'model_type': self.model_type,
            'input_size': self.input_size,
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'dropout': self.dropout,
            'lookback_window': self.lookback_window,
            'prediction_horizon': self.prediction_horizon,
            'scaler': self.scaler,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }
        
        torch.save(checkpoint, filepath)
        print(f"✓ Model saved to {filepath}")
    
    def load(self, filepath):
        """Load model checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model_type = checkpoint['model_type']
        self.input_size = checkpoint['input_size']
        self.hidden_size = checkpoint['hidden_size']
        self.num_layers = checkpoint['num_layers']
        self.dropout = checkpoint['dropout']
        self.lookback_window = checkpoint['lookback_window']
        self.prediction_horizon = checkpoint['prediction_horizon']
        self.scaler = checkpoint['scaler']
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        
        # Recreate model with loaded parameters
        if self.model_type == 'lstm':
            self.model = LSTMTrafficPredictor(
                self.input_size, self.hidden_size, self.num_layers, 
                self.dropout, self.prediction_horizon
            ).to(self.device)
        else:
            self.model = GRUTrafficPredictor(
                self.input_size, self.hidden_size, self.num_layers,
                self.dropout, self.prediction_horizon
            ).to(self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.is_fitted = True
        
        print(f"✓ Model loaded from {filepath}")
    
    def plot_training_curves(self, output_path='outputs/training_curves.png'):
        """Plot training and validation loss curves."""
        plt.figure(figsize=(10, 5), dpi=300)
        plt.plot(self.train_losses, label='Training Loss', linewidth=2)
        if self.val_losses:
            plt.plot(self.val_losses, label='Validation Loss', linewidth=2)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('MSE Loss', fontsize=12)
        plt.title(f'{self.model_type.upper()} Traffic Forecasting - Training Progress', fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Training curves saved to {output_path}")
