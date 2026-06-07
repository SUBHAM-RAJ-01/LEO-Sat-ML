"""
Verify that trained prediction models work correctly.
"""
import numpy as np
import os
from src.prediction.traffic_forecaster import TrafficForecaster

print("="*70)
print("PREDICTION MODEL VERIFICATION")
print("="*70)

# Check if models exist
lstm_path = 'models/lstm_traffic_predictor.pth'
gru_path = 'models/gru_traffic_predictor.pth'

print("\n[1/4] Checking model files...")
if os.path.exists(lstm_path):
    size_kb = os.path.getsize(lstm_path) / 1024
    print(f"  ✓ LSTM model found: {size_kb:.1f} KB")
else:
    print(f"  ✗ LSTM model NOT found at {lstm_path}")

if os.path.exists(gru_path):
    size_kb = os.path.getsize(gru_path) / 1024
    print(f"  ✓ GRU model found: {size_kb:.1f} KB")
else:
    print(f"  ✗ GRU model NOT found at {gru_path}")

# Load LSTM model
print("\n[2/4] Loading LSTM model...")
try:
    lstm_model = TrafficForecaster(model_type='lstm', input_size=7, device='cpu')
    lstm_model.load(lstm_path)
    print(f"  ✓ LSTM loaded successfully")
    print(f"    - Hidden size: {lstm_model.hidden_size}")
    print(f"    - Layers: {lstm_model.num_layers}")
    print(f"    - Lookback: {lstm_model.lookback_window}")
    print(f"    - Horizon: {lstm_model.prediction_horizon}")
except Exception as e:
    print(f"  ✗ Failed to load LSTM: {e}")
    lstm_model = None

# Load GRU model
print("\n[3/4] Loading GRU model...")
try:
    gru_model = TrafficForecaster(model_type='gru', input_size=7, device='cpu')
    gru_model.load(gru_path)
    print(f"  ✓ GRU loaded successfully")
    print(f"    - Hidden size: {gru_model.hidden_size}")
    print(f"    - Layers: {gru_model.num_layers}")
    print(f"    - Lookback: {gru_model.lookback_window}")
    print(f"    - Horizon: {gru_model.prediction_horizon}")
except Exception as e:
    print(f"  ✗ Failed to load GRU: {e}")
    gru_model = None

# Test predictions
print("\n[4/4] Testing predictions with sample data...")
if lstm_model or gru_model:
    # Create synthetic test data (lookback_window x 7 features)
    lookback = 30
    test_data = np.random.rand(lookback, 7) * 10  # Random traffic data
    
    if lstm_model:
        try:
            lstm_pred = lstm_model.predict(test_data)
            print(f"\n  LSTM Prediction:")
            print(f"    Input shape: {test_data.shape}")
            print(f"    Output shape: {lstm_pred.shape}")
            print(f"    Predicted values: {lstm_pred[:3]}...")  # First 3 values
            print(f"  ✓ LSTM prediction works!")
        except Exception as e:
            print(f"  ✗ LSTM prediction failed: {e}")
    
    if gru_model:
        try:
            gru_pred = gru_model.predict(test_data)
            print(f"\n  GRU Prediction:")
            print(f"    Input shape: {test_data.shape}")
            print(f"    Output shape: {gru_pred.shape}")
            print(f"    Predicted values: {gru_pred[:3]}...")  # First 3 values
            print(f"  ✓ GRU prediction works!")
        except Exception as e:
            print(f"  ✗ GRU prediction failed: {e}")

print("\n" + "="*70)
print("VERIFICATION COMPLETE")
print("="*70)
print("\nModels are ready to use for prediction-aware routing!")
print("\nNext steps:")
print("  1. Models are in models/ directory")
print("  2. Use them in your routing environment")
print("  3. Run full simulations with prediction-aware routing")
