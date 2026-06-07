"""
Fast training script for prediction models.
Optimized for speed with minimal constellation and reduced data collection.
"""
import sys
import time
import yaml
import numpy as np
import os

print("="*70)
print("FAST PREDICTION MODEL TRAINING")
print("="*70)
start_time = time.time()

# Create optimized config
print("\n[1/7] Creating optimized configuration...", flush=True)
config = {
    'simulation': {
        'time_step': 0.1,
        'max_timesteps': 100  # Reduced for speed
    },
    'constellation': {
        'orbit_model': 'circular',  # Avoid TLE downloads
        'layers': [
            {
                'altitude_km': 550,
                'num_planes': 4,  # Minimal constellation
                'sats_per_plane': 5,
                'inclination': 53.0
            }
        ],
        'earth_radius_km': 6371.0,
        'speed_of_light_km_s': 299792.458
    },
    'network': {
        'max_isl_distance_km': 5000.0,
        'nearest_neighbors': 4,
        'link_capacity_mbps': 300.0,
        'queue_capacity_packets': 50
    },
    'routing': {
        'propagation_scale': 0.3,
        'processing_delay_s': 0.0001,
        'queue_delay_per_packet_s': 0.00005
    },
    'traffic': {
        'slices': {
            'embb': {
                'priority': 1,
                'max_latency_ms': 50.0,
                'bandwidth_req_mbps': 100.0,
                'weight': 1.0,
                'packet_size_bytes': 1500
            }
        },
        'generation_rate_packets_per_sec': 20  # Reduced traffic
    },
    'prediction': {
        'hidden_size': 64,  # Smaller model
        'num_layers': 1,
        'dropout': 0.1,
        'lookback_window': 30,  # Shorter lookback
        'prediction_horizon': 5,  # Shorter horizon
        'epochs': 20,  # Fewer epochs
        'batch_size': 16
    }
}

print(f"  ✓ Config ready ({20} satellites, minimal traffic)", flush=True)

# Import modules
print(f"\n[2/7] Importing modules...", flush=True)
from src.env.topology import SatelliteNetwork
from src.traffic.generator import TrafficGenerator
from src.routing.drl_env import SatelliteRoutingEnv
from src.prediction.data_collector import NetworkDataCollector
from src.prediction.traffic_forecaster import TrafficForecaster
print(f"  ✓ Imports complete", flush=True)

# Save temporary config
temp_config_path = 'config/temp_fast_config.yaml'
os.makedirs('config', exist_ok=True)
with open(temp_config_path, 'w') as f:
    yaml.dump(config, f)

# Initialize components
print(f"\n[3/7] Initializing network (20 satellites)...", flush=True)
topology = SatelliteNetwork(config_path=temp_config_path)
print(f"  ✓ Network created: {len(topology.satellites)} satellites", flush=True)

print(f"\n[4/7] Setting up environment...", flush=True)
traffic_gen = TrafficGenerator(config, num_nodes=len(topology.satellites))
env = SatelliteRoutingEnv(topology, traffic_gen, config_path=temp_config_path)
collector = NetworkDataCollector(topology, config, buffer_size=5000)
print(f"  ✓ Environment ready", flush=True)

# Collect training data
print(f"\n[5/7] Collecting training data (3 episodes)...", flush=True)
episodes = 3

for ep in range(episodes):
    obs = env.reset()
    done = False
    step_count = 0
    
    while not done and step_count < 100:  # Max 100 steps per episode
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        current_time = env.current_step * env.dt
        collector.collect_step(env, current_time)
        step_count += 1
    
    print(f"  Episode {ep+1}/{episodes}: {step_count} steps collected", flush=True)

# Export and prepare data
link_df, node_df, global_df = collector.export_to_dataframe()
print(f"\n  ✓ Collected {len(link_df)} link records", flush=True)

if len(link_df) < 100:
    print(f"\n✗ ERROR: Not enough data ({len(link_df)} records). Need at least 100.")
    print("  Try running more episodes or longer simulations.")
    sys.exit(1)

# Prepare training data
print(f"\n[6/7] Preparing training data...", flush=True)
first_link_src = link_df['source'].iloc[0]
first_link_dst = link_df['destination'].iloc[0]
link_data = link_df[(link_df['source'] == first_link_src) & 
                   (link_df['destination'] == first_link_dst)]

feature_columns = ['queue_length', 'utilization', 'bandwidth_used',
                  'packet_drops', 'avg_delay', 'packet_arrival_rate', 'success_rate']

data = link_data[feature_columns].values
print(f"  Total samples: {len(data)}", flush=True)

# Split train/val
train_end = int(len(data) * 0.8)
train_data = data[:train_end]
val_data = data[train_end:]

print(f"  Train: {len(train_data)} samples", flush=True)
print(f"  Val: {len(val_data)} samples", flush=True)

# Train models
print(f"\n[7/7] Training prediction models...", flush=True)
pred_config = config['prediction']

os.makedirs('models', exist_ok=True)
os.makedirs('outputs/prediction', exist_ok=True)

# LSTM
print(f"\n  [A] Training LSTM model...", flush=True)
lstm_model = TrafficForecaster(
    model_type='lstm',
    input_size=7,
    hidden_size=pred_config['hidden_size'],
    num_layers=pred_config['num_layers'],
    dropout=pred_config['dropout'],
    lookback_window=pred_config['lookback_window'],
    prediction_horizon=pred_config['prediction_horizon'],
    device='cpu'
)

lstm_model.fit(
    train_data,
    val_data if len(val_data) > 30 else None,
    epochs=pred_config['epochs'],
    batch_size=pred_config['batch_size'],
    lr=0.001,
    patience=5,
    verbose=True
)

lstm_model.save('models/lstm_traffic_predictor.pth')
print(f"  ✓ LSTM saved to models/lstm_traffic_predictor.pth", flush=True)

# GRU
print(f"\n  [B] Training GRU model...", flush=True)
gru_model = TrafficForecaster(
    model_type='gru',
    input_size=7,
    hidden_size=pred_config['hidden_size'],
    num_layers=pred_config['num_layers'],
    dropout=pred_config['dropout'],
    lookback_window=pred_config['lookback_window'],
    prediction_horizon=pred_config['prediction_horizon'],
    device='cpu'
)

gru_model.fit(
    train_data,
    val_data if len(val_data) > 30 else None,
    epochs=pred_config['epochs'],
    batch_size=pred_config['batch_size'],
    lr=0.001,
    patience=5,
    verbose=True
)

gru_model.save('models/gru_traffic_predictor.pth')
print(f"  ✓ GRU saved to models/gru_traffic_predictor.pth", flush=True)

# Evaluate models
if len(val_data) > 50:
    print(f"\n[EVALUATION] Testing prediction accuracy...", flush=True)
    
    lstm_metrics = lstm_model.evaluate(val_data, target_column_idx=0)
    gru_metrics = gru_model.evaluate(val_data, target_column_idx=0)
    
    print(f"\n  LSTM Performance:")
    print(f"    MAE:  {lstm_metrics['mae']:.4f}")
    print(f"    RMSE: {lstm_metrics['rmse']:.4f}")
    print(f"    MAPE: {lstm_metrics['mape']:.2f}%")
    
    print(f"\n  GRU Performance:")
    print(f"    MAE:  {gru_metrics['mae']:.4f}")
    print(f"    RMSE: {gru_metrics['rmse']:.4f}")
    print(f"    MAPE: {gru_metrics['mape']:.2f}%")
    
    # Quick visualization
    try:
        from src.prediction.visualization import PredictionVisualizer
        visualizer = PredictionVisualizer(output_dir='outputs/prediction')
        
        viz_length = min(100, len(lstm_metrics['targets']))
        actual = lstm_metrics['targets'][:viz_length, 0]
        pred_lstm = lstm_metrics['predictions'][:viz_length, 0]
        pred_gru = gru_metrics['predictions'][:viz_length, 0]
        
        visualizer.plot_prediction_vs_actual(actual, pred_lstm, pred_gru)
        visualizer.plot_metrics_comparison(lstm_metrics, gru_metrics)
        
        print(f"\n  ✓ Visualizations saved to outputs/prediction/")
    except Exception as e:
        print(f"\n  ⚠ Visualization skipped: {e}")

# Cleanup
try:
    os.remove(temp_config_path)
except:
    pass

elapsed = time.time() - start_time
print(f"\n{'='*70}")
print(f"TRAINING COMPLETE in {elapsed:.1f} seconds!")
print(f"{'='*70}")
print(f"\nModels saved:")
print(f"  • models/lstm_traffic_predictor.pth")
print(f"  • models/gru_traffic_predictor.pth")
print(f"\nYou can now use these models for prediction-aware routing.")
