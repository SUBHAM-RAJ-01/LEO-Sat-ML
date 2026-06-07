"""
Full-scale training for 248 satellites with realistic traffic.
This matches your actual constellation size.
"""
import sys
import time
import yaml
import numpy as np
import os

print("="*70)
print("FULL-SCALE PREDICTION MODEL TRAINING (248 SATELLITES)")
print("="*70)
start_time = time.time()

# Load actual config
print("\n[1/7] Loading full configuration...", flush=True)
with open('config/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Override to use circular orbit (avoid TLE download delays)
original_orbit = config['constellation'].get('orbit_model', 'tle')
config['constellation']['orbit_model'] = 'circular'

# Calculate satellite count
total_sats = sum([layer['num_planes'] * layer['sats_per_plane'] 
                  for layer in config['constellation']['layers']])
print(f"  ✓ Configuration loaded: {total_sats} satellites", flush=True)
print(f"  ✓ Orbit model: circular (optimized)", flush=True)

# Save temporary config
temp_config_path = 'config/temp_full_config.yaml'
with open(temp_config_path, 'w') as f:
    yaml.dump(config, f)

# Import modules
print(f"\n[2/7] Importing modules...", flush=True)
from src.env.topology import SatelliteNetwork
from src.traffic.generator import TrafficGenerator
from src.routing.drl_env import SatelliteRoutingEnv
from src.prediction.data_collector import NetworkDataCollector
from src.prediction.traffic_forecaster import TrafficForecaster
from src.prediction.visualization import PredictionVisualizer
print(f"  ✓ Imports complete", flush=True)

# Initialize components
print(f"\n[3/7] Initializing network ({total_sats} satellites)...", flush=True)
print(f"  This may take 30-60 seconds...", flush=True)
init_start = time.time()
topology = SatelliteNetwork(config_path=temp_config_path)
init_time = time.time() - init_start
print(f"  ✓ Network created: {len(topology.satellites)} satellites in {init_time:.1f}s", flush=True)

print(f"\n[4/7] Setting up environment and data collection...", flush=True)
traffic_gen = TrafficGenerator(config, num_nodes=len(topology.satellites))
env = SatelliteRoutingEnv(topology, traffic_gen, config_path=temp_config_path)
collector = NetworkDataCollector(topology, config, buffer_size=50000)
print(f"  ✓ Environment ready", flush=True)

# Collect training data
episodes = 5
print(f"\n[5/7] Collecting training data ({episodes} episodes)...", flush=True)
print(f"  This is the longest step - collecting realistic traffic patterns", flush=True)

for ep in range(episodes):
    ep_start = time.time()
    obs = env.reset()
    done = False
    step_count = 0
    max_steps = 150  # Reasonable episode length
    
    while not done and step_count < max_steps:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        current_time = env.current_step * env.dt
        collector.collect_step(env, current_time)
        step_count += 1
        
        # Progress indicator
        if step_count % 50 == 0:
            print(f"    Episode {ep+1}/{episodes}: {step_count} steps...", flush=True)
    
    ep_time = time.time() - ep_start
    print(f"  ✓ Episode {ep+1}/{episodes}: {step_count} steps in {ep_time:.1f}s", flush=True)

# Export and prepare data
link_df, node_df, global_df = collector.export_to_dataframe()
print(f"\n  ✓ Collected {len(link_df)} link records", flush=True)

if len(link_df) < 100:
    print(f"\n✗ ERROR: Not enough data ({len(link_df)} records). Need at least 100.")
    sys.exit(1)

# Prepare training data
print(f"\n[6/7] Preparing training data...", flush=True)

# Get most active link for training
link_counts = link_df.groupby(['source', 'destination']).size()
best_link = link_counts.idxmax()
print(f"  Using most active link: {best_link[0]} -> {best_link[1]}", flush=True)

link_data = link_df[(link_df['source'] == best_link[0]) & 
                   (link_df['destination'] == best_link[1])]

feature_columns = ['queue_length', 'utilization', 'bandwidth_used',
                  'packet_drops', 'avg_delay', 'packet_arrival_rate', 'success_rate']

data = link_data[feature_columns].values
print(f"  Total samples for training: {len(data)}", flush=True)

# Split train/val
train_end = int(len(data) * 0.8)
train_data = data[:train_end]
val_data = data[train_end:]

print(f"  Train: {len(train_data)} samples", flush=True)
print(f"  Validation: {len(val_data)} samples", flush=True)

# Train models
print(f"\n[7/7] Training prediction models...", flush=True)
pred_config = config.get('prediction', {})
# Set defaults
pred_config.setdefault('hidden_size', 128)
pred_config.setdefault('num_layers', 2)
pred_config.setdefault('dropout', 0.2)
pred_config.setdefault('lookback_window', 50)
pred_config.setdefault('prediction_horizon', 10)
pred_config.setdefault('epochs', 50)
pred_config.setdefault('batch_size', 32)

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
    val_data if len(val_data) > 50 else None,
    epochs=pred_config['epochs'],
    batch_size=pred_config['batch_size'],
    lr=0.001,
    patience=10,
    verbose=True
)

lstm_model.save('models/lstm_traffic_predictor.pth')
print(f"\n  ✓ LSTM saved to models/lstm_traffic_predictor.pth", flush=True)

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
    val_data if len(val_data) > 50 else None,
    epochs=pred_config['epochs'],
    batch_size=pred_config['batch_size'],
    lr=0.001,
    patience=10,
    verbose=True
)

gru_model.save('models/gru_traffic_predictor.pth')
print(f"\n  ✓ GRU saved to models/gru_traffic_predictor.pth", flush=True)

# Evaluate models
if len(val_data) > 100:
    print(f"\n{'='*70}")
    print("EVALUATION RESULTS")
    print(f"{'='*70}")
    
    lstm_metrics = lstm_model.evaluate(val_data, target_column_idx=0)
    gru_metrics = gru_model.evaluate(val_data, target_column_idx=0)
    
    print(f"\nLSTM Performance:")
    print(f"  MAE:  {lstm_metrics['mae']:.4f}")
    print(f"  RMSE: {lstm_metrics['rmse']:.4f}")
    print(f"  MAPE: {lstm_metrics['mape']:.2f}%")
    
    print(f"\nGRU Performance:")
    print(f"  MAE:  {gru_metrics['mae']:.4f}")
    print(f"  RMSE: {gru_metrics['rmse']:.4f}")
    print(f"  MAPE: {gru_metrics['mape']:.2f}%")
    
    # Determine winner
    if gru_metrics['mae'] < lstm_metrics['mae']:
        print(f"\n🏆 Winner: GRU (lower error)")
    else:
        print(f"\n🏆 Winner: LSTM (lower error)")
    
    # Generate visualizations
    try:
        print(f"\nGenerating visualizations...", flush=True)
        visualizer = PredictionVisualizer(output_dir='outputs/prediction')
        
        viz_length = min(200, len(lstm_metrics['targets']))
        actual = lstm_metrics['targets'][:viz_length, 0]
        pred_lstm = lstm_metrics['predictions'][:viz_length, 0]
        pred_gru = gru_metrics['predictions'][:viz_length, 0]
        
        visualizer.plot_prediction_vs_actual(actual, pred_lstm, pred_gru)
        visualizer.plot_metrics_comparison(lstm_metrics, gru_metrics)
        
        # Error distribution
        errors_lstm = lstm_metrics['targets'] - lstm_metrics['predictions']
        errors_gru = gru_metrics['targets'] - gru_metrics['predictions']
        visualizer.plot_error_distribution(errors_lstm, errors_gru)
        
        print(f"✓ Visualizations saved to outputs/prediction/", flush=True)
    except Exception as e:
        print(f"⚠ Visualization error: {e}", flush=True)

# Cleanup
try:
    os.remove(temp_config_path)
except:
    pass

elapsed = time.time() - start_time
print(f"\n{'='*70}")
print(f"TRAINING COMPLETE!")
print(f"{'='*70}")
print(f"\nTotal time: {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
print(f"Satellites: {total_sats}")
print(f"Training samples: {len(train_data)}")
print(f"\nModels saved:")
print(f"  • models/lstm_traffic_predictor.pth")
print(f"  • models/gru_traffic_predictor.pth")
print(f"\nThese models are now ready for use with DRL routing!")
