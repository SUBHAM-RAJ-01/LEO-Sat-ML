"""
Unified script for traffic prediction and 3D visualization.
Combines training, evaluation, and visualization in one place.

Usage:
    python run_prediction.py --train           # Train LSTM/GRU models
    python run_prediction.py --visualize       # Generate 3D figures
    python run_prediction.py --demo            # Run complete demo
    python run_prediction.py --all             # Do everything
"""
import argparse
import yaml
import numpy as np
import networkx as nx
import os
from src.env.topology import SatelliteNetwork
from src.traffic.generator import TrafficGenerator
from src.routing.drl_env import SatelliteRoutingEnv
from src.prediction.data_collector import NetworkDataCollector
from src.prediction.visualization import PredictionVisualizer
from src.visualization.satellite_3d import Satellite3DVisualizer
from src.visualization.satellite_3d_advanced import AdvancedSatellite3DVisualizer


# ============================================================================
# TRAINING MODULE
# ============================================================================

def train_models(config, episodes=5):
    """Train LSTM and GRU models on collected traffic data."""
    print(f"\n{'='*60}\nTraining Traffic Prediction Models\n{'='*60}")
    
    # Initialize components
    topology = SatelliteNetwork()
    traffic_gen = TrafficGenerator(config, num_nodes=len(topology.satellites))
    env = SatelliteRoutingEnv(topology, traffic_gen)
    collector = NetworkDataCollector(topology, config, buffer_size=50000)
    
    # Collect data
    print(f"\nCollecting data from {episodes} episodes...")
    for ep in range(episodes):
        obs = env.reset()
        done = False
        step_count = 0
        last_collected_time = -1
        
        while not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            current_time = env.current_step * env.dt
            if current_time > last_collected_time:
                collector.collect_step(env, current_time)
                last_collected_time = current_time
                
            step_count += 1
            if step_count % 5000 == 0:
                print(f"    [Progress] Episode {ep+1}: {step_count} packets processed... (Sim Time: {current_time:.1f}s / {env.max_timesteps * env.dt:.1f}s)", flush=True)
        
        print(f"  Episode {ep+1}/{episodes} completed: {step_count} total packets processed.", flush=True)
    
    # Export data
    link_df, node_df, global_df = collector.export_to_dataframe()
    print(f"\n✓ Collected: {len(link_df)} link records, {len(global_df)} global records")
    
    # Prepare training data
    if len(link_df) == 0:
        print("ERROR: No data collected")
        return None, None
    
    first_link_src = link_df['source'].iloc[0]
    first_link_dst = link_df['destination'].iloc[0]
    link_data = link_df[(link_df['source'] == first_link_src) & 
                       (link_df['destination'] == first_link_dst)]
    
    feature_columns = ['queue_length', 'utilization', 'bandwidth_used',
                      'packet_drops', 'avg_delay', 'packet_arrival_rate', 'success_rate']
    
    data = link_data[feature_columns].values
    train_end = int(len(data) * 0.8)
    train_data = data[:train_end]
    val_data = data[train_end:]
    
    pred_config = config.get('prediction', {})
    from src.prediction.traffic_forecaster import TrafficForecaster
    
    # Train LSTM
    print("\n[1/2] Training LSTM...")
    lstm_model = TrafficForecaster(
        model_type='lstm', input_size=7,
        hidden_size=pred_config.get('hidden_size', 128),
        num_layers=pred_config.get('num_layers', 2),
        dropout=pred_config.get('dropout', 0.2),
        lookback_window=pred_config.get('lookback_window', 50),
        prediction_horizon=pred_config.get('prediction_horizon', 10),
        device='cpu'
    )
    
    lstm_model.fit(train_data, val_data if len(val_data) > 50 else None,
                  epochs=pred_config.get('epochs', 50), batch_size=32, 
                  lr=0.001, patience=10, verbose=True)
    
    os.makedirs('models', exist_ok=True)
    lstm_model.save('models/lstm_traffic_predictor.pth')
    print("✓ LSTM saved to models/lstm_traffic_predictor.pth")
    
    # Train GRU
    print("\n[2/2] Training GRU...")
    gru_model = TrafficForecaster(
        model_type='gru', input_size=7,
        hidden_size=pred_config.get('hidden_size', 128),
        num_layers=pred_config.get('num_layers', 2),
        dropout=pred_config.get('dropout', 0.2),
        lookback_window=pred_config.get('lookback_window', 50),
        prediction_horizon=pred_config.get('prediction_horizon', 10),
        device='cpu'
    )
    
    gru_model.fit(train_data, val_data if len(val_data) > 50 else None,
                 epochs=pred_config.get('epochs', 50), batch_size=32,
                 lr=0.001, patience=10, verbose=True)
    
    gru_model.save('models/gru_traffic_predictor.pth')
    print("✓ GRU saved to models/gru_traffic_predictor.pth")
    
    # Evaluate and visualize
    if len(val_data) > 100:
        print("\nEvaluating models...")
        visualizer = PredictionVisualizer(output_dir='outputs/prediction')
        
        lstm_metrics = lstm_model.evaluate(val_data, target_column_idx=0)
        gru_metrics = gru_model.evaluate(val_data, target_column_idx=0)
        
        print(f"  LSTM - MAE: {lstm_metrics['mae']:.4f}, RMSE: {lstm_metrics['rmse']:.4f}")
        print(f"  GRU  - MAE: {gru_metrics['mae']:.4f}, RMSE: {gru_metrics['rmse']:.4f}")
        
        # Generate plots
        viz_length = min(200, len(lstm_metrics['targets']))
        actual = lstm_metrics['targets'][:viz_length, 0]
        pred_lstm = lstm_metrics['predictions'][:viz_length, 0]
        pred_gru = gru_metrics['predictions'][:viz_length, 0]
        
        visualizer.plot_prediction_vs_actual(actual, pred_lstm, pred_gru)
        visualizer.plot_metrics_comparison(lstm_metrics, gru_metrics)
        
        print("✓ Plots saved to outputs/prediction/")
    
    print(f"\n{'='*60}\nTraining Complete!\n{'='*60}\n")
    return lstm_model, gru_model


# ============================================================================
# VISUALIZATION MODULE
# ============================================================================

def generate_visualizations(config):
    """Generate publication-quality 3D visualizations."""
    print(f"\n{'='*60}\nGenerating Publication-Quality 3D Visualizations\n{'='*60}")
    
    topology = SatelliteNetwork()
    visualizer = AdvancedSatellite3DVisualizer(topology, output_dir='outputs/3d_viz')
    
    # City coordinates
    cities = {
        'Doha': (25.2854, 51.5310), 'London': (51.5074, -0.1278),
        'Tokyo': (35.6762, 139.6503), 'Paris': (48.8566, 2.3522),
        'New York': (40.7128, -74.0060), 'Singapore': (1.3521, 103.8198)
    }
    
    city_pairs = [('Doha', 'London'), ('Tokyo', 'Paris'), ('New York', 'Singapore')]
    
    print("\nGenerating city-to-city routes...")
    routes_for_comparison = {}
    
    for src_city, dst_city in city_pairs:
        src_sat = find_closest_satellite(topology, *cities[src_city])
        dst_sat = find_closest_satellite(topology, *cities[dst_city])
        
        try:
            route = nx.shortest_path(topology.graph, source=src_sat, 
                                    target=dst_sat, weight='delay_s')
            
            # Single route with stats
            visualizer.visualize_route_publication(
                route=route, src_label=src_city, dst_label=dst_city,
                output_filename=f'route_{src_city}_{dst_city}.png',
                figsize=(10, 8), dpi=600, show_constellation=True,
                show_isl=False, show_axes=False
            )
            
            routes_for_comparison[f'{src_city}→{dst_city}'] = {
                'route': route,
                'src': src_city,
                'dst': dst_city
            }
            
            print(f"  ✓ {src_city} → {dst_city} ({len(route)-1} hops)")
        except:
            print(f"  ✗ {src_city} → {dst_city} (no path)")
    
    # Multi-panel comparison figure
    if len(routes_for_comparison) >= 2:
        print("\nGenerating multi-panel comparison figure...")
        visualizer.visualize_multi_route_comparison(
            routes_dict=routes_for_comparison,
            output_filename='comparison_multi_panel.png',
            figsize=(18, 6), dpi=600
        )
    
    print(f"\n✓ All visualizations saved to outputs/3d_viz/\n{'='*60}\n")


def find_closest_satellite(topology, lat, lon, alt_km=500):
    """Find satellite closest to geographic location."""
    earth_radius = 6371.0
    r = earth_radius + alt_km
    lat_rad, lon_rad = np.radians(lat), np.radians(lon)
    
    target_x = r * np.cos(lat_rad) * np.cos(lon_rad)
    target_y = r * np.cos(lat_rad) * np.sin(lon_rad)
    target_z = r * np.sin(lat_rad)
    target_pos = np.array([target_x, target_y, target_z])
    
    min_dist = float('inf')
    closest_sat = None
    
    for sat_id, sat in topology.satellites.items():
        dist = np.linalg.norm(sat.get_position() - target_pos)
        if dist < min_dist:
            min_dist = dist
            closest_sat = sat_id
    
    return closest_sat


# ============================================================================
# DEMO MODULE
# ============================================================================

class PredictionAwareRouter:
    """Router using LSTM/GRU predictions to avoid future congestion."""
    
    def __init__(self, topology, lstm_model=None, gru_model=None, data_collector=None):
        self.topology = topology
        self.lstm_model = lstm_model
        self.gru_model = gru_model
        self.data_collector = data_collector
    
    def predict_future_congestion(self, link_id, model_type='lstm'):
        """Predict future congestion for a link."""
        if not self.data_collector or not (self.lstm_model or self.gru_model):
            return 0.0
        
        history = self.data_collector.get_link_history(link_id, 'queue_length', window_size=50)
        if len(history) < 50:
            return 0.0
        
        features = np.column_stack([
            history,
            self.data_collector.get_link_history(link_id, 'utilization', window_size=50),
            self.data_collector.get_link_history(link_id, 'bandwidth_used', window_size=50),
            self.data_collector.get_link_history(link_id, 'packet_drops', window_size=50),
            self.data_collector.get_link_history(link_id, 'avg_delay', window_size=50),
            self.data_collector.get_link_history(link_id, 'packet_arrival_rate', window_size=50),
            self.data_collector.get_link_history(link_id, 'success_rate', window_size=50)
        ])
        
        model = self.lstm_model if model_type == 'lstm' else self.gru_model
        if model and model.is_fitted:
            prediction = model.predict(features)
            return np.mean(prediction)
        return 0.0
    
    def compute_prediction_aware_route(self, source, destination, model_type='lstm'):
        """Compute route considering predicted congestion."""
        def predictive_cost(u, v, edge_data):
            base_cost = edge_data.get('delay_s', 0.0) * 1000.0
            current_queue = edge_data.get('queue_len', 0)
            predicted_queue = self.predict_future_congestion((u, v), model_type)
            return base_cost + current_queue * 2.0 + predicted_queue * 5.0
        
        try:
            return nx.shortest_path(self.topology.graph, source=source,
                                   target=destination, weight=predictive_cost)
        except:
            return None


def run_demo(config):
    """Run integrated prediction-aware routing demo."""
    print(f"\n{'='*60}\nPrediction-Aware Routing Demo\n{'='*60}")
    
    # Initialize
    topology = SatelliteNetwork()
    traffic_gen = TrafficGenerator(config, num_nodes=len(topology.satellites))
    env = SatelliteRoutingEnv(topology, traffic_gen)
    collector = NetworkDataCollector(topology, config, buffer_size=10000)
    
    print("\nLoading prediction models...")
    try:
        from src.prediction.traffic_forecaster import TrafficForecaster
    except ImportError:
        print("\n⚠ Cannot import TrafficForecaster. Skipping prediction aware routing.\n")
        return
        
    lstm_model, gru_model = None, None
    try:
        lstm_model = TrafficForecaster(model_type='lstm', input_size=7, device='cpu')
        lstm_model.load('models/lstm_traffic_predictor.pth')
        print("  ✓ LSTM loaded")
    except:
        print("  ⚠ LSTM not found")
    
    try:
        gru_model = TrafficForecaster(model_type='gru', input_size=7, device='cpu')
        gru_model.load('models/gru_traffic_predictor.pth')
        print("  ✓ GRU loaded")
    except:
        print("  ⚠ GRU not found")
    
    if not lstm_model and not gru_model:
        print("\n⚠ No models found. Run: python run_prediction.py --train\n")
        return
    
    # Run simulation to collect data
    print("\nRunning simulation...")
    obs = env.reset()
    for _ in range(300):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        collector.collect_step(env, env.current_step * env.dt)
        if terminated or truncated:
            break
    
    # Compare routes
    print("\nComparing routing methods...")
    sat_ids = list(topology.satellites.keys())
    np.random.seed(42)
    source = np.random.choice(sat_ids[:len(sat_ids)//2])
    destination = np.random.choice(sat_ids[len(sat_ids)//2:])
    
    router = PredictionAwareRouter(topology, lstm_model, gru_model, collector)
    
    baseline_route = nx.shortest_path(topology.graph, source=source,
                                     target=destination, weight='delay_s')
    print(f"  Baseline: {len(baseline_route)-1} hops")
    
    if lstm_model and lstm_model.is_fitted:
        lstm_route = router.compute_prediction_aware_route(source, destination, 'lstm')
        if lstm_route:
            print(f"  LSTM-aware: {len(lstm_route)-1} hops")
    
    if gru_model and gru_model.is_fitted:
        gru_route = router.compute_prediction_aware_route(source, destination, 'gru')
        if gru_route:
            print(f"  GRU-aware: {len(gru_route)-1} hops")
    
    # Visualize
    print("\nGenerating visualizations...")
    visualizer_3d = AdvancedSatellite3DVisualizer(topology, output_dir='outputs/demo')
    visualizer_3d.visualize_route_publication(baseline_route, 'Source', 'Destination',
                                             'demo_route.png', dpi=600, show_axes=False)
    
    print(f"✓ Demo complete! Results in outputs/demo/\n{'='*60}\n")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Traffic Prediction & Visualization")
    parser.add_argument('--train', action='store_true', help='Train LSTM/GRU models')
    parser.add_argument('--visualize', action='store_true', help='Generate 3D figures')
    parser.add_argument('--demo', action='store_true', help='Run demo')
    parser.add_argument('--all', action='store_true', help='Do everything')
    parser.add_argument('--episodes', type=int, default=5, help='Training episodes')
    parser.add_argument('--config', type=str, default='config/config.yaml')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Add prediction config if needed
    if 'prediction' not in config:
        config['prediction'] = {
            'hidden_size': 128, 'num_layers': 2, 'dropout': 0.2,
            'lookback_window': 50, 'prediction_horizon': 10,
            'epochs': 50, 'batch_size': 32
        }
    
    # Create directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('outputs/prediction', exist_ok=True)
    os.makedirs('outputs/3d_viz', exist_ok=True)
    os.makedirs('outputs/demo', exist_ok=True)
    
    # Execute requested operations
    if args.all:
        train_models(config, args.episodes)
        generate_visualizations(config)
        run_demo(config)
    else:
        if args.train:
            train_models(config, args.episodes)
        if args.visualize:
            generate_visualizations(config)
        if args.demo:
            run_demo(config)
        
        if not (args.train or args.visualize or args.demo):
            print("\nUsage:")
            print("  python run_prediction.py --train           # Train models")
            print("  python run_prediction.py --visualize       # Generate 3D figures")
            print("  python run_prediction.py --demo            # Run demo")
            print("  python run_prediction.py --all             # Do everything\n")


if __name__ == "__main__":
    main()
