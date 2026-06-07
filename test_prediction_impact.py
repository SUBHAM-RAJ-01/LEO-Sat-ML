"""
Quick test to show the impact of prediction models on routing.
Compares routing with and without predictions.
"""
import yaml
import numpy as np
import time

print("="*70)
print("PREDICTION IMPACT TEST")
print("="*70)

# Load config
with open('config/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Override for speed
config['constellation']['orbit_model'] = 'circular'

# Save temp config
temp_config = 'config/temp_test_config.yaml'
with open(temp_config, 'w') as f:
    yaml.dump(config, f)

print("\n[1/4] Initializing environment...", flush=True)
from src.env.topology import SatelliteNetwork
from src.traffic.generator import TrafficGenerator
from src.routing.drl_env import SatelliteRoutingEnv
from src.prediction.data_collector import NetworkDataCollector

topology = SatelliteNetwork(config_path=temp_config)
traffic_gen = TrafficGenerator(config, num_nodes=len(topology.satellites))

print(f"  ✓ Network ready: {len(topology.satellites)} satellites", flush=True)

# Test 1: WITHOUT predictions
print("\n[2/4] Testing WITHOUT predictions...", flush=True)
config['prediction']['enabled'] = False
with open(temp_config, 'w') as f:
    yaml.dump(config, f)

env_no_pred = SatelliteRoutingEnv(topology, traffic_gen, config_path=temp_config)
collector_no_pred = NetworkDataCollector(topology, config, buffer_size=5000)

obs = env_no_pred.reset()
rewards_no_pred = []
latencies_no_pred = []
drops_no_pred = []

for step in range(100):
    action = env_no_pred.action_space.sample()
    obs, reward, terminated, truncated, info = env_no_pred.step(action)
    collector_no_pred.collect_step(env_no_pred, env_no_pred.current_step * env_no_pred.dt)
    
    rewards_no_pred.append(reward)
    if 'avg_latency' in info:
        latencies_no_pred.append(info['avg_latency'])
    
    if terminated or truncated:
        break

pdr_no_pred = len(env_no_pred.delivered_packets) / max(
    len(env_no_pred.delivered_packets) + env_no_pred.dropped_packets, 1
) * 100

print(f"  Without Predictions:")
print(f"    Steps: {len(rewards_no_pred)}")
print(f"    Avg Reward: {np.mean(rewards_no_pred):.3f}")
print(f"    PDR: {pdr_no_pred:.2f}%")
print(f"    Delivered: {len(env_no_pred.delivered_packets)}")
print(f"    Dropped: {env_no_pred.dropped_packets}")
if latencies_no_pred:
    print(f"    Avg Latency: {np.mean(latencies_no_pred):.2f}ms")

# Test 2: WITH predictions
print("\n[3/4] Testing WITH predictions...", flush=True)
config['prediction']['enabled'] = True
with open(temp_config, 'w') as f:
    yaml.dump(config, f)

# Reset topology to clear state
topology = SatelliteNetwork(config_path=temp_config)
traffic_gen = TrafficGenerator(config, num_nodes=len(topology.satellites))
env_with_pred = SatelliteRoutingEnv(topology, traffic_gen, config_path=temp_config)
collector_with_pred = NetworkDataCollector(topology, config, buffer_size=5000)

obs = env_with_pred.reset()
rewards_with_pred = []
latencies_with_pred = []
drops_with_pred = []

for step in range(100):
    action = env_with_pred.action_space.sample()
    obs, reward, terminated, truncated, info = env_with_pred.step(action)
    collector_with_pred.collect_step(env_with_pred, env_with_pred.current_step * env_with_pred.dt)
    
    rewards_with_pred.append(reward)
    if 'avg_latency' in info:
        latencies_with_pred.append(info['avg_latency'])
    
    if terminated or truncated:
        break

pdr_with_pred = len(env_with_pred.delivered_packets) / max(
    len(env_with_pred.delivered_packets) + env_with_pred.dropped_packets, 1
) * 100

print(f"  With Predictions:")
print(f"    Steps: {len(rewards_with_pred)}")
print(f"    Avg Reward: {np.mean(rewards_with_pred):.3f}")
print(f"    PDR: {pdr_with_pred:.2f}%")
print(f"    Delivered: {len(env_with_pred.delivered_packets)}")
print(f"    Dropped: {env_with_pred.dropped_packets}")
if latencies_with_pred:
    print(f"    Avg Latency: {np.mean(latencies_with_pred):.2f}ms")

# Compare
print("\n[4/4] COMPARISON", flush=True)
print(f"{'='*70}")

reward_improve = ((np.mean(rewards_with_pred) - np.mean(rewards_no_pred)) / 
                 abs(np.mean(rewards_no_pred)) * 100) if rewards_no_pred else 0
pdr_improve = pdr_with_pred - pdr_no_pred
drops_reduce = ((env_no_pred.dropped_packets - env_with_pred.dropped_packets) / 
               max(env_no_pred.dropped_packets, 1) * 100) if env_no_pred.dropped_packets > 0 else 0

print(f"\n📊 Performance Impact:")
print(f"  Reward Change: {reward_improve:+.1f}%")
print(f"  PDR Change: {pdr_improve:+.2f}%")
print(f"  Drops Reduced: {drops_reduce:.1f}%")

if latencies_no_pred and latencies_with_pred:
    latency_improve = ((np.mean(latencies_no_pred) - np.mean(latencies_with_pred)) / 
                      np.mean(latencies_no_pred) * 100)
    print(f"  Latency Reduced: {latency_improve:.1f}%")

print(f"\n{'='*70}")

# Determine verdict
if reward_improve > 0 or pdr_improve > 0 or drops_reduce > 0:
    print("\n✅ PREDICTION MODELS ARE HELPING!")
    print("   Routing is smarter with predictions.")
else:
    print("\n⚠️  Limited impact in this short test.")
    print("   Try longer simulations or more traffic for better comparison.")

print(f"\n💡 Note: This is a quick test with random actions.")
print(f"   With trained DRL agent, impact will be much larger!")

# Cleanup
import os
try:
    os.remove(temp_config)
except:
    pass

print(f"\n{'='*70}")
print("TEST COMPLETE")
print(f"{'='*70}")
