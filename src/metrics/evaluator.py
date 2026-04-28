import numpy as np
import networkx as nx
from src.routing.baseline import RoutingBaseline
from stable_baselines3 import PPO

class SystemEvaluator:
    def __init__(self, env, topology, model_path="models/ppo_routing.zip"):
        self.vec_env = env
        # Handle unpacking SB3 DummyVecEnv
        self.raw_env = env.envs[0].unwrapped if hasattr(env, 'envs') else env
        self.topology = topology
        self.baseline = RoutingBaseline(self.topology)
        
        # Check Topology Connectivity
        components = list(nx.weakly_connected_components(self.topology.graph))
        if len(components) > 1:
            print("\n" + "!" * 50)
            print(f"CRITICAL WARNING: CONSTELLATION IS DISCONNECTED ({len(components)} components)")
            print("Routing PDR will be extremely low as nodes in different components cannot communicate.")
            print("!" * 50 + "\n")
        else:
            print("Successfully verified global constellation connectivity (1 component).")
        
        try:
            self.drl_model = PPO.load(model_path, env=self.vec_env)
            # Verify space compatibility manually to avoid late crashes
            if self.drl_model.observation_space.shape != self.vec_env.observation_space.shape:
                 raise ValueError(f"Shape mismatch: model={self.drl_model.observation_space.shape}, env={self.vec_env.observation_space.shape}")
            self.has_model = True
            print(f"Successfully loaded DRL model from {model_path}.")
        except (Exception, ValueError) as e:
            self.has_model = False
            print(f"Warning: DRL model is incompatible or not found at {model_path} ({e}).")
            print("Evaluation will proceed using baseline algorithms (Dijkstra/OSPF) and Random actions for DRL slot.")
            
    def run_evaluation(self, routing_method="drl"):
        print(f"Running evaluation using {routing_method.upper()}...")
        utilization_history = []
        
        if routing_method == "drl" and self.has_model:
            # Ensure evaluation models don't auto-normalize based on rolling stats
            if hasattr(self.vec_env, 'training'):
                self.vec_env.training = False
                self.vec_env.norm_reward = False
                
            obs, _ = self.raw_env.reset()
            done = False
            
            while not done:
                # Manually normalize to avoid DummyVecEnv organically resetting the tracked variable arrays
                vec_obs = np.array([obs])
                norm_obs = self.vec_env.normalize_obs(vec_obs) if hasattr(self.vec_env, 'normalize_obs') else vec_obs
                
                action, _ = self.drl_model.predict(norm_obs, deterministic=True)
                act_scalar = action.item() if isinstance(action, np.ndarray) else action
                
                obs, reward, terminated, truncated, _ = self.raw_env.step(act_scalar)
                done = terminated or truncated
                
                # Snapshot average queue congestion (scaled against max queue limit from config)
                queue_cap = self.raw_env.config['network']['queue_capacity_packets']
                utils = [min(100.0, (d.get('queue_len', 0) / queue_cap) * 100.0) for u, v, d in self.topology.graph.edges(data=True)]
                if utils: utilization_history.append(np.mean(utils))
        else:
            # Dijkstra and OSPF run on un-normalized environments directly
            obs, _ = self.raw_env.reset()
            done = False
            
            while not done:
                if routing_method == "dijkstra":
                    action = self._get_baseline_action("dijkstra")
                elif routing_method == "ospf":
                    action = self._get_baseline_action("ospf")
                else: # fallback
                    action = self.raw_env.action_space.sample()
                    
                obs, reward, terminated, truncated, _ = self.raw_env.step(action)
                done = terminated or truncated
                
                queue_cap = self.raw_env.config['network']['queue_capacity_packets']
                utils = [min(100.0, (d.get('queue_len', 0) / queue_cap) * 100.0) for u, v, d in self.topology.graph.edges(data=True)]
                if utils: utilization_history.append(np.mean(utils))
                
        self.avg_utilization = np.mean(utilization_history) if utilization_history else 0.0
        return self._calculate_metrics()
        
    def _get_baseline_action(self, method):
        current_pkt = self.raw_env.current_packet
        if not current_pkt or getattr(current_pkt, 'is_dropped', False):
            return self.raw_env.action_space.sample()
            
        current_node = current_pkt.current_node
        dest = current_pkt.destination
        
        if method == "dijkstra":
            next_hop = self.baseline.get_next_hop_dijkstra(current_node, dest)
        else:
            next_hop = self.baseline.get_next_hop_ospf(current_node, dest)
            
        if next_hop is None:
            return self.raw_env.action_space.sample()
            
        neighbors = list(self.topology.graph.successors(current_node))
        try:
            return neighbors.index(next_hop)
        except ValueError:
            return self.raw_env.action_space.sample()
            
    def _calculate_metrics(self):
        delivered = len(self.raw_env.delivered_packets)
        dropped = self.raw_env.dropped_packets
        total = delivered + dropped
        pdr = (delivered / total) * 100 if total > 0 else 0.0
        
        latency_by_slice = {}
        qos_by_slice = {}
        jitter_by_slice = {}
        throughput_by_slice = {}
        
        hop_counts = [len(p.path) for p in self.raw_env.delivered_packets]
        avg_hop = float(np.mean(hop_counts)) if hop_counts else 0.0
        
        sim_time_s = max(self.raw_env.current_step * self.raw_env.dt, 1e-6)
        
        if delivered > 0:
            slice_pkts = {}
            for p in self.raw_env.delivered_packets:
                if p.slice_type not in slice_pkts:
                    slice_pkts[p.slice_type] = []
                slice_pkts[p.slice_type].append(p)
                
            for s_type, pkts in slice_pkts.items():
                delays_ms = [p.accumulated_delay * 1000.0 for p in pkts]
                latency_by_slice[s_type] = np.mean(delays_ms) 
                jitter_by_slice[s_type] = np.std(delays_ms) if len(delays_ms) > 1 else 0.0
                
                qos_met = sum(1 for p in pkts if p.accumulated_delay <= p.max_latency)
                qos_by_slice[s_type] = (qos_met / len(pkts)) * 100.0
                
                total_bytes = sum(getattr(p, 'size_bytes', 1500) for p in pkts)
                throughput_by_slice[s_type] = (total_bytes * 8) / 1000000.0 / sim_time_s
                
        metrics = {
            "pdr": pdr,
            "latency_ms": latency_by_slice,
            "jitter_ms": jitter_by_slice,
            "throughput_mbps": throughput_by_slice,
            "qos_satisfaction": qos_by_slice,
            "avg_bandwidth_utilization": self.avg_utilization,
            "avg_hop_count": avg_hop,
            "total_delivered": delivered,
            "total_dropped": dropped,
            "drop_reasons": {
                "loops": getattr(self.raw_env, 'dropped_loops', 0),
                "max_hops": getattr(self.raw_env, 'dropped_max_hops', 0),
                "invalid": getattr(self.raw_env, 'dropped_invalid', 0),
                "other": dropped - (getattr(self.raw_env, 'dropped_loops', 0) + getattr(self.raw_env, 'dropped_max_hops', 0) + getattr(self.raw_env, 'dropped_invalid', 0))
            }
        }
        return metrics
