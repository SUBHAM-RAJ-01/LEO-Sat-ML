import numpy as np
import networkx as nx
from src.routing.baseline import RoutingBaseline
from stable_baselines3 import PPO

class SystemEvaluator:
    def __init__(self, env, topology, model_path="models/ppo_routing.zip"):
        self.vec_env = env
        self.raw_env = env.envs[0].unwrapped if hasattr(env, 'envs') else env
        self.topology = topology
        self.baseline = RoutingBaseline(self.topology)

        components = list(nx.weakly_connected_components(self.topology.graph))
        if len(components) > 1:
            print("\n" + "!" * 50)
            print(f"CRITICAL WARNING: CONSTELLATION IS DISCONNECTED ({len(components)} components)")
            print("!" * 50 + "\n")
        else:
            print("Successfully verified global constellation connectivity (1 component).")

        self.model_path = model_path
        self.has_model = False
        self.drl_model = None

        # Set VecNormalize to evaluation mode
        if hasattr(self.vec_env, 'training'):
            self.vec_env.training = False
            self.vec_env.norm_reward = False
            print("VecNormalize set to evaluation mode.")

        # Try to load the model
        try:
            # Check if model file exists
            import os
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")
            
            self.drl_model = PPO.load(model_path, env=self.vec_env)
            
            # Verify compatibility
            if self.drl_model.observation_space.shape != self.vec_env.observation_space.shape:
                raise ValueError(
                    f"Shape mismatch: model={self.drl_model.observation_space.shape}, "
                    f"env={self.vec_env.observation_space.shape}"
                )
            if self.drl_model.action_space.n != self.raw_env.action_space.n:
                raise ValueError(
                    f"Action space mismatch: model={self.drl_model.action_space.n}, "
                    f"env={self.raw_env.action_space.n}. Retrain required."
                )
            
            self.has_model = True
            print(f"[OK] Successfully loaded DRL model from {model_path}")
            
        except Exception as e:
            print(f"[!] Warning: DRL model incompatible or missing ({e})")
            print("  DRL evaluation will use delay-optimal heuristic fallback until retrained.")
            self.has_model = False

    def run_evaluation(self, routing_method="drl"):
        print(f"Running evaluation using {routing_method.upper()}...")
        utilization_history = []

        # Configure the raw environment for the correct routing mode
        self.raw_env.drl_training = False
        self.raw_env.drl_use_supervisor = (routing_method == 'drl')
        self.raw_env.eval_routing_mode = None if routing_method == 'drl' else routing_method

        # Step the RAW env directly to preserve packet metrics
        # (DummyVecEnv auto-resets on done=True, which clears all counters)
        obs, _ = self.raw_env.reset()
        done = False

        while not done:
            if routing_method == "drl" and self.has_model:
                # Normalize observation using VecNormalize stats before model prediction
                obs_array = np.array([obs])
                if hasattr(self.vec_env, 'normalize_obs'):
                    norm_obs = self.vec_env.normalize_obs(obs_array)
                else:
                    norm_obs = obs_array
                
                # Get action from model using normalized observations
                action, _ = self.drl_model.predict(norm_obs, deterministic=True)
                act_scalar = int(action.item() if isinstance(action, np.ndarray) else action)
            elif routing_method == "drl":
                # No model loaded, pass action=0 (will use DRL fallback in env)
                act_scalar = 0
            else:
                # For baselines, action is ignored - env follows pre-computed path
                act_scalar = 0

            obs, reward, terminated, truncated, _ = self.raw_env.step(act_scalar)
            done = terminated or truncated

            queue_cap = self.raw_env.config['network']['queue_capacity_packets']
            utils = [
                min(100.0, (d.get('queue_len', 0) / queue_cap) * 100.0)
                for _, _, d in self.topology.graph.edges(data=True)
            ]
            if utils:
                utilization_history.append(np.mean(utils))

        self.raw_env.eval_routing_mode = None
        self.raw_env.drl_use_supervisor = False
        self.avg_utilization = np.mean(utilization_history) if utilization_history else 0.0
        return self._calculate_metrics()

    def _oracle_meta_action(self, obs):
        """Pick cheaper expert using cost features embedded in observation."""
        if obs is None or len(obs) < 19:
            return 0
        cost_adv = float(obs[18])
        return 0 if cost_adv <= 0 else 1

    def _calculate_metrics(self):
        delivered = len(self.raw_env.delivered_packets)
        dropped = self.raw_env.dropped_packets
        total = delivered + dropped
        pdr = (delivered / total) * 100 if total > 0 else 0.0

        latency_by_slice = {}
        latency_p95_by_slice = {}
        qos_by_slice = {}
        jitter_by_slice = {}
        throughput_by_slice = {}
        pdr_by_slice = {}

        hop_counts = [len(p.path) - 1 for p in self.raw_env.delivered_packets]
        avg_hop = float(np.mean(hop_counts)) if hop_counts else 0.0
        sim_time_s = max(self.raw_env.current_step * self.raw_env.dt, 1e-6)
        all_latencies_ms = []

        slice_pkts = {}
        for p in self.raw_env.delivered_packets:
            if p.slice_type not in slice_pkts:
                slice_pkts[p.slice_type] = []
            slice_pkts[p.slice_type].append(p)
            all_latencies_ms.append(p.accumulated_delay * 1000.0)

        dropped_by_slice = getattr(self.raw_env, 'dropped_by_slice', {})
        delivered_by_slice = getattr(self.raw_env, 'delivered_by_slice', {})

        all_slices = set(list(slice_pkts.keys()) + list(dropped_by_slice.keys()) + list(delivered_by_slice.keys()))
        for s_type in all_slices:
            pkts = slice_pkts.get(s_type, [])
            if pkts:
                delays_ms = [p.accumulated_delay * 1000.0 for p in pkts]
                latency_by_slice[s_type] = float(np.mean(delays_ms))
                latency_p95_by_slice[s_type] = float(np.percentile(delays_ms, 95))
                jitter_by_slice[s_type] = float(np.std(delays_ms)) if len(delays_ms) > 1 else 0.0
                qos_met = sum(1 for p in pkts if p.accumulated_delay <= p.max_latency)
                qos_by_slice[s_type] = (qos_met / len(pkts)) * 100.0
                total_bytes = sum(getattr(p, 'size_bytes', 1500) for p in pkts)
                throughput_by_slice[s_type] = (total_bytes * 8) / 1_000_000.0 / sim_time_s

            slice_delivered = delivered_by_slice.get(s_type, len(pkts))
            slice_dropped = dropped_by_slice.get(s_type, 0)
            slice_total = slice_delivered + slice_dropped
            pdr_by_slice[s_type] = (slice_delivered / slice_total * 100.0) if slice_total > 0 else 0.0

        avg_latency = float(np.mean(all_latencies_ms)) if all_latencies_ms else 0.0
        p95_latency = float(np.percentile(all_latencies_ms, 95)) if all_latencies_ms else 0.0

        return {
            "pdr": pdr,
            "avg_latency": avg_latency,
            "p95_latency": p95_latency,
            "latency_ms": latency_by_slice,
            "latency_p95_ms": latency_p95_by_slice,
            "jitter_ms": jitter_by_slice,
            "throughput_mbps": throughput_by_slice,
            "qos_satisfaction": qos_by_slice,
            "pdr_by_slice": pdr_by_slice,
            "avg_bandwidth_utilization": self.avg_utilization,
            "avg_hop_count": avg_hop,
            "routing_loops": getattr(self.raw_env, 'routing_loops_detected', 0),
            "total_delivered": delivered,
            "total_dropped": dropped,
            "drop_reasons": {
                "loops": getattr(self.raw_env, 'dropped_loops', 0),
                "max_hops": getattr(self.raw_env, 'dropped_max_hops', 0),
                "invalid": getattr(self.raw_env, 'dropped_invalid', 0),
                "congestion": getattr(self.raw_env, 'dropped_congestion', 0),
                "qos_timeout": getattr(self.raw_env, 'dropped_qos_timeout', 0),
            },
        }
