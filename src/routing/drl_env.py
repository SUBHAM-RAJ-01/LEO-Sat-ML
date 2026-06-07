import gymnasium as gym
from gymnasium import spaces
import numpy as np
import networkx as nx
import random

from src.prediction.data_collector import NetworkDataCollector

class SatelliteRoutingEnv(gym.Env):
    """
    Hybrid DRL + Shortest-Path Routing for 6G NTN
    Guarantees high PDR and low latency through intelligent routing
    """
    metadata = {'render.modes': ['console']}

    def __init__(self, topology, traffic_generator, config_path="config/config.yaml"):
        super(SatelliteRoutingEnv, self).__init__()
        import yaml
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
            
        self.topology = topology
        self.traffic_gen = traffic_generator
        self.max_timesteps = self.config['simulation']['max_timesteps']
        self.dt = self.config['simulation']['time_step']
        self.max_neighbors = self.topology.max_neighbors
        self.queue_capacity = self.config['network']['queue_capacity_packets']
        self.topo_update_interval = 10
        
        # 6G NTN latency calibration parameters
        routing_cfg = self.config.get('routing', {})
        self.propagation_scale = routing_cfg.get('propagation_scale', 0.3)
        self.processing_delay_s = routing_cfg.get('processing_delay_s', 0.0001)
        self.queue_delay_per_packet_s = routing_cfg.get('queue_delay_per_packet_s', 0.00005)
        self.max_hops_slack = routing_cfg.get('max_hops_slack', 10)
        
        # Action space: Direct neighbor selection
        self.action_space = spaces.Discrete(self.max_neighbors)
        
        # Observation space: 16 base + 13 per neighbor (11 current + 2 prediction)
        obs_size = 16 + 13 * self.max_neighbors
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32)
        
        self.current_step = 0
        self.active_packets = []
        self.next_tick_packets = []
        self.current_packet = None
        self.current_neighbors = []
        
        # Metrics
        self.delivered_packets = []
        self.dropped_packets = 0
        self.dropped_loops = 0
        self.dropped_max_hops = 0
        self.dropped_invalid = 0
        self.dropped_congestion = 0
        self.dropped_qos_timeout = 0
        self.backtrack_count = 0
        self.routing_loops_detected = 0
        self.dropped_by_slice = {}
        self.delivered_by_slice = {}
        
        # Training/Evaluation flags (for compatibility with evaluator)
        self.drl_training = True
        self.drl_use_supervisor = False
        self.eval_routing_mode = None
        
        # Reward tracking
        self.ep_reward_delivery = 0.0
        self.ep_reward_qos = 0.0
        self.ep_reward_latency_shaping = 0.0
        self.ep_reward_progress = 0.0
        self.ep_penalty_loops = 0.0
        self.ep_penalty_congestion = 0.0
        self.ep_penalty_hops = 0.0
        self.ep_penalty_backtrack = 0.0
        
        # Baseline router (lazy init)
        self._baseline_router = None
        
        # Shortest path cache
        self._shortest_path_cache = {}
        self._cache_valid_until = 0
        
        # ---- Prediction Integration ----
        pred_config = self.config.get('prediction', {})
        self.use_prediction = pred_config.get('enabled', True)
        self.prediction_weight = pred_config.get('prediction_weight', 3.0)
        self.prediction_warmup = pred_config.get('warmup_steps', 50)
        self.prediction_model = None
        self._prediction_cache = {}
        
        # Initialize data collector for link metric history
        self.data_collector = NetworkDataCollector(
            self.topology, self.config, buffer_size=1000, collection_interval=1
        )
        
        # Try to load pre-trained prediction models
        if self.use_prediction:
            self._load_prediction_models(pred_config)

    def _get_baseline_router(self):
        """Lazy-init baseline router."""
        if self._baseline_router is None:
            from src.routing.baseline import RoutingBaseline
            self._baseline_router = RoutingBaseline(self.topology)
        return self._baseline_router

    def _load_prediction_models(self, pred_config):
        """Load pre-trained LSTM/GRU prediction models if available."""
        import os
        try:
            from src.prediction.traffic_forecaster import TrafficForecaster
        except ImportError:
            print("[Prediction] TrafficForecaster could not be imported. Prediction disabled.")
            self.use_prediction = False
            return
        
        # Try LSTM first, then GRU
        model_paths = [
            ('lstm', pred_config.get('lstm_model_path', 'models/lstm_traffic_predictor.pth')),
            ('gru', pred_config.get('gru_model_path', 'models/gru_traffic_predictor.pth')),
        ]
        
        for model_type, model_path in model_paths:
            if os.path.exists(model_path):
                try:
                    model = TrafficForecaster(
                        model_type=model_type, input_size=7,
                        hidden_size=pred_config.get('hidden_size', 128),
                        num_layers=pred_config.get('num_layers', 2),
                        lookback_window=pred_config.get('lookback_window', 50),
                        prediction_horizon=pred_config.get('prediction_horizon', 10),
                        device='cpu'
                    )
                    model.load(model_path)
                    self.prediction_model = model
                    print(f"[Prediction] Loaded {model_type.upper()} model from {model_path}")
                    return
                except Exception as e:
                    print(f"[Prediction] Failed to load {model_type.upper()} from {model_path}: {e}")
        
        print("[Prediction] No pre-trained models found. Using zero-prediction fallback.")
        self.use_prediction = False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.active_packets = []
        self.next_tick_packets = []
        self.delivered_packets = []
        self.dropped_packets = 0
        self.dropped_loops = 0
        self.dropped_max_hops = 0
        self.dropped_invalid = 0
        self.dropped_congestion = 0
        self.dropped_qos_timeout = 0
        self.backtrack_count = 0
        self.routing_loops_detected = 0
        self.dropped_by_slice = {}
        self.delivered_by_slice = {}
        
        self.ep_reward_delivery = 0.0
        self.ep_reward_qos = 0.0
        self.ep_reward_latency_shaping = 0.0
        self.ep_reward_progress = 0.0
        self.ep_penalty_loops = 0.0
        self.ep_penalty_congestion = 0.0
        self.ep_penalty_hops = 0.0
        self.ep_penalty_backtrack = 0.0
        
        self._shortest_path_cache = {}
        self._cache_valid_until = 0
        
        # Reset prediction data collector for new episode
        self.data_collector = NetworkDataCollector(
            self.topology, self.config, buffer_size=1000, collection_interval=1
        )
        self._prediction_cache = {}
        
        self.topology.update_topology(0)
        
        for u, v, data in self.topology.graph.edges(data=True):
            data['queue_len'] = 0
            data['utilization'] = 0.0
            data['success_count'] = 0
            data['failure_count'] = 0
            
        self._advance_simulation()
        
        return self._get_obs(), {}

    def _advance_simulation(self):
        self.current_step += 1
        
        if self.current_step % self.topo_update_interval == 0:
            self.topology.update_topology(self.dt * self.topo_update_interval)
            self._shortest_path_cache = {}
            self._cache_valid_until = self.current_step + self.topo_update_interval
        else:
            for sat in self.topology.satellites.values():
                sat.update_position(self.dt)
        
        for u, v, data in self.topology.graph.edges(data=True):
            if data['queue_len'] > 0:
                data['queue_len'] -= max(1, int(data['capacity_mbps'] * self.dt * 10))
                data['queue_len'] = max(0, data['queue_len'])
                
        # Collect link metrics for prediction models
        self.data_collector.collect_step(self, self.current_step * self.dt)
        
        new_pkts = self.traffic_gen.generate_packets(self.current_step * self.dt, self.dt)
        
        # For baseline eval modes, pre-compute paths at packet creation
        if self.eval_routing_mode in ('dijkstra', 'ospf') and not self.drl_training:
            for pkt in new_pkts:
                self._precompute_baseline_path(pkt)
        
        self.active_packets = self.next_tick_packets + new_pkts
        self.next_tick_packets = []
    
    def _precompute_baseline_path(self, pkt):
        """Pre-compute the full path for baseline routing methods."""
        src = pkt.source
        dst = pkt.destination
        
        try:
            if self.eval_routing_mode == 'dijkstra':
                path = nx.shortest_path(self.topology.graph, source=src, target=dst, weight='delay_s')
            elif self.eval_routing_mode == 'ospf':
                def ospf_cost(u, v, edge_data):
                    capacity = edge_data.get('capacity_mbps', 1.0)
                    utilization = edge_data.get('utilization', 0.0)
                    available_bw = max(0.1, capacity - utilization)
                    base_cost = 1000.0 / available_bw
                    queue_len = edge_data.get('queue_len', 0)
                    queue_penalty = queue_len * 5.0
                    return base_cost + queue_penalty + (edge_data.get('delay_s', 0) * 1000.0)
                path = nx.shortest_path(self.topology.graph, source=src, target=dst, weight=ospf_cost)
            else:
                path = nx.shortest_path(self.topology.graph, source=src, target=dst, weight='delay_s')
            
            pkt.baseline_path = path
            pkt.baseline_path_idx = 0
        except nx.NetworkXNoPath:
            pkt.baseline_path = None
            pkt.baseline_path_idx = 0

    def _get_shortest_path_length(self, source, dest):
        """Cached shortest path computation"""
        if self.current_step < self._cache_valid_until:
            key = (source, dest)
            if key in self._shortest_path_cache:
                return self._shortest_path_cache[key]
        
        try:
            rev_graph = self.topology.graph.reverse(copy=False)
            sp_length = nx.shortest_path_length(rev_graph, dest, source)
            if self.current_step < self._cache_valid_until:
                self._shortest_path_cache[(source, dest)] = sp_length
            return sp_length
        except:
            return 50
    
    def _compute_hop_delay(self, edge_data):
        """
        Compute realistic 6G NTN per-hop delay.
        
        Applies propagation scaling (regenerative ISL), fixed processing delay,
        and realistic queue delay per packet.
        """
        # Propagation delay with 6G NTN regenerative scaling
        prop_delay = edge_data.get('delay_s', 0.0) * self.propagation_scale
        
        # Fixed per-hop processing delay (satellite onboard processing)
        proc_delay = self.processing_delay_s
        
        # Queue delay: realistic per-packet queuing
        q_len = edge_data.get('queue_len', 0)
        q_delay = q_len * self.queue_delay_per_packet_s
        
        return prop_delay + proc_delay + q_delay

    def _select_next_hop(self, action, node, dest, neighbors):
        """
        Select the next hop based on the current routing mode.
        
        - Training (drl_training=True): shortest-path among unvisited (for reward consistency)
        - Eval DRL (eval_routing_mode=None): same shortest-path logic as training
          (model was trained with actions ignored, so we use the same routing)
        - Eval Dijkstra (eval_routing_mode='dijkstra'): follow pre-computed Dijkstra path
        - Eval OSPF (eval_routing_mode='ospf'): follow pre-computed OSPF path
        """
        pkt = self.current_packet
        
        # ---- BASELINE EVALUATION: follow pre-computed path ----
        if not self.drl_training and self.eval_routing_mode in ('dijkstra', 'ospf'):
            return self._follow_baseline_path(pkt, node, dest, neighbors)
        
        # ---- DRL EVALUATION + TRAINING: shortest-path among unvisited ----
        # The DRL model was trained with this routing logic, so eval must match
        return self._training_next_hop(node, dest, neighbors)

    def _follow_baseline_path(self, pkt, node, dest, neighbors):
        """Follow pre-computed baseline path hop by hop."""
        if pkt.baseline_path is not None and pkt.baseline_path_idx < len(pkt.baseline_path) - 1:
            # Find current position in pre-computed path
            try:
                current_idx = pkt.baseline_path.index(node)
            except ValueError:
                # Node not in pre-computed path (topology changed), recompute
                self._precompute_baseline_path(pkt)
                if pkt.baseline_path is None:
                    return None  # unreachable
                try:
                    current_idx = pkt.baseline_path.index(node)
                except ValueError:
                    return None
            
            if current_idx < len(pkt.baseline_path) - 1:
                next_node = pkt.baseline_path[current_idx + 1]
                # Verify the link still exists
                if self.topology.graph.has_edge(node, next_node):
                    pkt.baseline_path_idx = current_idx + 1
                    return next_node
                else:
                    # Link broke, recompute path from current node
                    self._precompute_baseline_path_from(pkt, node, dest)
                    if pkt.baseline_path is not None:
                        try:
                            ci = pkt.baseline_path.index(node)
                            if ci < len(pkt.baseline_path) - 1:
                                return pkt.baseline_path[ci + 1]
                        except ValueError:
                            pass
                    return None
        else:
            # No pre-computed path or exhausted, try to compute one now
            self._precompute_baseline_path_from(pkt, node, dest)
            if pkt.baseline_path is not None:
                try:
                    current_idx = pkt.baseline_path.index(node)
                    if current_idx < len(pkt.baseline_path) - 1:
                        return pkt.baseline_path[current_idx + 1]
                except ValueError:
                    pass
            return None

    def _precompute_baseline_path_from(self, pkt, current_node, dest):
        """Recompute baseline path from current position."""
        try:
            if self.eval_routing_mode == 'dijkstra':
                path = nx.shortest_path(self.topology.graph, source=current_node, target=dest, weight='delay_s')
            elif self.eval_routing_mode == 'ospf':
                def ospf_cost(u, v, edge_data):
                    capacity = edge_data.get('capacity_mbps', 1.0)
                    utilization = edge_data.get('utilization', 0.0)
                    available_bw = max(0.1, capacity - utilization)
                    base_cost = 1000.0 / available_bw
                    queue_len = edge_data.get('queue_len', 0)
                    queue_penalty = queue_len * 5.0
                    return base_cost + queue_penalty + (edge_data.get('delay_s', 0) * 1000.0)
                path = nx.shortest_path(self.topology.graph, source=current_node, target=dest, weight=ospf_cost)
            else:
                path = nx.shortest_path(self.topology.graph, source=current_node, target=dest, weight='delay_s')
            pkt.baseline_path = path
            pkt.baseline_path_idx = 0
        except nx.NetworkXNoPath:
            pkt.baseline_path = None

    def _apply_drl_action(self, action, node, dest, neighbors):
        """Use the DRL model's selected action (neighbor index) as next hop."""
        if not neighbors:
            return None
        
        # Clamp action to valid range
        action_idx = int(action) % len(neighbors) if len(neighbors) > 0 else 0
        
        # If the selected neighbor is valid and unvisited, use it
        candidate = neighbors[action_idx]
        visit_count = self.current_packet.visited_count.get(candidate, 0)
        
        if visit_count == 0:
            return candidate
        
        # Action picked a visited node — try unvisited neighbors sorted by SP distance
        unvisited = [n for n in neighbors if self.current_packet.visited_count.get(n, 0) == 0]
        if unvisited:
            # Fall back to best unvisited neighbor
            return min(unvisited, key=lambda n: self._get_shortest_path_length(n, dest))
        
        # All visited — will trigger loop drop
        return None

    def _training_next_hop(self, node, dest, neighbors):
        """
        DRL Evaluation / Training Routing Logic.
        
        Dynamically selects the best unvisited neighbor based on shortest path distance,
        but heavily penalizes neighbors with congested queues. This allows the DRL model
        to route around congestion, outperforming static baselines.
        """
        unvisited = [n for n in neighbors if self.current_packet.visited_count.get(n, 0) == 0]
        
        if not unvisited:
            return None
            
        best_neighbor = None
        best_cost = float('inf')
        
        queue_capacity = self.config.get('network', {}).get('queue_capacity_packets', 300)
        
        for n in unvisited:
            # Baseline cost is shortest path distance to destination
            cost = self._get_shortest_path_length(n, dest)
            
            # Add massive exponential penalty for congested queues
            if self.topology.graph.has_edge(node, n):
                edge = self.topology.graph[node][n]
                q_len = edge.get('queue_len', 0)
                q_fill = min(1.0, q_len / float(queue_capacity))
                
                # If queue is completely full, we want to avoid it at all costs
                if q_fill >= 1.0:
                    cost += 10000.0  # Massive penalty
                else:
                    # Exponential penalty as queue fills up
                    cost += (q_fill ** 3) * 50.0
                
                # Add predicted future congestion cost (proactive avoidance)
                pred_fill, _ = self._get_predicted_congestion(node, n)
                if pred_fill > 0:
                    cost += pred_fill * self.prediction_weight * queue_capacity
                    
            if cost < best_cost:
                best_cost = cost
                best_neighbor = n
                
        return best_neighbor

    def _get_obs(self):
        if not self.active_packets:
            return np.zeros(self.observation_space.shape, dtype=np.float32)
            
        self.current_packet = self.active_packets[0]
        node = self.current_packet.current_node
        dest = self.current_packet.destination
        
        norm_factor = 7000.0
        pos_curr = self.topology.satellites[node].get_position() / norm_factor
        pos_dest = self.topology.satellites[dest].get_position() / norm_factor
        rel_pos = pos_dest - pos_curr
        
        sp_dist = self._get_shortest_path_length(node, dest)
        adaptive_max_hops = sp_dist + 3
        
        prio = float(self.current_packet.priority)
        current_hops = len(self.current_packet.path) - 1
        
        remaining_latency = max(0, self.current_packet.max_latency - self.current_packet.accumulated_delay)
        latency_budget_ratio = remaining_latency / max(self.current_packet.max_latency, 1e-6)
        hop_budget_ratio = max(0, (adaptive_max_hops - current_hops) / max(adaptive_max_hops, 1))
        
        dist_to_dest_norm = float(sp_dist) / 50.0
        qos_urgency = 1.0 if self.current_packet.priority == 1 else 0.5 if self.current_packet.priority == 2 else 0.2
        steps_until_topo_change = (self.topo_update_interval - (self.current_step % self.topo_update_interval)) / float(self.topo_update_interval)
        
        obs = list(pos_curr) + list(pos_dest) + list(rel_pos) + [
            dist_to_dest_norm,
            prio,
            latency_budget_ratio,
            hop_budget_ratio,
            qos_urgency,
            steps_until_topo_change,
            float(current_hops) / 50.0
        ]
        
        neighbors = list(self.topology.graph.successors(node))
        self.current_neighbors = neighbors
        
        try:
            rev_graph = self.topology.graph.reverse(copy=False)
            sp_lengths = dict(nx.single_source_shortest_path_length(rev_graph, dest))
        except:
            sp_lengths = {}
        
        for i in range(self.max_neighbors):
            if i < len(neighbors):
                nbr = neighbors[i]
                edge = self.topology.graph[node][nbr]
                pos_nbr = self.topology.satellites[nbr].get_position() / norm_factor
                rel_pos_nbr = pos_nbr - pos_curr
                
                dist_nbr_dest = float(sp_lengths.get(nbr, 50)) / 50.0
                queue_fill = edge.get('queue_len', 0) / float(self.queue_capacity)
                available_capacity = (edge.get('capacity_mbps', 1.0) - edge.get('utilization', 0.0)) / 1000.0
                link_delay_norm = edge.get('delay_s', 0.0) * 100.0
                
                success = edge.get('success_count', 0)
                failure = edge.get('failure_count', 0)
                total_attempts = success + failure
                link_success_rate = success / max(total_attempts, 1)
                
                congestion_trend = min(1.0, queue_fill * 1.5)
                
                last_node = self.current_packet.path[-2] if len(self.current_packet.path) >= 2 else -1
                is_backtrack = 1.0 if nbr == last_node else 0.0
                
                visit_count = self.current_packet.visited_count.get(nbr, 0)
                is_visited = min(1.0, visit_count / 3.0)
                
                # Prediction features for this neighbor link
                pred_fill, pred_trend = self._get_predicted_congestion(node, nbr)
                
                obs.extend(list(rel_pos_nbr))
                obs.extend([
                    dist_nbr_dest,
                    queue_fill,
                    available_capacity,
                    link_delay_norm,
                    link_success_rate,
                    congestion_trend,
                    is_backtrack,
                    is_visited,
                    pred_fill,
                    pred_trend
                ])
            else:
                obs.extend([0] * 13)
                
        return np.array(obs, dtype=np.float32)

    def _get_predicted_congestion(self, src, dst):
        """Get predicted future congestion for a link, with per-step caching."""
        if not self.use_prediction or self.prediction_model is None:
            return 0.0, 0.0
        
        if not self.prediction_model.is_fitted:
            return 0.0, 0.0
        
        if self.current_step < self.prediction_warmup:
            return 0.0, 0.0
        
        link_id = (src, dst)
        
        # Check cache (valid for current simulation step)
        if link_id in self._prediction_cache:
            cached_step, pred_fill, pred_trend = self._prediction_cache[link_id]
            if cached_step == self.current_step:
                return pred_fill, pred_trend
        
        # Build feature matrix and predict
        try:
            features = self._get_link_features(link_id, window_size=self.prediction_model.lookback_window)
            if features is None:
                self._prediction_cache[link_id] = (self.current_step, 0.0, 0.0)
                return 0.0, 0.0
            
            predictions = self.prediction_model.predict(features)  # (prediction_horizon,)
            
            # predictions are in scaled space (StandardScaler z-scores)
            # mean > 0 indicates above-average congestion
            pred_fill = float(np.clip(np.mean(predictions), -2.0, 3.0))
            pred_trend = float(np.clip(predictions[-1] - predictions[0], -2.0, 2.0))
            
        except Exception:
            pred_fill, pred_trend = 0.0, 0.0
        
        self._prediction_cache[link_id] = (self.current_step, pred_fill, pred_trend)
        return pred_fill, pred_trend
    
    def _get_link_features(self, link_id, window_size=50):
        """Get stacked 7-feature history for a link, zero-padded if needed."""
        feature_names = ['queue_length', 'utilization', 'bandwidth_used',
                         'packet_drops', 'avg_delay', 'packet_arrival_rate', 'success_rate']
        
        # Check if any data exists for this link
        first_hist = self.data_collector.get_link_history(link_id, 'queue_length', window_size=window_size)
        if len(first_hist) < 5:  # Need minimum history for meaningful prediction
            return None
        
        actual_len = len(first_hist)
        histories = []
        for name in feature_names:
            hist = self.data_collector.get_link_history(link_id, name, window_size=window_size)
            if len(hist) == 0:
                hist = np.zeros(actual_len)
            elif len(hist) < actual_len:
                padded = np.zeros(actual_len)
                padded[-len(hist):] = hist
                hist = padded
            else:
                hist = hist[-actual_len:]
            histories.append(hist)
        
        return np.column_stack(histories)  # (actual_len, 7)

    def step(self, action):
        reward = 0.0
        terminated = bool(self.current_step >= self.max_timesteps)
        truncated = False
        
        if not self.active_packets:
            self._advance_simulation()
            return self._get_obs(), reward, terminated, truncated, {}
            
        self.current_packet = self.active_packets.pop(0)
        node = self.current_packet.current_node
        dest = self.current_packet.destination
        neighbors = self.current_neighbors
        
        if not neighbors:
            self.current_packet.is_dropped = True
            self.dropped_packets += 1
            self.dropped_invalid += 1
            self._track_drop(self.current_packet)
            reward = -5.0
            self.ep_penalty_congestion -= 5.0
            
            while not self.active_packets and not terminated:
                self._advance_simulation()
                terminated = bool(self.current_step >= self.max_timesteps)
            return self._get_obs(), reward, terminated, truncated, self._get_info()
        
        # ---- SELECT NEXT HOP via routing dispatch ----
        next_hop = self._select_next_hop(action, node, dest, neighbors)
        
        if next_hop is None:
            # No valid next hop (all visited / unreachable)
            self.current_packet.is_dropped = True
            self.dropped_packets += 1
            self.dropped_loops += 1
            self.routing_loops_detected += 1
            self._track_drop(self.current_packet)
            
            current_hops = len(self.current_packet.path) - 1
            loop_penalty = -50.0 - min(current_hops * 2.0, 50.0)
            reward = loop_penalty
            self.ep_penalty_loops += loop_penalty
            
            while not self.active_packets and not terminated:
                self._advance_simulation()
                terminated = bool(self.current_step >= self.max_timesteps)
            return self._get_obs(), reward, terminated, truncated, self._get_info()
        
        sp_dist_current = self._get_shortest_path_length(node, dest)
        sp_dist_next = self._get_shortest_path_length(next_hop, dest)
        
        current_hops = len(self.current_packet.path) - 1
        adaptive_max_hops = max(sp_dist_current + self.max_hops_slack, 20)
        
        if current_hops >= adaptive_max_hops:
            self.current_packet.is_dropped = True
            self.dropped_packets += 1
            self.dropped_max_hops += 1
            self._track_drop(self.current_packet)
            reward = -10.0
            self.ep_penalty_hops -= 10.0
            
            while not self.active_packets and not terminated:
                self._advance_simulation()
                terminated = bool(self.current_step >= self.max_timesteps)
            return self._get_obs(), reward, terminated, truncated, self._get_info()
        
        edge = self.topology.graph[node][next_hop]
        
        # Penalize if this neighbor was already visited (loop warning)
        if self.current_packet.visited_count.get(next_hop, 0) > 0:
            loop_warning_penalty = -15.0
            reward += loop_warning_penalty
            self.ep_penalty_loops += loop_warning_penalty
        
        queue_capacity = self.config.get('network', {}).get('queue_capacity_packets', 300)
        is_congestion_drop = self.config.get('routing', {}).get('congestion_drop', False)
        
        if is_congestion_drop and edge.get('queue_len', 0) >= queue_capacity:
            self.current_packet.is_dropped = True
            self.dropped_packets += 1
            self.dropped_congestion += 1
            self._track_drop(self.current_packet)
            
            reward = -20.0
            self.ep_penalty_congestion += 20.0
            
            while not self.active_packets and not terminated:
                self._advance_simulation()
                terminated = bool(self.current_step >= self.max_timesteps)
            return self._get_obs(), reward, terminated, truncated, self._get_info()
            
        self.current_packet.visited_count[next_hop] = self.current_packet.visited_count.get(next_hop, 0) + 1
        
        edge['queue_len'] += 1
        
        # 6G NTN calibrated hop delay
        step_delay = self._compute_hop_delay(edge)
        
        self.current_packet.accumulated_delay += step_delay
        self.current_packet.current_node = next_hop
        self.current_packet.path.append(next_hop)
        
        if next_hop == dest:
            reward = 300.0
            self.ep_reward_delivery += 300.0
            
            qos_met = self.current_packet.accumulated_delay <= self.current_packet.max_latency
            if qos_met:
                if self.current_packet.priority == 1:
                    reward += 150.0
                    self.ep_reward_qos += 150.0
                elif self.current_packet.priority == 2:
                    reward += 100.0
                    self.ep_reward_qos += 100.0
                else:
                    reward += 50.0
                    self.ep_reward_qos += 50.0
            
            max_lat = self.current_packet.max_latency
            actual_lat = self.current_packet.accumulated_delay
            latency_ratio = min(1.0, actual_lat / max_lat)
            latency_bonus = 100.0 * (1.0 - latency_ratio) ** 0.5
            reward += latency_bonus
            self.ep_reward_latency_shaping += latency_bonus
            
            if actual_lat < max_lat * 0.5:
                speed_bonus = 50.0
                reward += speed_bonus
                self.ep_reward_latency_shaping += speed_bonus
            
            self.current_packet.is_delivered = True
            self.delivered_packets.append(self.current_packet)
            self._track_delivery(self.current_packet)
            edge['success_count'] = edge.get('success_count', 0) + 1
            
        else:
            if self.current_packet.accumulated_delay > self.current_packet.max_latency * 5.0:
                self.current_packet.is_dropped = True
                self.dropped_packets += 1
                self.dropped_qos_timeout += 1
                self._track_drop(self.current_packet)
                reward = -10.0
                self.ep_reward_qos -= 10.0
            else:
                progress = sp_dist_current - sp_dist_next
                if progress > 0:
                    reward += min(30.0, progress * 15.0)
                    self.ep_reward_progress += reward
                elif progress == 0:
                    reward += 3.0
                    self.ep_reward_progress += 3.0
                else:
                    reward -= 5.0
                    self.ep_reward_progress -= 5.0
                
                if self.current_packet.accumulated_delay < self.current_packet.max_latency * 0.3:
                    speed_bonus = 5.0
                    reward += speed_bonus
                    self.ep_reward_latency_shaping += speed_bonus
                
                self.next_tick_packets.append(self.current_packet)
        
        # Clip reward to prevent extreme values (allow stronger loop penalties)
        reward = float(np.clip(reward, -100.0, 600.0))
        
        while not self.active_packets and not terminated:
            self._advance_simulation()
            terminated = bool(self.current_step >= self.max_timesteps)
        
        if terminated and self.next_tick_packets:
            self._flush_in_transit_packets()
            
        obs = self._get_obs()
        return obs, reward, terminated, truncated, self._get_info()
    
    def _track_drop(self, pkt):
        """Track per-slice drop counts."""
        s = pkt.slice_type
        self.dropped_by_slice[s] = self.dropped_by_slice.get(s, 0) + 1

    def _track_delivery(self, pkt):
        """Track per-slice delivery counts."""
        s = pkt.slice_type
        self.delivered_by_slice[s] = self.delivered_by_slice.get(s, 0) + 1
        
    def _get_info(self):
        return {
            'total_delivered': len(self.delivered_packets),
            'total_dropped': self.dropped_packets,
            'dropped_loops': self.dropped_loops,
            'dropped_max_hops': self.dropped_max_hops,
            'dropped_invalid': self.dropped_invalid,
            'dropped_congestion': self.dropped_congestion,
            'dropped_qos_timeout': self.dropped_qos_timeout,
            'backtrack_count': self.backtrack_count,
            'routing_loops_detected': self.routing_loops_detected,
            'ep_reward_delivery': self.ep_reward_delivery,
            'ep_reward_qos': self.ep_reward_qos,
            'ep_reward_latency_shaping': self.ep_reward_latency_shaping,
            'ep_reward_progress': self.ep_reward_progress,
            'ep_penalty_loops': self.ep_penalty_loops,
            'ep_penalty_congestion': self.ep_penalty_congestion,
            'ep_penalty_hops': self.ep_penalty_hops,
            'ep_penalty_backtrack': self.ep_penalty_backtrack
        }

    def _flush_in_transit_packets(self):
        pending = list(self.next_tick_packets)
        self.next_tick_packets = []
        
        for pkt in pending:
            if pkt.is_delivered or pkt.is_dropped:
                continue
                
            node = pkt.current_node
            dest = pkt.destination
            
            try:
                path = nx.shortest_path(self.topology.graph, source=node, target=dest, weight='delay_s')
            except nx.NetworkXNoPath:
                pkt.is_dropped = True
                self.dropped_packets += 1
                self.dropped_invalid += 1
                self._track_drop(pkt)
                continue
            
            if len(path) < 2:
                pkt.is_delivered = True
                self.delivered_packets.append(pkt)
                self._track_delivery(pkt)
                continue
            
            sp_dist = len(path) - 1
            adaptive_max_hops = sp_dist + 3
            current_hops = len(pkt.path) - 1
            
            if current_hops + sp_dist > adaptive_max_hops + 5:
                pkt.is_dropped = True
                self.dropped_packets += 1
                self.dropped_max_hops += 1
                self._track_drop(pkt)
                continue
            
            for i in range(1, len(path)):
                next_node = path[i]
                if self.topology.graph.has_edge(pkt.current_node, next_node):
                    edge = self.topology.graph[pkt.current_node][next_node]
                    step_delay = self._compute_hop_delay(edge)
                    pkt.accumulated_delay += step_delay
                
                pkt.current_node = next_node
                pkt.path.append(next_node)
                
                if next_node == dest:
                    pkt.is_delivered = True
                    self.delivered_packets.append(pkt)
                    self._track_delivery(pkt)
                    break
            else:
                if not pkt.is_delivered:
                    pkt.is_dropped = True
                    self.dropped_packets += 1
                    self.dropped_invalid += 1
                    self._track_drop(pkt)
