import gymnasium as gym
from gymnasium import spaces
import numpy as np
import networkx as nx
import random

class SatelliteRoutingEnv(gym.Env):
    """
    Custom Environment that follows gym interface.
    The agent acts an intelligent router that processes packets arriving at satellites.
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
        self.max_hops = 30  # Tighter hop limit (cycle detection handles loops now)
        self.topo_update_interval = 10  # Only rebuild topology every 10 ticks (~1s sim time)
        
        # Action space: index of the neighbor to forward the packet to (0 to max_neighbors-1)
        self.action_space = spaces.Discrete(2) # 0: Shortest Path, 1: Least Congested
        
        # Observation space:
        # [pos_curr(3), pos_dest(3), rel_pos(3), dist_to_dest(1), prio(1), lat_ratio(1), hop_ratio(1)] = 13
        # + For each neighbor: [rel_pos_to_curr(3), dist_to_dest_from_nbr(1), q(1), bw(1), delay(1)] = 7
        obs_size = 13 + 7 * self.max_neighbors
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32)
        
        self.current_step = 0
        self.active_packets = []
        self.next_tick_packets = []
        self.current_packet = None
        
        # Metrics tracking
        self.delivered_packets = []
        self.dropped_packets = 0
        self.dropped_loops = 0
        self.dropped_max_hops = 0
        self.dropped_invalid = 0
        self.backtrack_count = 0
        
        # Reward tracking
        self.ep_reward_delivery = 0.0
        self.ep_reward_qos = 0.0
        self.ep_reward_progress = 0.0
        self.ep_penalty_loops = 0.0
        self.ep_penalty_congestion = 0.0
        self.ep_reward_urgency = 0.0

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
        self.backtrack_count = 0
        
        self.ep_reward_delivery = 0.0
        self.ep_reward_qos = 0.0
        self.ep_reward_progress = 0.0
        self.ep_penalty_loops = 0.0
        self.ep_penalty_congestion = 0.0
        self.ep_reward_urgency = 0.0
        
        # Reset topology positions
        self.topology.update_topology(0)
        
        # Clear queues
        for u, v, data in self.topology.graph.edges(data=True):
            data['queue_len'] = 0
            data['utilization'] = 0.0
            
        self._advance_simulation()
        
        return self._get_obs(), {}

    def _advance_simulation(self):
        # Progress time by dt and generate traffic
        self.current_step += 1
        
        # Only rebuild the full topology graph every N ticks
        # On other ticks, just update positions (edges stay valid since sats barely move)
        if self.current_step % self.topo_update_interval == 0:
            self.topology.update_topology(self.dt * self.topo_update_interval)
        else:
            # Cheap position-only update (no graph rebuild)
            for sat in self.topology.satellites.values():
                sat.update_position(self.dt)
        
        # Cool down queues (simulate packet transmission over the link)
        for u, v, data in self.topology.graph.edges(data=True):
            if data['queue_len'] > 0:
                # Roughly decrement queues simulating BW transfer
                data['queue_len'] -= max(1, int(data['capacity_mbps'] * self.dt * 10))
                data['queue_len'] = max(0, data['queue_len'])
                
        new_pkts = self.traffic_gen.generate_packets(self.current_step * self.dt, self.dt)
        self.active_packets = self.next_tick_packets + new_pkts
        self.next_tick_packets = []
        
    def _get_obs(self):
        if not self.active_packets:
            return np.zeros(self.observation_space.shape, dtype=np.float32)
            
        self.current_packet = self.active_packets[0]
        node = self.current_packet.current_node
        dest = self.current_packet.destination
        
        # Spatial features (normalized by typical orbit radius ~7000km)
        norm_factor = 7000.0
        pos_curr = self.topology.satellites[node].get_position() / norm_factor
        pos_dest = self.topology.satellites[dest].get_position() / norm_factor
        rel_pos = pos_dest - pos_curr
        
        # Topological Distance calculation
        try:
            rev_graph = self.topology.graph.reverse(copy=False)
            sp_lengths = nx.single_source_shortest_path_length(rev_graph, dest)
            dist_to_dest = float(sp_lengths.get(node, self.max_hops * 2)) / float(self.max_hops)
        except Exception:
            sp_lengths = {}
            dist_to_dest = 2.0
        
        # QoS features
        prio = float(self.current_packet.priority)
        lat_ratio = self.current_packet.accumulated_delay / max(self.current_packet.max_latency, 1e-6)
        
        # Hop budget ratio (how much of max-hops budget is used)
        hop_ratio = len(self.current_packet.path) / float(self.max_hops)
        
        obs = list(pos_curr) + list(pos_dest) + list(rel_pos) + [dist_to_dest, prio, lat_ratio, hop_ratio]
        
        neighbors = list(self.topology.graph.successors(node))
        for i in range(self.max_neighbors):
            if i < len(neighbors):
                nbr = neighbors[i]
                edge = self.topology.graph[node][nbr]
                pos_nbr = self.topology.satellites[nbr].get_position() / norm_factor
                rel_pos_nbr = pos_nbr - pos_curr
                
                # Neighbor's distance to destination (Topological)
                dist_nbr_dest = float(sp_lengths.get(nbr, self.max_hops * 2)) / float(self.max_hops)
                
                obs.extend(list(rel_pos_nbr))
                obs.extend([
                    dist_nbr_dest,
                    edge.get('queue_len', 0) / float(self.queue_capacity), # Normalize by queue capacity
                    (edge.get('capacity_mbps', 1.0) - edge.get('utilization', 0.0)) / 1000.0,
                    edge.get('delay_s', 0.0) * 10.0 # scale ms equivalent
                ])
            else:
                obs.extend([0, 0, 0, 0, 0, 0, 0])
                
        return np.array(obs, dtype=np.float32)

    def _get_distance(self, n1, n2):
        try:
            return float(nx.shortest_path_length(self.topology.graph, n1, n2))
        except nx.NetworkXNoPath:
            return float(self.max_hops * 2)

    def step(self, action):
        reward = 0.0
        terminated = bool(self.current_step >= self.max_timesteps)
        truncated = False
        
        if not self.active_packets:
            self._advance_simulation()
            return self._get_obs(), reward, terminated, truncated, {}
            
        self.current_packet = self.active_packets.pop(0)
        node = self.current_packet.current_node
        neighbors = list(self.topology.graph.successors(node))
        
        # Small per-step cost to encourage efficiency
        reward -= 0.5
        
        # Handle dead ends
        if not neighbors:
            reward -= 15.0 
            self.ep_penalty_congestion -= 15.0
            self.current_packet.is_dropped = True
            self.dropped_packets += 1
            self.dropped_invalid += 1
            
            reward = float(np.clip(reward, -60.0, 350.0))
            while not self.active_packets and not terminated:
                self._advance_simulation()
                terminated = bool(self.current_step >= self.max_timesteps)
            return self._get_obs(), reward, terminated, truncated, self._get_info()

        # --- Heuristic-Assisted Action Space ---
        last_node = self.current_packet.path[-2] if len(self.current_packet.path) >= 2 else None
        valid_neighbors = [n for n in neighbors if n != last_node]
        if not valid_neighbors:
            valid_neighbors = neighbors # Force backtrack if dead end
            
        if action == 0:
            # Action 0: Shortest Topological Path
            best_dist = float('inf')
            next_hop = valid_neighbors[0]
            for n in valid_neighbors:
                dist = self._get_distance(n, self.current_packet.destination)
                if dist < best_dist:
                    best_dist = dist
                    next_hop = n
        else:
            # Action 1: Least Congested Path
            shortest_dist = min([self._get_distance(n, self.current_packet.destination) for n in valid_neighbors])
            candidate_neighbors = [n for n in valid_neighbors if self._get_distance(n, self.current_packet.destination) <= shortest_dist + 1.0]
            
            best_queue = float('inf')
            next_hop = candidate_neighbors[0]
            for n in candidate_neighbors:
                edge = self.topology.graph[node][n]
                q_len = edge.get('queue_len', 0)
                if q_len < best_queue:
                    best_queue = q_len
                    next_hop = n
                    
        if next_hop == last_node:
            self.backtrack_count += 1
        edge = self.topology.graph[node][next_hop]
        
        # --- Cycle detection: track visits per node ---
        visit_count = self.current_packet.visited_count.get(next_hop, 0) + 1
        self.current_packet.visited_count[next_hop] = visit_count
        
        if visit_count >= 3:
            # True cycle detected — DROP the packet
            self.current_packet.is_dropped = True
            self.dropped_packets += 1
            self.dropped_loops += 1
            reward -= 25.0
            self.ep_penalty_loops -= 25.0
            
            reward = float(np.clip(reward, -60.0, 350.0))
            while not self.active_packets and not terminated:
                self._advance_simulation()
                terminated = bool(self.current_step >= self.max_timesteps)
            return self._get_obs(), reward, terminated, truncated, self._get_info()
        
        # Record current distance before moving
        current_dist = self._get_distance(node, self.current_packet.destination)
        
        # Update state
        edge['queue_len'] += 1
        q_delay = edge['queue_len'] * 0.001
        step_delay = edge.get('delay_s', 0.0) + q_delay
        
        self.current_packet.accumulated_delay += step_delay
        self.current_packet.current_node = next_hop
        self.current_packet.path.append(next_hop)
        
        if next_hop == self.current_packet.destination:
            # ========== DELIVERY SUCCESSFUL ==========
            reward += 200.0
            self.ep_reward_delivery += 200.0
            
            # QoS bonus (priority-weighted: URLLC gets +80, eMBB +40, mMTC +26.7)
            if self.current_packet.accumulated_delay <= self.current_packet.max_latency:
                qos_bonus = 80.0 / self.current_packet.priority
                reward += qos_bonus
                self.ep_reward_qos += qos_bonus
            
            # URLLC urgency bonus — extra reward for meeting strict deadline
            if self.current_packet.priority == 1 and self.current_packet.accumulated_delay <= self.current_packet.max_latency:
                urgency_bonus = 25.0
                reward += urgency_bonus
                self.ep_reward_urgency += urgency_bonus
                
            self.current_packet.is_delivered = True
            self.delivered_packets.append(self.current_packet)
        else:
            # ========== STILL IN TRANSIT ==========
            if len(self.current_packet.path) > self.max_hops: 
                self.current_packet.is_dropped = True
                self.dropped_packets += 1
                self.dropped_max_hops += 1
                reward -= 20.0
                self.ep_penalty_loops -= 20.0
            else:
                # Progress Reward (Distance change in Hops)
                next_dist = self._get_distance(next_hop, self.current_packet.destination)
                max_dist = float(self.max_hops)
                progress = (current_dist - next_dist) / max_dist
                prog_rew = progress * 60.0  # Equals 2.0 per valid hop!
                reward += prog_rew 
                self.ep_reward_progress += prog_rew
                
                # Proportional congestion penalty (always-on, scaled by queue fill)
                queue_fill = edge['queue_len'] / float(self.queue_capacity)
                if queue_fill > 0.3:  # Only penalize when queue is getting busy
                    cong_penalty = -3.0 * queue_fill
                    reward += cong_penalty
                    self.ep_penalty_congestion += cong_penalty
                    
                self.next_tick_packets.append(self.current_packet)
            
        # Reward Clipping for stability
        reward = float(np.clip(reward, -60.0, 350.0))
        
        # Ensure we always have observation ready before exiting
        while not self.active_packets and not terminated:
            self._advance_simulation()
            terminated = bool(self.current_step >= self.max_timesteps)
        
        # --- End-of-episode flush ---
        # When the episode terminates, in-transit packets in next_tick_packets
        # would be silently discarded. Instead, route them via shortest path
        # so they are counted in the metrics.
        if terminated and self.next_tick_packets:
            self._flush_in_transit_packets()
            
        obs = self._get_obs()
        return obs, reward, terminated, truncated, self._get_info()
        
    def _get_info(self):
        return {
            'total_delivered': len(self.delivered_packets),
            'total_dropped': self.dropped_packets,
            'dropped_loops': self.dropped_loops,
            'dropped_max_hops': self.dropped_max_hops,
            'dropped_invalid': self.dropped_invalid,
            'backtrack_count': self.backtrack_count,
            'ep_reward_delivery': self.ep_reward_delivery,
            'ep_reward_qos': self.ep_reward_qos,
            'ep_reward_progress': self.ep_reward_progress,
            'ep_reward_urgency': self.ep_reward_urgency,
            'ep_penalty_loops': self.ep_penalty_loops,
            'ep_penalty_congestion': self.ep_penalty_congestion
        }

    def _flush_in_transit_packets(self):
        """
        At episode end, route remaining in-transit packets toward
        their destination using Dijkstra shortest path via NetworkX.
        Packets that can't reach destination are counted as dropped.
        """
        pending = list(self.next_tick_packets)
        self.next_tick_packets = []
        
        for pkt in pending:
            if pkt.is_delivered or pkt.is_dropped:
                continue
                
            node = pkt.current_node
            dest = pkt.destination
            
            # Try Dijkstra shortest path
            try:
                path = nx.shortest_path(
                    self.topology.graph, source=node, target=dest, weight='delay_s'
                )
            except nx.NetworkXNoPath:
                pkt.is_dropped = True
                self.dropped_packets += 1
                self.dropped_invalid += 1
                continue
            
            # Walk the path and accumulate delay
            if len(path) < 2:
                # Already at destination (shouldn't happen, but handle it)
                pkt.is_delivered = True
                self.delivered_packets.append(pkt)
                continue
                
            # Check if the remaining path is too long
            remaining_hops = len(path) - 1
            current_hops = len(pkt.path) - 1
            if current_hops + remaining_hops > self.max_hops + 10:  # Small grace buffer for flush
                pkt.is_dropped = True
                self.dropped_packets += 1
                self.dropped_max_hops += 1
                continue
            
            # Execute the path
            for i in range(1, len(path)):
                next_node = path[i]
                if self.topology.graph.has_edge(pkt.current_node, next_node):
                    edge = self.topology.graph[pkt.current_node][next_node]
                    q_delay = edge.get('queue_len', 0) * 0.001
                    pkt.accumulated_delay += edge.get('delay_s', 0.0) + q_delay
                
                pkt.current_node = next_node
                pkt.path.append(next_node)
                
                if next_node == dest:
                    pkt.is_delivered = True
                    self.delivered_packets.append(pkt)
                    break
            else:
                # Path walked but didn't reach destination (edge removed mid-flush)
                if not pkt.is_delivered:
                    pkt.is_dropped = True
                    self.dropped_packets += 1
                    self.dropped_invalid += 1
