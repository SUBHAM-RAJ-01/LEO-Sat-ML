"""
Real-time data collection for traffic prediction in LEO satellite networks.
Collects queue lengths, link utilization, bandwidth, packet drops, delays, and other metrics.
"""
import numpy as np
import pandas as pd
from collections import deque, defaultdict
import pickle
import os
from datetime import datetime


class NetworkDataCollector:
    """
    Collects historical network statistics for traffic prediction.
    Tracks per-link and per-satellite metrics over time.
    """
    
    def __init__(self, topology, config, buffer_size=10000, collection_interval=1):
        """
        Args:
            topology: SatelliteNetwork instance
            config: Configuration dictionary
            buffer_size: Maximum number of timesteps to store in memory
            collection_interval: Collect data every N simulation steps
        """
        self.topology = topology
        self.config = config
        self.buffer_size = buffer_size
        self.collection_interval = collection_interval
        self.step_counter = 0
        
        # Time-series buffers for link metrics
        self.link_metrics = defaultdict(lambda: {
            'queue_length': deque(maxlen=buffer_size),
            'utilization': deque(maxlen=buffer_size),
            'bandwidth_used': deque(maxlen=buffer_size),
            'packet_drops': deque(maxlen=buffer_size),
            'avg_delay': deque(maxlen=buffer_size),
            'packet_arrival_rate': deque(maxlen=buffer_size),
            'success_rate': deque(maxlen=buffer_size),
            'timestamps': deque(maxlen=buffer_size)
        })
        
        # Time-series buffers for satellite node metrics
        self.node_metrics = defaultdict(lambda: {
            'total_queue': deque(maxlen=buffer_size),
            'incoming_traffic': deque(maxlen=buffer_size),
            'outgoing_traffic': deque(maxlen=buffer_size),
            'drop_count': deque(maxlen=buffer_size),
            'avg_congestion': deque(maxlen=buffer_size),
            'timestamps': deque(maxlen=buffer_size)
        })
        
        # Global network metrics
        self.global_metrics = {
            'avg_network_utilization': deque(maxlen=buffer_size),
            'total_packets_in_flight': deque(maxlen=buffer_size),
            'network_congestion_level': deque(maxlen=buffer_size),
            'timestamps': deque(maxlen=buffer_size)
        }
        
        # Previous state for rate calculation
        self.prev_link_state = {}
        self.prev_node_state = {}
        
        # Dataset export
        self.export_dir = "data/traffic_history"
        os.makedirs(self.export_dir, exist_ok=True)
        
    def collect_step(self, env, current_time):
        """
        Collect metrics for the current simulation step.
        
        Args:
            env: SatelliteRoutingEnv instance
            current_time: Current simulation time in seconds
        """
        self.step_counter += 1
        
        if self.step_counter % self.collection_interval != 0:
            return
        
        # Collect link-level metrics
        for u, v, edge_data in self.topology.graph.edges(data=True):
            link_id = (u, v)
            
            queue_len = edge_data.get('queue_len', 0)
            capacity = edge_data.get('capacity_mbps', 1.0)
            utilization = edge_data.get('utilization', 0.0)
            delay = edge_data.get('delay_s', 0.0)
            
            success_count = edge_data.get('success_count', 0)
            failure_count = edge_data.get('failure_count', 0)
            total_attempts = success_count + failure_count
            success_rate = success_count / max(total_attempts, 1)
            
            # Calculate packet arrival rate
            prev_queue = self.prev_link_state.get(link_id, {}).get('queue', 0)
            prev_success = self.prev_link_state.get(link_id, {}).get('success', 0)
            
            arrival_rate = max(0, queue_len - prev_queue + (success_count - prev_success))
            drops = max(0, failure_count - self.prev_link_state.get(link_id, {}).get('failure', 0))
            
            self.link_metrics[link_id]['queue_length'].append(queue_len)
            self.link_metrics[link_id]['utilization'].append(min(100.0, utilization))
            self.link_metrics[link_id]['bandwidth_used'].append(utilization)
            self.link_metrics[link_id]['packet_drops'].append(drops)
            self.link_metrics[link_id]['avg_delay'].append(delay * 1000.0)  # ms
            self.link_metrics[link_id]['packet_arrival_rate'].append(arrival_rate)
            self.link_metrics[link_id]['success_rate'].append(success_rate * 100.0)
            self.link_metrics[link_id]['timestamps'].append(current_time)
            
            self.prev_link_state[link_id] = {
                'queue': queue_len,
                'success': success_count,
                'failure': failure_count
            }
        
        # Collect node-level metrics
        for node_id in self.topology.graph.nodes():
            outgoing_edges = list(self.topology.graph.out_edges(node_id, data=True))
            incoming_edges = list(self.topology.graph.in_edges(node_id, data=True))
            
            total_queue = sum(e[2].get('queue_len', 0) for e in outgoing_edges)
            avg_congestion = np.mean([e[2].get('queue_len', 0) / 
                                     self.config['network']['queue_capacity_packets'] 
                                     for e in outgoing_edges]) * 100.0 if outgoing_edges else 0.0
            
            incoming_traffic = sum(e[2].get('utilization', 0.0) for e in incoming_edges)
            outgoing_traffic = sum(e[2].get('utilization', 0.0) for e in outgoing_edges)
            
            drop_count = sum(e[2].get('failure_count', 0) for e in outgoing_edges)
            prev_drops = self.prev_node_state.get(node_id, {}).get('drops', 0)
            new_drops = max(0, drop_count - prev_drops)
            
            self.node_metrics[node_id]['total_queue'].append(total_queue)
            self.node_metrics[node_id]['incoming_traffic'].append(incoming_traffic)
            self.node_metrics[node_id]['outgoing_traffic'].append(outgoing_traffic)
            self.node_metrics[node_id]['drop_count'].append(new_drops)
            self.node_metrics[node_id]['avg_congestion'].append(avg_congestion)
            self.node_metrics[node_id]['timestamps'].append(current_time)
            
            self.prev_node_state[node_id] = {'drops': drop_count}
        
        # Collect global metrics
        all_queues = [e[2].get('queue_len', 0) for e in self.topology.graph.edges(data=True)]
        all_utils = [e[2].get('utilization', 0.0) for e in self.topology.graph.edges(data=True)]
        queue_cap = self.config['network']['queue_capacity_packets']
        
        avg_util = np.mean(all_utils) if all_utils else 0.0
        total_packets = sum(all_queues)
        congestion_level = np.mean([q / queue_cap for q in all_queues]) * 100.0 if all_queues else 0.0
        
        self.global_metrics['avg_network_utilization'].append(avg_util)
        self.global_metrics['total_packets_in_flight'].append(total_packets)
        self.global_metrics['network_congestion_level'].append(congestion_level)
        self.global_metrics['timestamps'].append(current_time)
    
    def get_link_history(self, link_id, metric='queue_length', window_size=None):
        """
        Retrieve historical data for a specific link.
        
        Args:
            link_id: Tuple (source, destination)
            metric: Metric name
            window_size: Number of recent samples to return (None = all)
        
        Returns:
            numpy array of historical values
        """
        if link_id not in self.link_metrics:
            return np.array([])
        
        data = list(self.link_metrics[link_id][metric])
        if window_size:
            data = data[-window_size:]
        return np.array(data)
    
    def get_node_history(self, node_id, metric='total_queue', window_size=None):
        """Retrieve historical data for a specific satellite node."""
        if node_id not in self.node_metrics:
            return np.array([])
        
        data = list(self.node_metrics[node_id][metric])
        if window_size:
            data = data[-window_size:]
        return np.array(data)
    
    def get_global_history(self, metric='network_congestion_level', window_size=None):
        """Retrieve global network metric history."""
        data = list(self.global_metrics[metric])
        if window_size:
            data = data[-window_size:]
        return np.array(data)
    
    def export_to_dataframe(self):
        """Export all collected data to pandas DataFrames for analysis."""
        link_records = []
        for link_id, metrics in self.link_metrics.items():
            timestamps = list(metrics['timestamps'])
            for i, ts in enumerate(timestamps):
                record = {
                    'timestamp': ts,
                    'source': link_id[0],
                    'destination': link_id[1],
                    'queue_length': list(metrics['queue_length'])[i],
                    'utilization': list(metrics['utilization'])[i],
                    'bandwidth_used': list(metrics['bandwidth_used'])[i],
                    'packet_drops': list(metrics['packet_drops'])[i],
                    'avg_delay': list(metrics['avg_delay'])[i],
                    'packet_arrival_rate': list(metrics['packet_arrival_rate'])[i],
                    'success_rate': list(metrics['success_rate'])[i]
                }
                link_records.append(record)
        
        link_df = pd.DataFrame(link_records)
        
        node_records = []
        for node_id, metrics in self.node_metrics.items():
            timestamps = list(metrics['timestamps'])
            for i, ts in enumerate(timestamps):
                record = {
                    'timestamp': ts,
                    'node_id': node_id,
                    'total_queue': list(metrics['total_queue'])[i],
                    'incoming_traffic': list(metrics['incoming_traffic'])[i],
                    'outgoing_traffic': list(metrics['outgoing_traffic'])[i],
                    'drop_count': list(metrics['drop_count'])[i],
                    'avg_congestion': list(metrics['avg_congestion'])[i]
                }
                node_records.append(record)
        
        node_df = pd.DataFrame(node_records)
        
        global_records = []
        timestamps = list(self.global_metrics['timestamps'])
        for i, ts in enumerate(timestamps):
            record = {
                'timestamp': ts,
                'avg_network_utilization': list(self.global_metrics['avg_network_utilization'])[i],
                'total_packets_in_flight': list(self.global_metrics['total_packets_in_flight'])[i],
                'network_congestion_level': list(self.global_metrics['network_congestion_level'])[i]
            }
            global_records.append(record)
        
        global_df = pd.DataFrame(global_records)
        
        return link_df, node_df, global_df
    
    def save_to_disk(self, prefix="training_run"):
        """Save collected data to disk for offline training."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        link_df, node_df, global_df = self.export_to_dataframe()
        
        link_df.to_csv(f"{self.export_dir}/{prefix}_link_metrics_{timestamp}.csv", index=False)
        node_df.to_csv(f"{self.export_dir}/{prefix}_node_metrics_{timestamp}.csv", index=False)
        global_df.to_csv(f"{self.export_dir}/{prefix}_global_metrics_{timestamp}.csv", index=False)
        
        # Also save raw deque data for fast reload
        with open(f"{self.export_dir}/{prefix}_raw_data_{timestamp}.pkl", 'wb') as f:
            pickle.dump({
                'link_metrics': dict(self.link_metrics),
                'node_metrics': dict(self.node_metrics),
                'global_metrics': self.global_metrics
            }, f)
        
        print(f"✓ Traffic data saved to {self.export_dir}/")
        return f"{self.export_dir}/{prefix}_raw_data_{timestamp}.pkl"
    
    def load_from_disk(self, filepath):
        """Load previously collected data."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.link_metrics = defaultdict(lambda: {
            'queue_length': deque(maxlen=self.buffer_size),
            'utilization': deque(maxlen=self.buffer_size),
            'bandwidth_used': deque(maxlen=self.buffer_size),
            'packet_drops': deque(maxlen=self.buffer_size),
            'avg_delay': deque(maxlen=self.buffer_size),
            'packet_arrival_rate': deque(maxlen=self.buffer_size),
            'success_rate': deque(maxlen=self.buffer_size),
            'timestamps': deque(maxlen=self.buffer_size)
        }, data['link_metrics'])
        
        self.node_metrics = defaultdict(lambda: {
            'total_queue': deque(maxlen=self.buffer_size),
            'incoming_traffic': deque(maxlen=self.buffer_size),
            'outgoing_traffic': deque(maxlen=self.buffer_size),
            'drop_count': deque(maxlen=self.buffer_size),
            'avg_congestion': deque(maxlen=self.buffer_size),
            'timestamps': deque(maxlen=self.buffer_size)
        }, data['node_metrics'])
        
        self.global_metrics = data['global_metrics']
        
        print(f"✓ Traffic data loaded from {filepath}")
