import numpy as np
import random

class Packet:
    def __init__(self, pkt_id, source, destination, slice_type, size_bytes, priority, creation_time, max_latency):
        self.pkt_id = pkt_id
        self.source = source
        self.destination = destination
        self.slice_type = slice_type
        self.size_bytes = size_bytes
        self.priority = priority
        self.creation_time = creation_time
        self.max_latency = max_latency
        
        # Track metrics
        self.path = [source]
        self.current_node = source
        self.accumulated_delay = 0.0
        self.is_delivered = False
        self.is_dropped = False
        self.visited_count = {source: 1}  # Track visits per node for cycle detection

    def __repr__(self):
        return f"<Packet {self.pkt_id} | {self.slice_type} | {self.source}->{self.destination}>"

class TrafficGenerator:
    def __init__(self, config, num_nodes):
        self.config = config["traffic"]
        self.num_nodes = num_nodes
        self.packet_counter = 0
        self.slices = self.config['slices']
        self.base_rate = self.config['generation_rate_packets_per_sec']
        
    def generate_packets(self, current_time, dt_seconds):
        """
        Generate random packets according to slice configurations, supporting uniform distributions.
        """
        packets = []
        expected_packets = self.base_rate * dt_seconds
        
        # Determine actual number to create using Poisson for bursty/realistic arrivals
        num_to_create = np.random.poisson(expected_packets)
        
        if num_to_create == 0:
            return packets
            
        # Slice probability weights
        slice_names = list(self.slices.keys())
        weights = [self.slices[s]['weight'] for s in slice_names]
        weights = np.array(weights) / np.sum(weights)
        
        for _ in range(num_to_create):
            src, dst = random.sample(range(self.num_nodes), 2)
            slice_type = np.random.choice(slice_names, p=weights)
            s_config = self.slices[slice_type]
            
            p = Packet(
                pkt_id=self.packet_counter,
                source=src,
                destination=dst,
                slice_type=slice_type,
                size_bytes=s_config['packet_size_bytes'],
                priority=s_config['priority'],
                creation_time=current_time,
                max_latency=s_config['max_latency_ms'] / 1000.0  # Convert to seconds
            )
            packets.append(p)
            self.packet_counter += 1
            
        return packets
