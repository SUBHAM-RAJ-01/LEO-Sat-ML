import networkx as nx

class RoutingBaseline:
    def __init__(self, topology):
        """
        topology: instance of SatelliteNetwork
        """
        self.topology = topology

    def get_next_hop_dijkstra(self, source, destination):
        """
        Pure shortest path routing based on propagation delay (distance).
        Returns the ID of the next hop satellite.
        """
        if source == destination:
            return destination
            
        try:
            path = nx.shortest_path(self.topology.graph, source=source, target=destination, weight='delay_s')
            if len(path) > 1:
                return path[1]
            return destination
        except nx.NetworkXNoPath:
            return None

    def get_next_hop_ospf(self, source, destination):
        """
        OSPF-like routing: Cost is penalized by high utilization & large queues,
        simulating congestion-aware shortest path.
        """
        if source == destination:
            return destination
            
        def ospf_cost(u, v, edge_data):
            capacity = edge_data.get('capacity_mbps', 1.0)
            utilization = edge_data.get('utilization', 0.0)
            available_bw = max(0.1, capacity - utilization) # Prevent division by zero
            
            # Simplified OSPF cost calculation
            base_cost = 1000.0 / available_bw
            
            queue_len = edge_data.get('queue_len', 0)
            queue_penalty = queue_len * 5.0 # Arbitrary queue penalty
            
            return base_cost + queue_penalty + (edge_data.get('delay_s', 0) * 1000.0)

        try:
            path = nx.shortest_path(self.topology.graph, source=source, target=destination, weight=ospf_cost)
            if len(path) > 1:
                return path[1]
            return destination
        except nx.NetworkXNoPath:
            return None
