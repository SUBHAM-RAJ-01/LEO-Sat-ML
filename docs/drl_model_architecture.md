# DRL Routing Model Architecture and Constraints

This document details the Deep Reinforcement Learning (DRL) routing model used for packet forwarding in our simulated 6G LEO satellite constellation network. 

## 1. Overview
The agent functions as an intelligent router at each satellite node. Rather than constructing completely predetermined end-to-end paths, the agent processes packets dynamically on a hop-by-hop basis, adapting to rapidly changing link conditions, varying queue capacities, and predictable orbital dynamics.

The model is built on top of the **Proximal Policy Optimization (PPO)** algorithm implemented via Stable Baselines 3, interacting with a custom Gymnasium environment (`SatelliteRoutingEnv`).

---

## 2. Environment Details

### **State / Observation Space**
The state space provides the agent with spatial, topological, and queuing information.

**Total Dimension**: `13 + (7 × max_neighbors)`
*   **Packet & Current Node Features** (13 values):
    *   **Current Position (3)**: Normalized $(x, y, z)$ coordinates of the current satellite.
    *   **Destination Position (3)**: Normalized $(x, y, z)$ coordinates of the destination satellite.
    *   **Relative Position (3)**: Vector pointing from current to destination.
    *   **Topological Distance (1)**: Normalized shortest-path distance (hops) to the destination.
    *   **Priority (1)**: The required QoS priority of the packet.
    *   **Latency Ratio (1)**: Current accumulated delay / maximum allowed latency.
    *   **Hop Budget Ratio (1)**: Current hop count / `max_hops` limit.
*   **Neighbor Features** (7 values per neighbor × `max_neighbors`):
    *   **Relative Position (3)**: Vector pointing from current satellite to the neighbor.
    *   **Topological Distance to Dest (1)**: The neighbor's shortest-path distance to the destination.
    *   **Queue Length (1)**: Normalized current queue utilization on the link to that neighbor.
    *   **Available Bandwidth (1)**: Remaining link capacity.
    *   **Link Delay (1)**: Normalized propagation delay to that neighbor.
*(If a node has fewer than `max_neighbors`, the remaining slots are zero-padded).*

### **Action Space**
The action space uses a `Discrete` space representing the forwarding decision.
*   **Action Space Mode**: `Discrete(2)` 
*   Depending on the active configuration, the agent selects between high-level routing heuristics (e.g., Shortest Path vs. Least Congested link) or directly selects the spatial neighbor index to forward to.

---

## 3. Reward Formulation

The environment uses a shaped reward system to encourage immediate progress, enforce QoS requirements, and penalize loops or congestion.

**Positive Rewards:**
*   **Delivery Reward**: Substantial positive reward (e.g., +200) granted when the packet successfully reaches its destination node.
*   **Progress Reward**: Granted when a selected neighbor brings the packet topologically closer to the destination than the current node. This helps address sparse rewards.
*   **QoS/Urgency Reward**: Additional bonus granted upon successful delivery if the packet arrives well within its maximum latency bounds.

**Penalties (Negative Rewards):**
*   **Step Penalty**: A very small constant penalty (-0.5) applied per hop to encourage the most efficient (shortest) path possible.
*   **Loop Penalty**: Significant deduction if the agent routes a packet back to a node it has already visited. The packet is instantly dropped to clear network congestion.
*   **Congestion Penalty**: Deduction for routing traffic into edges where the queue capacity has been exceeded.
*   **Invalid Action / Dead End**: Penalty (-15) for choosing an empty neighbor slot or routing into a dead end, resulting in a dropped packet.

---

## 4. Network & Operational Constraints

To maintain realism and stability, the routing environment imposes several strict constraints:

1.  **Hop Count Limit (`max_hops = 30`)**
    *   *Constraint*: To prevent infinite loops caused by transient loops or bad policies, any packet exceeding 30 routing hops is immediately dropped (TTL expired).
2.  **Queue Capacity (`queue_capacity_packets`)**
    *   *Constraint*: Each inter-satellite link has a fixed queue buffer. If a packet is routed to a link where `queue_len >= queue_capacity`, it is dropped (imitating buffer overflow).
3.  **Strict Loop Drop Policy**
    *   *Constraint*: If a neighbor chosen by the agent is already in the packet's `path` history, the packet is immediately dropped to prevent cyclic congestion.
4.  **Dynamic Topology Constraints**
    *   *Constraint*: LEO topologies change rapidly. Inter-Satellite Links (ISLs) are recalculated based on distance and line-of-sight constraints (Earth blocking). The agent must adapt to ISLs continuously breaking and reforming.
    *   *System Impact*: The environment updates the routing graph and ISL weights periodically based on standard orbital mechanics (using TLE data).

---

## 5. Traffic Profiles and QoS
Packets are injected dynamically into the simulation by the `TrafficGenerator`. They carry variable attributes:
*   **Background Traffic**: Normal priority, generous maximum latency constraint.
*   **Time-Sensitive Traffic**: High priority, strict latency bounds. The agent observes the packet's `Latency Ratio` state variable to learn emergency detour paths when standard paths are congested.