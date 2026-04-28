# 6G-Oriented QoS-Aware DRL Routing in LEO Constellations

This project implements a complete, modular, simulation-driven system for optimizing routing and bandwidth allocation in multi-orbit Low Earth Orbit (LEO) satellite constellation networks. It uses **Deep Reinforcement Learning (DRL)** to dynamically route Non-Terrestrial Network (NTN) traffic while satisfying strict 6G Quality of Service (QoS) requirements (URLLC, eMBB, mMTC).

## Key Features
- **Dynamic 3D Orbital Mechanics:** Simplified circular motion modeling across multi-layer topologies.
- **Gymnasium PPO Environment:** Formulates satellite routing and congestion control as an RL MDP.
- **6G Network Slicing:** Generates heterogeneous traffic flows mimicking URLLC, eMBB, and mMTC.
- **Baseline Implementations:** Evaluate against Dijkstra's Shortest Path and bandwidth-aware OSPF.
- **Publication-ready Output:** Matplotlib integration to map PDR, Latency, and QoS.

## Quick Start
```bash
pip install -r requirements.txt
python main.py --mode train
python main.py --mode eval
```

For more documentation, see the `docs/` folder:
- [Setup and Run Guide](docs/setup_and_run.md)
- [Conceptual Project Explanation](docs/project_explanation.md)
