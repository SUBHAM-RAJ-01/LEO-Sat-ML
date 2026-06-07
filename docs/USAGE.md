# Quick Usage Guide

## Installation

```bash
C:\Users\srajp\AppData\Local\Programs\Python\Python312\python.exe -m pip install -r requirements.txt
```

## Main Files

### 1. **`main.py`** - Original DRL Routing System
Train and evaluate PPO-based routing (existing system).

```bash
# Train DRL agent
C:\Users\srajp\AppData\Local\Programs\Python\Python312\python.exe main.py --mode train

# Evaluate DRL vs baselines
C:\Users\srajp\AppData\Local\Programs\Python\Python312\python.exe main.py --mode eval --episodes 10
```

### 2. **`run_prediction.py`** - NEW Traffic Prediction & Visualization
All-in-one script for LSTM/GRU prediction and 3D visualization.

```bash
# Train LSTM/GRU models
C:\Users\srajp\AppData\Local\Programs\Python\Python312\python.exe run_prediction.py --train --episodes 10

# Generate 3D city-to-city visualizations
C:\Users\srajp\AppData\Local\Programs\Python\Python312\python.exe run_prediction.py --visualize

# Run prediction-aware routing demo
C:\Users\srajp\AppData\Local\Programs\Python\Python312\python.exe run_prediction.py --demo

# Do everything
C:\Users\srajp\AppData\Local\Programs\Python\Python312\python.exe run_prediction.py --all
```

---

## Complete Workflows

### Workflow A: DRL Routing Only (Original System)
```bash
# Step 1: Train DRL agent
C:\Users\srajp\AppData\Local\Programs\Python\Python312\python.exe main.py --mode train
# Output: models/ppo_routing.zip

# Step 2: Evaluate routing performance
C:\Users\srajp\AppData\Local\Programs\Python\Python312\python.exe main.py --mode eval --episodes 10
# Output: outputs/*.png (PDR, latency, QoS plots)
```

### Workflow B: Traffic Prediction Only (New Feature)
```bash
# Step 1: Train LSTM/GRU models
C:\Users\srajp\AppData\Local\Programs\Python\Python312\python.exe run_prediction.py --train --episodes 10
# Output: models/lstm_traffic_predictor.pth, models/gru_traffic_predictor.pth

# Step 2: Generate 3D visualizations
C:\Users\srajp\AppData\Local\Programs\Python\Python312\python.exe run_prediction.py --visualize
# Output: outputs/3d_viz/*.png (city routes, constellation)

# Step 3: Run prediction demo
C:\Users\srajp\AppData\Local\Programs\Python\Python312\python.exe run_prediction.py --demo
# Output: outputs/demo/*.png

# Or do all prediction steps at once:
C:\Users\srajp\AppData\Local\Programs\Python\Python312\python.exe run_prediction.py --all
```

### Workflow C: Both Systems (For Research Paper)
These are **independent** - run them separately:

```bash
# First: DRL system
C:\Users\srajp\AppData\Local\Programs\Python\Python312\python.exe main.py --mode train
C:\Users\srajp\AppData\Local\Programs\Python\Python312\python.exe main.py --mode eval

# Then: Prediction system  
C:\Users\srajp\AppData\Local\Programs\Python\Python312\python.exe run_prediction.py --all

# Now you have all figures for your paper!
```

---

## Quick Commands

**Note:** DRL and LSTM/GRU are **separate systems** - run them independently!

```bash
# DRL System Only (15-20 min)
C:\Users\srajp\AppData\Local\Programs\Python\Python312\python.exe main.py --mode train
C:\Users\srajp\AppData\Local\Programs\Python\Python312\python.exe main.py --mode eval

# Prediction System Only (10-15 min)
C:\Users\srajp\AppData\Local\Programs\Python\Python312\python.exe run_prediction.py --train --episodes 10
C:\Users\srajp\AppData\Local\Programs\Python\Python312\python.exe run_prediction.py --visualize
C:\Users\srajp\AppData\Local\Programs\Python\Python312\python.exe run_prediction.py --demo

# Or all prediction features at once
C:\Users\srajp\AppData\Local\Programs\Python\Python312\python.exe run_prediction.py --all
```

---

## Output Structure

```
models/
  ├── ppo_routing.zip                  (from main.py --mode train)
  ├── lstm_traffic_predictor.pth       (from run_prediction.py --train)
  └── gru_traffic_predictor.pth        (from run_prediction.py --train)

outputs/
  ├── prediction/                      (from run_prediction.py --train)
  │   ├── prediction_comparison.png
  │   └── metrics_comparison.png
  ├── 3d_viz/                          (from run_prediction.py --visualize)
  │   ├── route_Doha_London.png
  │   ├── route_Tokyo_Paris.png
  │   └── constellation.png
  ├── demo/                            (from run_prediction.py --demo)
  │   └── demo_route.png
  └── [eval plots]                     (from main.py --mode eval)
      ├── pdr_comparison.png
      ├── latency_comparison.png
      └── ...
```

---

## Configuration

Edit `config/config.yaml` to adjust:

```yaml
# Traffic prediction settings
prediction:
  hidden_size: 128           # Model capacity
  num_layers: 2              # LSTM/GRU layers
  epochs: 50                 # Training epochs
  lookback_window: 50        # History length
  prediction_horizon: 10     # Future steps to predict

# DRL settings (existing)
drl:
  total_timesteps: 1500000
  learning_rate: 0.0001
```

---

## What Changed?

### ✅ Still Works (Original)
- `python main.py --mode train` - Train PPO
- `python main.py --mode eval` - Evaluate routing
- All existing features unchanged

### ✅ New (Consolidated)
- **ONE file** instead of 3: `run_prediction.py`
  - Replaces: `train_traffic_prediction.py`, `generate_3d_visualizations.py`, `demo_prediction_routing.py`
- Cleaner folder structure
- No extra markdown files

---

## Troubleshooting

**Issue:** "No models found"
```bash
# Train models first
C:\Users\srajp\AppData\Local\Programs\Python\Python312\python.exe run_prediction.py --train
```

**Issue:** "Not enough data"
```bash
# Increase episodes
C:\Users\srajp\AppData\Local\Programs\Python\Python312\python.exe run_prediction.py --train --episodes 20
```

**Issue:** Want faster training
```bash
# Edit config.yaml
prediction:
  epochs: 20  # Reduce from 50
```

---

## For Publications (IEEE Access)

**Important:** These are separate experiments - run independently!

```bash
# Experiment 1: DRL Routing Performance
C:\Users\srajp\AppData\Local\Programs\Python\Python312\python.exe main.py --mode train
C:\Users\srajp\AppData\Local\Programs\Python\Python312\python.exe main.py --mode eval --episodes 10
# Figures: outputs/*.png (PDR, latency, QoS comparison)

# Experiment 2: Traffic Prediction & Visualization
C:\Users\srajp\AppData\Local\Programs\Python\Python312\python.exe run_prediction.py --train --episodes 20
C:\Users\srajp\AppData\Local\Programs\Python\Python312\python.exe run_prediction.py --visualize
# Figures: outputs/3d_viz/*.png (routes), outputs/prediction/*.png (accuracy)

# Collected Figures:
# - outputs/*.png → DRL routing results
# - outputs/3d_viz/*.png → 3D constellation & routes (300 DPI)
# - outputs/prediction/*.png → LSTM/GRU prediction accuracy
```

---

---

**Summary:** Two independent systems - `main.py` for DRL routing, `run_prediction.py` for traffic prediction.
