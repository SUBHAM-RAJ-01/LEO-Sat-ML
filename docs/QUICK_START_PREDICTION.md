# Quick Start - Prediction Models

## ✅ Current Status
- **LSTM Model:** Trained & Ready (models/lstm_traffic_predictor.pth)
- **GRU Model:** Trained & Ready (models/gru_traffic_predictor.pth)
- **Verified:** Both models tested and working
- **Time:** Completed in 24 seconds

---

## 🚀 Quick Commands

### Verify Models Work
```bash
python verify_prediction_models.py
```

### Retrain Models (if needed)
```bash
python train_prediction_fast.py
```
Fast training: ~24 seconds, 20 satellites, minimal data

### Full Training (more accurate, slower)
```bash
python run_prediction.py --train --episodes 5
```
Slower: ~5-10 minutes, 200+ satellites, more data

### Run Demo
```bash
python run_prediction.py --demo
```

---

## 📊 What the Models Do

**Input:** Last 30 timesteps of link metrics (7 features):
1. Queue length
2. Utilization
3. Bandwidth used
4. Packet drops
5. Average delay
6. Packet arrival rate
7. Success rate

**Output:** Predicted values for next 5 timesteps

**Use Case:** Avoid future congestion by predicting traffic patterns

---

## 💡 Integration

Models are **automatically loaded** in routing when:
- Files exist in `models/` directory
- Config has `prediction.enabled: true`

No manual integration needed - just train and run!

---

## 🎯 Performance

| Model | MAE   | RMSE  | Winner |
|-------|-------|-------|--------|
| LSTM  | 0.357 | 0.479 | -      |
| GRU   | 0.205 | 0.252 | ⭐ Yes |

**Recommendation:** GRU performs better

---

## 📁 Files Overview

```
models/
├── lstm_traffic_predictor.pth     ← LSTM model
└── gru_traffic_predictor.pth      ← GRU model (better)

outputs/prediction/
├── prediction_comparison.png       ← Visual comparison
└── metrics_comparison.png          ← Performance chart

Scripts:
├── train_prediction_fast.py        ← Fast training (USE THIS)
├── verify_prediction_models.py     ← Test models work
└── run_prediction.py               ← Full training options
```

---

## ✨ Done!

Your prediction models are trained and ready to use. The routing system will automatically use them to make smarter routing decisions by avoiding future congestion.
