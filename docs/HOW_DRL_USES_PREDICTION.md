# How DRL Uses Prediction Models - Complete Guide

## 📊 Training Results (248 Satellites)

**Just Completed:**
- **Network:** 248 satellites (full constellation)
- **Training Time:** 2.1 minutes
- **Data Collected:** 1,693,896 link records
- **Training Samples:** 600
- **Validation Samples:** 150

### Model Performance

| Model | MAE      | RMSE     | Winner |
|-------|----------|----------|--------|
| LSTM  | 0.0006   | 0.0008   | -      |
| GRU   | 0.0001   | 0.0001   | ⭐ Yes  |

**Excellent!** Very low error rates mean accurate predictions.

---

## 🔄 How DRL Uses Predictions

### 1. **Automatic Integration**

The prediction models are **automatically loaded** when the DRL environment starts:

```python
# In src/routing/drl_env.py (already implemented)
class SatelliteRoutingEnv(gym.Env):
    def __init__(self, topology, traffic_generator, config_path):
        # ... initialization ...
        
        # Try to load pre-trained prediction models
        if self.use_prediction:
            try:
                self.prediction_model = TrafficForecaster(
                    model_type='lstm', input_size=7, device='cpu'
                )
                self.prediction_model.load(pred_config.get('lstm_model_path'))
                print("[Prediction] Loaded LSTM model")
            except:
                print("[Prediction] No pre-trained models found")
```

### 2. **Prediction in Observation Space**

The DRL agent **sees** predictions as part of its observations:

**Observation includes:**
- Current metrics (11 features per neighbor)
- **Predicted congestion** (1 feature per neighbor) ← NEW
- **Predicted utilization** (1 feature per neighbor) ← NEW

```python
# Per neighbor observation (13 features total):
[
    delay,              # Current delay
    queue_len,          # Current queue
    utilization,        # Current utilization
    # ... 8 more current metrics ...
    predicted_queue,    # ← FUTURE congestion (from LSTM/GRU)
    predicted_util      # ← FUTURE utilization (from LSTM/GRU)
]
```

### 3. **Routing Cost Calculation**

Predictions are used to compute **smarter routing costs**:

```python
def predictive_cost(u, v, edge_data):
    # Base cost (propagation delay)
    base_cost = edge_data.get('delay_s', 0.0) * 1000.0  # ms
    
    # Current congestion
    current_queue = edge_data.get('queue_len', 0)
    
    # PREDICTED future congestion
    predicted_queue = self.predict_future_congestion((u, v), model_type='gru')
    
    # Total cost = base + current + PREDICTED
    total_cost = base_cost + current_queue * 2.0 + predicted_queue * 5.0
    #                                                    ↑
    #                                        Higher weight = avoid future congestion
    
    return total_cost
```

**Key:** Predicted congestion has weight of **5.0** vs current's **2.0**, so the router **proactively avoids** links that will become congested.

### 4. **Prediction Warmup**

Predictions only activate after collecting enough history:

```python
# In config.yaml
prediction:
  enabled: true
  warmup_steps: 50  # Collect 50 timesteps before predicting
  prediction_weight: 3.0  # How much to trust predictions
```

- **Before warmup:** Uses baseline routing (no predictions)
- **After warmup:** Uses LSTM/GRU predictions for smarter routing

---

## 🎯 Impact on Routing Performance

### Without Predictions (Baseline)
```
Routing decision based ONLY on current state:
  → Route = shortest_path(current_delays + current_queues)
  → Can't see traffic building up
  → May route into congestion
```

### With Predictions (DRL + LSTM/GRU)
```
Routing decision based on current + FUTURE state:
  → Route = shortest_path(current + PREDICTED_congestion * 5.0)
  → Sees traffic building up before it peaks
  → Proactively avoids future hotspots
  → Better load balancing
```

### Expected Improvements
1. **Lower Latency:** Avoid congested paths before they peak
2. **Higher PDR:** Fewer drops due to queue overflow
3. **Better Load Balancing:** Traffic spreads across network
4. **Smoother Performance:** Fewer sudden congestion events

---

## 📈 How to Test the Impact

### Option 1: Quick Demo
```bash
python run_prediction.py --demo
```
This runs a comparison:
- Baseline routing (no prediction)
- LSTM-aware routing
- GRU-aware routing

### Option 2: Full Evaluation
```bash
python evaluate.py --mode prediction --episodes 10
```
Runs full evaluation with prediction models.

### Option 3: Training with Predictions
```bash
python train.py
```
The DRL agent trains **with** prediction observations, learning to use them.

---

## 🔧 Configuration

### Enable/Disable Predictions

In `config/config.yaml`:

```yaml
prediction:
  enabled: true  # ← Set to false to disable
  lstm_model_path: "models/lstm_traffic_predictor.pth"
  gru_model_path: "models/gru_traffic_predictor.pth"
  lookback_window: 50  # History used for prediction
  prediction_horizon: 10  # Steps ahead to predict
  prediction_weight: 3.0  # How much to trust predictions
  warmup_steps: 50  # Steps before predictions activate
```

### Tune Prediction Weight

Higher weight = trust predictions more:
- `prediction_weight: 1.0` - Low trust, mostly use current state
- `prediction_weight: 3.0` - **Default**, balanced
- `prediction_weight: 5.0` - High trust, strongly avoid predicted congestion

---

## 🧪 Technical Details

### Prediction Process

1. **Data Collection**
   ```python
   # Collect link history (last 50 timesteps)
   history = collector.get_link_history(
       link_id=(sat_a, sat_b),
       metric='queue_length',
       window_size=50
   )
   ```

2. **Feature Preparation**
   ```python
   # Stack 7 features for prediction
   features = np.column_stack([
       queue_history,
       utilization_history,
       bandwidth_history,
       drops_history,
       delay_history,
       arrival_rate_history,
       success_rate_history
   ])  # Shape: (50, 7)
   ```

3. **Model Prediction**
   ```python
   # GRU predicts next 10 timesteps
   prediction = gru_model.predict(features)
   # Output shape: (10,)
   # Use mean as expected future congestion
   future_congestion = np.mean(prediction)
   ```

4. **Routing Decision**
   ```python
   # Compute path with predictive costs
   path = nx.shortest_path(
       graph,
       source=src,
       target=dst,
       weight=lambda u, v, d: compute_cost(u, v, d, future_congestion)
   )
   ```

### Caching

Predictions are cached to avoid recomputation:
```python
# Cache predictions for 10 timesteps
self._prediction_cache[link_id] = {
    'prediction': future_congestion,
    'valid_until': current_time + 10
}
```

---

## 📊 Monitoring Predictions

### Check if Predictions are Active

Look for this in logs when running:
```
[Prediction] Loaded LSTM model from models/lstm_traffic_predictor.pth
[Prediction] Loaded GRU model from models/gru_traffic_predictor.pth
```

### Check Prediction Usage

In routing logs:
```
Step 50: Prediction warmup complete, activating predictions
Step 51: Using GRU predictions for routing
```

### Visualization

Run with visualization to see:
```bash
python run_prediction.py --visualize
```

Generates:
- `prediction_comparison.png` - LSTM vs GRU accuracy
- `metrics_comparison.png` - Performance metrics
- `error_distribution.png` - Prediction errors

---

## 🎓 Key Concepts

### 1. **Proactive vs Reactive**
- **Reactive (baseline):** React to congestion after it happens
- **Proactive (with prediction):** Avoid congestion before it peaks

### 2. **Temporal Awareness**
- Without prediction: "This link is congested NOW"
- With prediction: "This link WILL BE congested in 5 steps"

### 3. **Load Balancing**
- Predictions help distribute traffic before hotspots form
- Results in smoother, more efficient network utilization

---

## 🚀 Next Steps

1. **✅ Models Trained** - You have working LSTM/GRU models
2. **✅ Auto-Integration** - Environment loads them automatically
3. **▶️ Test Impact** - Run evaluation to measure improvement
4. **▶️ Train DRL Agent** - Train PPO agent with prediction features
5. **▶️ Compare** - Baseline vs Prediction-aware performance

---

## 📝 Summary

Your prediction models are **trained** and **ready**. The DRL environment will:

1. ✅ Load models automatically
2. ✅ Collect link history (50 timesteps)
3. ✅ Predict future congestion (10 steps ahead)
4. ✅ Include predictions in observations (13 features per neighbor)
5. ✅ Use predictions in routing costs (weight = 5.0)
6. ✅ Proactively avoid future congestion

**Result:** Smarter routing that sees into the future! 🔮
