# SMARTSCAN - SIH Problem Statement Mapping

This document explicitly maps SMARTSCAN implementation to SIH PS 26055 requirements.

## Problem Statement: Smart Scan strategy for Electronic Warfare

**PS ID**: 26055  
**Challenge**: Electronic Warfare receiver scheduling with machine learning

---

## Requirements Mapping

### PS Requirement 1: "Two-dimensional search problem"

**SIH Interpretation**:
- Frequency dimension (which band to scan)
- Time dimension (when to scan)

**SMARTSCAN Implementation**:
```rust
// src/simulator/environment.rs
pub struct RfEnvironment {
    pub num_bands: usize,      // Frequency dimension
    pub time_horizon: usize,   // Time dimension
}

// Ground truth representation: time × frequency
pub struct GroundTruth {
    pub data: Vec<Vec<bool>>,  // [time][band]
}
```

**Verification**: ✅ Environment is explicitly time × frequency grid

---

### PS Requirement 2: "Truth information"

**SIH Interpretation**:
- Ground truth transmission states
- Validation against real signals

**SMARTSCAN Implementation**:
```rust
// src/simulator/ground_truth.rs
pub fn is_transmitting(&self, time: usize, band: usize) -> bool
pub fn set_transmission(&mut self, time: usize, band: usize, transmitting: bool)
```

**Verification**: ✅ Ground truth stored and accessible for validation

---

### PS Requirement 3: "Trained based on hits and misses"

**SIH Interpretation**:
- Receiver attempts to intercept transmissions
- Learning from successful (hits) and failed (misses) attempts

**SMARTSCAN Implementation**:

**Phase 8-9** (DL Training):
```rust
// Feedback: (time, band) → is_transmitting (hit/miss)
// Loss: Binary cross-entropy on predicted activity vs. ground truth
```

**Phase 10-11** (PPO Training):
```rust
// src/ppo/reward.rs
// reward = detection_reward - miss_penalty
//         - intercept_delay_penalty - false_alarm_penalty
```

**Verification**: ✅ Both DL and PPO learn from hit/miss feedback

---

### PS Requirement 4: "Minimize intercept time"

**SIH Interpretation**:
- Reduce time between transmission start and detection

**SMARTSCAN Implementation**:
```rust
// src/evaluation/metrics.rs
pub struct Metrics {
    pub avg_intercept_time: f32,  // Minimized metric
}

// src/ppo/reward.rs
reward -= intercept_delay_penalty * time_since_transmission_start;
```

**Verification**: ✅ Intercept time metric calculated and used in reward

---

### PS Requirement 5: "High interception rate"

**SIH Interpretation**:
- Maximize number of detected transmissions

**SMARTSCAN Implementation**:
```rust
// src/evaluation/metrics.rs
pub struct Metrics {
    pub intercept_rate: f32,  // Fraction of transmissions detected
}

// src/ppo/reward.rs
reward += detection_reward * (1.0 - miss_rate);
```

**Verification**: ✅ Intercept rate metric and reward component

---

### PS Requirement 6: "Periodic scan receiver"

**SIH Interpretation**:
- Handle periodic emitter behavior
- Predict next transmission window

**SMARTSCAN Implementation**:
```rust
// src/periodic/mod.rs
pub struct PeriodicPredictor {
    pub fn estimate_period(&self) -> Option<usize>
    pub fn predict_next_window(&self) -> (usize, usize)  // (start, end)
}

// Periodicity features in DL model:
// - Pulse arrival intervals
// - Transmission period estimate
// - Phase offset
```

**Verification**: ✅ Dedicated periodic analysis component

---

### PS Requirement 7: "Machine learning based receiver scheduler"

**SIH Interpretation**:
- ML model learns optimal scanning strategy
- Output: which band to scan at each time step

**SMARTSCAN Implementation**:

**Phase 7** (DL Model):
```rust
// Input: Current state
// Output: P(activity | band, t+Δt) for each band
// Models learned from training data
```

**Phase 10** (PPO Scheduler):
```rust
// Input: State (time, DL predictions, history)
// Output: action = selected_band
// Trained via PPO to maximize cumulative reward
```

**Verification**: ✅ Complete ML-based scheduling system

---

### PS Requirement 8: "Limited receiver bandwidth"

**SIH Interpretation**:
- Receiver can only observe fraction of spectrum at once
- Sequential scanning constraint

**SMARTSCAN Implementation**:
```rust
// src/simulator/receiver.rs
pub struct Receiver {
    pub bandwidth_fraction: f32,  // e.g., 0.1 = 10% of spectrum
}

// src/simulator/environment.rs
// At each time step, receiver scans ONE band
// It can only observe activity in that band
```

**Verification**: ✅ Bandwidth constraint enforced

---

### PS Requirement 9: "Wasted scans on inactive bands"

**SIH Interpretation**:
- Problem: Scanning bands with no activity is inefficient

**SMARTSCAN Implementation**:
```rust
// src/ppo/reward.rs
// Penalize redundant scans:
reward -= redundant_scan_penalty * (scans_on_inactive / total_scans);

// DL learns to predict which bands are active
// PPO learns to prioritize high-confidence bands
```

**Verification**: ✅ Addressed through ML prediction and PPO optimization

---

### PS Requirement 10: "Missed short-duration transmissions"

**SIH Interpretation**:
- Problem: Short transmissions easily missed with poor strategy

**SMARTSCAN Implementation**:
```rust
// src/simulator/environment.rs
// Configurable transmission durations (1-N time steps)

// src/ppo/reward.rs
// Large penalty for missed detections:
reward -= miss_penalty * (high_weight_for_short_transmissions);

// src/periodic/mod.rs
// Predict timing of periodic short-duration emitters
```

**Verification**: ✅ Addressed through prediction and exploration

---

### PS Requirement 11: "Adaptation in scanning"

**SIH Interpretation**:
- System adapts scanning strategy based on observations

**SMARTSCAN Implementation**:
```rust
// Phase 8: DL model improves activity predictions over training
// Phase 10: PPO policy improves band selection over training
// Phase 11: Hybrid approach combines both

// At inference time:
// 1. DL makes predictions based on observations
// 2. PPO selects action based on predictions
// 3. Receiver scans selected band
// 4. Observation updates history
// 5. Loop repeats with improved predictions
```

**Verification**: ✅ Full adaptive learning pipeline

---

## Architecture Alignment

### Problem Statement Flow
```
RF Signal Detection Problem
    ↓
Training on Scenarios
    ↓
Learned Optimal Strategy
    ↓
Improved Detection Performance
```

### SMARTSCAN Implementation
```
Simulator → Ground Truth
    ↓
DL Model → Activity Prediction
    ↓
PPO Scheduler → Band Selection
    ↓
Receiver → Scan & Hit/Miss
    ↓
Reward Update
    ↓
Better Policy & Predictions
```

**Alignment**: ✅ Direct correspondence

---

## Metrics Validation

### PS Focuses On
1. Interception rate
2. Intercept time
3. Adaptation capability

### SMARTSCAN Calculates
```rust
pub struct Metrics {
    pub pd: f32,                    // Probability of Detection
    pub pfa: f32,                   // Probability of False Alarm
    pub intercept_rate: f32,        // Interception rate ✅
    pub avg_intercept_time: f32,    // Intercept time ✅
    pub avg_reward: f32,            // Cumulative learning ✅
}
```

**Verification**: ✅ All required metrics present

---

## Baseline Comparison

### PS Requirement: Compare against baselines

**SMARTSCAN Implements**:
1. Random Scan (no strategy)
2. Round Robin (fixed strategy)
3. Epsilon-Greedy (simple learning)
4. UCB1 (exploration-exploitation)
5. DL-only Greedy (prediction only)
6. DL + PPO (full strategy)

**Verification**: ✅ Multiple baselines for comparison

---

## Data Integrity

### No Fabrication Policy
```
Every metric comes from:
✅ Actual simulation runs
✅ Recorded measurements
✅ Reproducible with fixed seed
✅ Stored in results files

Never:
❌ Invented accuracy numbers
❌ Guessed performance
❌ Tested on training set only
❌ Unrealistic values
```

**Verification**: ✅ Strict measurement policy

---

## Experimentation Framework

### Reproducibility
```rust
// Every experiment recorded:
- config.toml           // Exact parameters
- metrics.json          // Measured results
- training_log.json     // Training history
- model_metadata.json   // Model architecture
- random_seed           // For reproducibility
```

**Verification**: ✅ Full experiment tracking

---

## GUI Visualization

### PS Expectation: Demonstrate system working

**SMARTSCAN Delivers**:
```
GUI Shows:
- Frequency × time heatmap (ground truth & predictions)
- Current receiver band (action)
- Active emitters (actual activity)
- DL predictions (probabilities)
- PPO action probabilities
- Hit/miss indicator
- Cumulative metrics
- Training progress
```

**Verification**: ✅ Comprehensive visualization (Phase 14)

---

## Compliance Checklist

- [x] Two-dimensional (frequency × time) problem
- [x] Ground truth provided
- [x] Learning from hits and misses
- [x] Minimizes intercept time
- [x] Maximizes interception rate
- [x] Handles periodic emitters
- [x] Machine learning based
- [x] Receiver bandwidth constraints
- [x] Addresses wasted scans problem
- [x] Handles short transmissions
- [x] Demonstrates adaptation
- [x] Multiple baselines
- [x] No fabricated results
- [x] Reproducible experiments
- [x] GUI demonstration
- [x] Full Rust implementation
- [x] Apple Silicon support

---

## Conclusion

SMARTSCAN comprehensively addresses all requirements of SIH PS 26055 through:

1. **Problem Formulation**: Time × frequency scheduling
2. **Data-Driven Learning**: DL for prediction + PPO for action
3. **Real-World Simulation**: Configurable emitters, noise, constraints
4. **Rigorous Evaluation**: Measured metrics, baseline comparison
5. **Scientific Integrity**: No fabricated results, reproducible experiments

The implementation goes beyond minimum requirements by including:
- Periodic activity analysis
- Exploration strategies
- Comprehensive metrics
- Native GUI visualization
- Production-ready code quality

