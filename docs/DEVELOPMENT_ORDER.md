# SMARTSCAN Development Order

This document outlines the phased implementation approach for SMARTSCAN.

## Phase Breakdown

### Phase 1: Rust Project + CLI + Configuration + Logging ✅
**Status**: COMPLETE

**Deliverables**:
- Cargo workspace setup
- CLI with full subcommand support (clap)
- Configuration system with TOML
- Logging infrastructure (tracing)
- Module structure for all components
- Unit tests for core modules
- No clippy warnings or format issues
- Release binary builds successfully

**Commands Working**:
- `smartscan --help`
- `smartscan demo --steps N`
- `smartscan --config FILE COMMAND`

**Tests**: 17 unit tests passing

---

### Phase 2: RF Simulator + Receiver + Ground Truth
**Status**: TODO

**Responsibilities**:
- Ground truth generation (time × frequency grid)
- Emitter behavior simulation
- Receiver model
- Scenario definition
- Deterministic/reproducible scenarios
- Configurable:
  - Number of bands
  - Time horizon
  - Emitter count
  - Activity patterns
  - Receiver bandwidth
  - Detection probability
  - False alarm probability
  - Random seed

**Key Functions**:
```rust
fn simulate_step(env: &mut Environment, band: usize) -> (bool, f32)  // Returns (hit, reward)
fn get_ground_truth(env: &Environment) -> &GroundTruth
fn generate_scenario(config: &SimConfig) -> Scenario
```

---

### Phase 3: Baselines (Random + Round Robin)
**Status**: TODO

**Baselines to implement**:
1. Random Scan
2. Round Robin
3. (Epsilon-Greedy, UCB1, etc. in later phases)

**Key Functions**:
```rust
impl Scheduler for RandomScheduler { ... }
impl Scheduler for RoundRobinScheduler { ... }
```

---

### Phase 4: Metrics Engine
**Status**: TODO

**Metrics to calculate**:
- Probability of Detection (Pd)
- Probability of False Alarm (Pfa)
- Intercept Rate
- Average Intercept Time
- Average Reward
- Miss Rate
- Redundant Scans

**Key Functions**:
```rust
fn calculate_metrics(results: &[EpisodeResult]) -> Metrics
fn compute_pd_pfa(detections: &[bool], ground_truth: &GroundTruth) -> (f32, f32)
```

---

### Phase 5: Dataset Downloader + Streaming Loader
**Status**: TODO

**Tasks**:
- Implement TSRD downloader
- Document JC Wise data source
- Streaming/chunked processing
- Memory-efficient data loading
- Data caching strategy
- Verification and checksums

**Commands**:
```bash
cargo run -- data download --dataset tsrd --mode scan --split train
cargo run -- data inspect --dataset tsrd
```

---

### Phase 6: Feature Extraction + Temporal Windows
**Status**: TODO

**Input Features**:
- Time of Arrival (ToA)
- Centre Frequency
- Pulse Width
- Amplitude
- Angle of Arrival (optional)
- Time since last scan
- Time since last detection
- Recent activity rate
- Current time (normalized)

**Output**: Temporal windows for training

---

### Phase 7: GRU/LSTM Deep Learning Model
**Status**: TODO

**Architecture**:
- GRU or LSTM layers
- Configurable:
  - Sequence length
  - Hidden size
  - Number of layers
  - Dropout
  - Learning rate

**Output**: P(activity | band, time)

---

### Phase 8: DL Training + Validation
**Status**: TODO

**Components**:
- Training loop
- Validation loop
- Early stopping
- Checkpointing
- Learning rate scheduling
- Metrics: MAE, RMSE, precision, recall, F1

---

### Phase 9: PPO Environment Integration
**Status**: TODO

**Responsibilities**:
- Create RL environment wrapper
- State representation
- Action space definition
- Reward calculation
- Episode termination

---

### Phase 10: PPO Implementation + Training
**Status**: TODO

**Components**:
- Actor network
- Critic/value network
- Rollout buffer
- PPO objective (clipped)
- GAE calculation
- Policy and value loss
- Entropy bonus

---

### Phase 11: DL + PPO Hybrid Scheduler
**Status**: TODO

**Integration**:
- Use DL output as PPO state feature
- Combined training loop
- Experiment runner

---

### Phase 12: Periodic Strategy
**Status**: TODO

**Tasks**:
- Periodicity detection
- Phase estimation
- Prediction of next active window
- Confidence metrics

---

### Phase 13: Benchmark Framework
**Status**: TODO

**Responsibilities**:
- Run identical scenarios for all schedulers
- Record metrics to JSON/CSV
- Generate comparison plots
- No fake results

---

### Phase 14: Native GUI
**Status**: TODO

**Framework**: egui/eframe

**Visualization**:
- Frequency-time heatmap
- Receiver band indicator
- Active emitters
- DL predictions
- PPO probabilities
- Metrics display
- Training progress

---

### Phase 15: Full Integration
**Status**: TODO

**Commands**:
- `cargo run -- setup`
- `cargo run -- data preprocess`
- `cargo run -- train all`
- `cargo run -- evaluate`
- `cargo run -- benchmark`
- `cargo run -- gui`
- `cargo run -- demo`

---

### Phase 16: Tests + Documentation
**Status**: TODO

**Components**:
- Unit tests for all modules
- Integration tests
- Final documentation
- README updates
- API documentation
- Dataset documentation
- PS mapping documentation

---

## Timeline

**Week 1**: Phase 1-4
**Week 2**: Phase 5-9
**Week 3**: Phase 10-13
**Week 4**: Phase 14-16
**Final**: Integration & polish

---

## Validation Checklist

- [ ] Every metric is measured, never fabricated
- [ ] All experiments are reproducible
- [ ] No data leakage
- [ ] Clean Rust code (no clippy warnings)
- [ ] Comprehensive tests
- [ ] Full documentation
- [ ] Comparison with baselines
- [ ] GUI fully functional
- [ ] Works on macOS Apple Silicon

