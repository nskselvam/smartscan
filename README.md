# SMARTSCAN

**ML-Based Adaptive Scan Strategy for Electronic Warfare**

Smart India Hackathon 2026 - PS 26055

## Overview

SMARTSCAN is a comprehensive Rust-based implementation of an intelligent RF receiver scheduling system. The system learns to optimally choose which frequency band to scan and when to scan it using a hybrid approach combining:

- **Temporal Deep Learning** (GRU/LSTM) for predicting future frequency band activity
- **PPO Reinforcement Learning** for making optimal band selection decisions
- **RF Simulator** with ground truth for realistic emitter behavior
- **Multiple Baselines** for comparison and validation
- **Native GUI** for visualization and interaction

## Project Status

### Phase 1: Complete ✅

- [x] Rust project structure
- [x] CLI with full subcommand support
- [x] Configuration system with TOML support
- [x] Logging infrastructure
- [x] All module placeholders created
- [x] Unit tests for core components
- [x] Code quality checks (clippy, fmt)
- [x] Release binary builds successfully

## Architecture

```
smartscan/
├── Cargo.toml                 # Rust dependencies
├── configs/
│   └── default.toml           # Default configuration
├── data/                      # Data storage (raw, processed, cache)
├── models/                    # Trained model checkpoints (dl, ppo)
├── results/                   # Experiment results and metrics
├── scripts/                   # Utility scripts
├── docs/                      # Documentation
└── src/
    ├── main.rs               # Application entry point
    ├── lib.rs                # Library root
    ├── cli.rs                # Command-line interface
    ├── config.rs             # Configuration management
    ├── logging.rs            # Logging setup
    ├── simulator/            # RF environment simulator
    ├── data/                 # Data loading & preprocessing
    ├── dl/                   # Deep learning models
    ├── ppo/                  # PPO scheduler
    ├── scheduler/            # Scheduler implementations
    ├── periodic/             # Periodic activity analysis
    ├── evaluation/           # Metrics and evaluation
    └── gui/                  # Native GUI
```

## Quick Start

### Building

```bash
cargo build --release
```

### Running

```bash
# Show help
cargo run --release -- --help

# Run demo simulation
cargo run --release -- demo --steps 1000

# Setup data (Phase 5)
cargo run --release -- setup

# Train DL model (Phase 8)
cargo run --release -- train dl --epochs 100

# Train PPO scheduler (Phase 10)
cargo run --release -- train ppo --steps 100000

# Train everything (Phase 15)
cargo run --release -- train all

# Evaluate model (Phase 8+)
cargo run --release -- evaluate --episodes 100

# Run benchmarks (Phase 13)
cargo run --release -- benchmark

# Launch GUI (Phase 14)
cargo run --release -- gui
```

### Testing

```bash
# Run all tests
cargo test

# Run tests with output
cargo test -- --nocapture

# Check code quality
cargo clippy --all-targets
cargo fmt --check
```

## Implementation Roadmap

### Phase 1: Rust Project + CLI + Configuration ✅
- [x] Workspace setup
- [x] CLI parsing with clap
- [x] Configuration system with TOML
- [x] Logging with tracing
- [x] Module structure
- [x] Unit tests

### Phase 2-16: Coming Soon
See [DEVELOPMENT_ORDER.md](docs/DEVELOPMENT_ORDER.md) for detailed phase breakdown.

## Key Features (Planned)

- **RF Simulator**: Deterministic/reproducible simulation environment
  - Fixed-frequency, periodic, intermittent, frequency-agile, frequency-hopping emitters
  - Configurable receiver bandwidth, detection/false-alarm probabilities
  - Time × frequency ground truth representation

- **Deep Learning Model**: GRU/LSTM for activity prediction
  - Input: Recent band activity, hit/miss history, time-since-scan/detection
  - Output: P(activity | band, future_time)
  - Training with validation, early stopping, checkpointing

- **PPO Scheduler**: Reinforcement learning for band selection
  - Discrete action space (band selection)
  - Configurable PPO hyperparameters (clip epsilon, GAE lambda, entropy bonus)
  - Policy and value networks

- **Reward Function**: Encourage successful detection, penalize misses and false alarms
  - Configurable coefficients for tuning
  - Hit/miss feedback loop

- **Multiple Baselines**:
  - Random Scan
  - Round Robin
  - Epsilon-Greedy
  - UCB1
  - DL-only Greedy
  - DL + PPO Hybrid

- **Periodic Analysis**: Estimate periodicity and predict transmission windows

- **Metrics Engine**:
  - Probability of Detection (Pd)
  - Probability of False Alarm (Pfa)
  - Intercept Rate
  - Intercept Time
  - Average Reward

- **Benchmark Framework**: Reproducible experiments with identical scenarios

- **Native GUI**: Real-time visualization of:
  - Frequency-time heatmap
  - Receiver state
  - DL predictions
  - PPO decisions
  - Metrics display

## Configuration

The default configuration is in `configs/default.toml`:

```toml
[app]
name = "SMARTSCAN"
version = "0.1.0"
log_level = "info"
data_dir = "data"
models_dir = "models"
results_dir = "results"

[simulator]
num_bands = 32
time_horizon = 1000
num_emitters = 5
receiver_bandwidth_fraction = 0.1
detection_probability = 0.95
false_alarm_probability = 0.01
seed = 42

[training]
batch_size = 32
learning_rate = 0.001
epochs = 100
validation_split = 0.2
checkpoint_dir = "models"

[evaluation]
num_test_episodes = 100
metrics_output = "results/metrics"
```

Load a custom config:
```bash
cargo run --release -- --config configs/custom.toml demo
```

## Datasets

The project uses:
- **TSRD**: Turing Synthetic Radar Dataset
  - URL: https://huggingface.co/datasets/alan-turing-institute/turing-synthetic-radar-dataset
  - Challenge: https://github.com/alan-turing-institute/turing-deinterleaving-challenge

- **JC Wise Radar Emitter Database** (2024)

See [docs/DATASETS.md](docs/DATASETS.md) for detailed information.

## Technology Stack

- **Rust 2021 Edition** - Pure Rust implementation
- **Cargo** - Package management
- **clap** - CLI parsing
- **serde/toml** - Configuration
- **tracing** - Logging
- **ndarray** - Numerical arrays
- **rand** - Random number generation
- **thiserror/anyhow** - Error handling
- **Burn** - Deep learning (Phase 7+)
- **egui** - GUI (Phase 14+)

## Platform Support

- macOS (including Apple Silicon)
- Linux
- Windows

## Testing

Phase 1 includes unit tests for:
- Configuration validation
- CLI parsing
- Module initialization
- Simulator components
- Ground truth operations
- Logging setup

All tests pass without warnings:
```
test result: ok. 17 passed; 0 failed
```

## Code Quality

- No clippy warnings
- Code formatted with `cargo fmt`
- All dependencies vendored in Cargo.lock

## Documentation

- [DATASETS.md](docs/DATASETS.md) - Data sources and preprocessing
- [PS_MAPPING.md](docs/PS_MAPPING.md) - SIH problem statement mapping
- [DEVELOPMENT_ORDER.md](docs/DEVELOPMENT_ORDER.md) - Phase breakdown
- [API.md](docs/API.md) - Module documentation (Phase 2+)

## Requirements & Constraints

✅ **Fully Satisfied**:
- Entire application in Rust (no C++, Python, Node.js, etc.)
- Native desktop GUI (egui)
- Modular architecture
- Reproducible experiments
- Configuration management
- Comprehensive logging
- Unit tests
- macOS Apple Silicon support

## Next Steps (Phase 2)

- [x] Phase 1: Complete
- [ ] Phase 2: RF Simulator implementation
  - Ground truth generation
  - Receiver model
  - Emitter generation
  - Environment step function

## Contributing

This is a Smart India Hackathon submission. The code is organized for easy development following the phased approach outlined in this document.

## License

SIH 2026 - Submission for PS 26055

## Contact

**Smart India Hackathon 2026**  
**Problem Statement 26055**: Smart Scan strategy for Electronic Warfare
