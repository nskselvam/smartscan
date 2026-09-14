# SMARTSCAN Datasets

This document describes the datasets used in SMARTSCAN and how they map to the SIH problem.

## Dataset Strategy

SMARTSCAN uses two complementary data sources:

1. **TSRD** (Turing Synthetic Radar Dataset) - For realistic emitter behavior patterns
2. **RF Simulator** - For creating the actual SIH scheduling environment

This approach ensures we:
- Leverage realistic pulse-level data from TSRD
- Create controlled, reproducible scenarios with ground truth
- Avoid fabricating data or making unsupported claims

## Primary Dataset: TSRD

### Source
- **Official Repository**: https://huggingface.co/datasets/alan-turing-institute/turing-synthetic-radar-dataset
- **Challenge**: https://github.com/alan-turing-institute/turing-deinterleaving-challenge
- **License**: Check official repository
- **Size**: Extremely large (billions of pulses)

### What TSRD Provides
- Realistic radar pulse data
- Multiple emitter scenarios
- Pulse features:
  - Time of Arrival (ToA)
  - Centre Frequency
  - Pulse Width
  - Amplitude
  - Angle of Arrival

### TSRD Limitations
TSRD is a **pulse-deinterleaving dataset**, not directly a receiver scheduling dataset.

**Important**: We do NOT claim TSRD directly provides SIH scheduling labels. Instead:

1. **TSRD** → Extract realistic emitter behavior patterns
2. **RF Simulator** → Create time × frequency ground truth
3. **DL Model** → Learn activity prediction
4. **PPO Scheduler** → Learn optimal band selection

### Modes
- **Scan Mode**: Beam-switching receiver
- **Stare Mode**: Fixed-beam receiver

### Splits
- **Train**: Training data
- **Validation**: Validation data
- **Test**: Test data

### Local Storage
```
data/
├── raw/
│   ├── tsrd_scan_train/
│   ├── tsrd_scan_val/
│   ├── tsrd_scan_test/
│   ├── tsrd_stare_train/
│   └── ...
├── processed/
│   ├── features/
│   ├── windows/
│   └── scenarios/
└── cache/
    └── dataset_index.json
```

### Download
```bash
cargo run --release -- data download --dataset tsrd --mode scan --split train
```

### Data Engineering Requirements
- **Do NOT** load entire dataset into RAM
- **Use** streaming/chunked processing
- **Implement** buffered I/O and parallel preprocessing
- **Store** preprocessed data in compact format
- **Cache** intermediate results

## Secondary Dataset: JC Wise Radar Emitter Database

### Source
- **Exact URL**: TBD (search official SIH problem statement)
- **License**: Check official source
- **Expected Features**: Emitter characteristics, frequencies, signal properties

### Usage
- Supplement TSRD patterns
- Verify realistic frequency ranges and behavior

## Synthetic RF Simulator Dataset

### Generation
The RF simulator generates:
- **Ground Truth**: time × frequency matrix
- **Emitter Activity**: Configurable patterns
- **Receiver Observations**: With noise and false alarms

### Emitter Types
1. **Fixed Frequency**: Transmits on same frequency
2. **Periodic**: Regular transmission windows
3. **Intermittent**: Random on/off activity
4. **Frequency Agile**: Changing frequencies
5. **Frequency Hopping**: Rapid frequency changes

### Configuration
```toml
[simulator]
num_bands = 32
time_horizon = 1000
num_emitters = 5
receiver_bandwidth_fraction = 0.1
detection_probability = 0.95
false_alarm_probability = 0.01
seed = 42
```

### Reproducibility
- Deterministic with fixed seed
- All scenarios stored in results directory
- Identical scenarios used for baseline comparison

## Data Pipeline

```
TSRD Raw Data
    ↓
Streaming Loader (chunked)
    ↓
Feature Extraction
    (ToA, freq, pulse width, AoA, etc.)
    ↓
Normalization
    ↓
Temporal Window Generation
    (no future leakage)
    ↓
Compact Training Format
    (HDF5, binary, or custom)
    ↓
Training Data Cache
```

## Feature Engineering

### Input Features (per time step)
- Recent band activity (last N scans)
- Recent hit/miss history
- Time since last scan (per band)
- Time since last detection (per band)
- Observed transmission frequency
- Recent centre frequency
- Pulse timing features (ToA)
- Pulse width
- Amplitude
- AoA (if available)
- Historical activity rate
- Current time (normalized to [0,1])
- Current receiver state

### Normalization
- StandardScaler for continuous features
- Min-max normalization for bounded features
- Handle missing values (zero-fill or interpolation)

### Temporal Windows
- Sequence length: Configurable (default 32)
- No future information leakage
- Scenario-aware train/val/test splits

## Data Validation

Before training:
1. Check dataset file integrity
2. Verify feature shapes
3. Validate no NaN/Inf values
4. Check value ranges
5. Confirm train/val/test splits don't overlap (scenario-aware)

## Disk Space Requirements

### Estimated (adjusted based on actual downloads)
- TSRD Raw: ~500 GB (full dataset)
- TSRD Processed: ~100 GB (compressed features)
- Models: ~5 GB (checkpoints)
- Results: ~10 GB (experiment logs)
- **Total for full training**: ~600 GB

### Development Mode
- Subset download: ~50 GB
- Allows full pipeline development
- All commands work

## Development vs. Production

### Development Mode
```bash
cargo run -- data download --dataset tsrd --subset 0.1 --split train
```
- Downloads 10% of training data
- Sufficient for testing pipeline
- Runs faster

### Full Training Mode
```bash
cargo run -- data download --dataset tsrd --split train
```
- Downloads complete dataset
- Streaming processing prevents memory overflow
- Includes validation and test splits

## Privacy & Licensing

- All datasets use official, public sources
- No synthetic data labeled as real
- All sources clearly documented
- Respect original licenses
- No copyrighted data beyond fair use for research

## Phase 5 Implementation

When implementing data download (Phase 5):

1. Verify HuggingFace token (HUGGINGFACE_TOKEN env var)
2. Check disk space availability
3. Download with progress bar
4. Resume interrupted downloads
5. Verify checksums
6. Build dataset index
7. Print disk usage summary

