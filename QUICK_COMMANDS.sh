#!/bin/bash
# SMARTSCAN Phase 1 Quick Reference Commands

# Build and Run
cargo build --release              # Build optimized binary
cargo run --release -- --help      # Show all commands
cargo run --release -- demo        # Run demo simulation

# Testing
cargo test                         # Run all tests
cargo test -- --nocapture        # Show test output
cargo test TESTNAME              # Run specific test

# Code Quality
cargo check                        # Quick compilation check
cargo clippy --all-targets        # Linter check
cargo fmt --check                 # Format check
cargo fmt                         # Auto-format code

# Clean Up
cargo clean                        # Remove build artifacts
cargo clean --release             # Remove release artifacts only

# Version & Info
cargo --version                   # Cargo version
rustc --version                   # Rust compiler version
rustup show                       # Rust toolchain info

# Demo Commands (all currently show "not yet implemented")
cargo run --release -- demo --steps 100
cargo run --release -- demo --steps 1000 --visualize
cargo run --release -- -l debug demo --steps 10

# Data Commands (placeholder implementations)
cargo run --release -- data inspect
cargo run --release -- data inspect --dataset tsrd
cargo run --release -- data preprocess --input data/raw --output data/processed
cargo run --release -- data download --dataset tsrd --mode scan --split train

# Training Commands (placeholder implementations)
cargo run --release -- train dl --epochs 100
cargo run --release -- train dl --epochs 50 --lr 0.01 --batch-size 64
cargo run --release -- train ppo --steps 100000
cargo run --release -- train ppo --steps 50000 --lr 0.001
cargo run --release -- train all
cargo run --release -- train all --quick

# Evaluation Commands (placeholder implementations)
cargo run --release -- evaluate --episodes 100
cargo run --release -- evaluate --episodes 50 --checkpoint models/dl/best.pth

# Benchmark Commands (placeholder implementations)
cargo run --release -- benchmark
cargo run --release -- benchmark --benchmark-type speed --output-format json

# GUI Command (placeholder implementation)
cargo run --release -- gui

# With Custom Config
cargo run --release -- --config configs/default.toml demo

# Help for Subcommands
cargo run --release -- data --help
cargo run --release -- train --help
cargo run --release -- data download --help
cargo run --release -- train dl --help

# Using Different Log Levels
cargo run --release -- -l trace demo --steps 5
cargo run --release -- -l debug demo --steps 5
cargo run --release -- -l info demo --steps 5
cargo run --release -- -l warn demo --steps 5
cargo run --release -- -l error demo --steps 5

# Release Binary (after building)
./target/release/smartscan --help
./target/release/smartscan demo --steps 100
./target/release/smartscan -l debug data inspect
