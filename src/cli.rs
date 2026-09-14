use clap::{Parser, Subcommand};
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(name = "SMARTSCAN")]
#[command(about = "ML-Based Adaptive Scan Strategy for Electronic Warfare", long_about = None)]
#[command(version = "0.1.0")]
pub struct Args {
    /// Configuration file path
    #[arg(short, long, value_name = "FILE")]
    pub config: Option<PathBuf>,

    /// Log level (trace, debug, info, warn, error)
    #[arg(short, long, value_name = "LEVEL", default_value = "info")]
    pub log_level: String,

    #[command(subcommand)]
    pub command: Command,
}

#[derive(Subcommand, Debug)]
pub enum Command {
    /// Setup data directories and download datasets
    Setup {
        /// Mode: full or demo
        #[arg(long, default_value = "demo")]
        mode: String,

        /// Force re-download of datasets
        #[arg(long)]
        force: bool,
    },

    /// Data operations
    Data {
        #[command(subcommand)]
        subcommand: DataCommand,
    },

    /// Training operations
    Train {
        #[command(subcommand)]
        subcommand: TrainCommand,
    },

    /// Evaluation operations
    Evaluate {
        /// Model checkpoint to evaluate
        #[arg(long)]
        checkpoint: Option<PathBuf>,

        /// Number of episodes
        #[arg(long, default_value = "100")]
        episodes: usize,
    },

    /// Benchmark operations
    Benchmark {
        /// Benchmark type: speed, memory, all
        #[arg(long, default_value = "all")]
        benchmark_type: String,

        /// Output format: json, csv, text
        #[arg(long, default_value = "text")]
        output_format: String,
    },

    /// Run a demo simulation
    Demo {
        /// Number of simulation steps
        #[arg(long, default_value = "1000")]
        steps: usize,

        /// Visualize (requires GUI)
        #[arg(long)]
        visualize: bool,
    },

    /// Launch GUI
    Gui,
}

#[derive(Subcommand, Debug)]
pub enum DataCommand {
    /// Inspect available datasets
    Inspect {
        /// Dataset name
        #[arg(long)]
        dataset: Option<String>,
    },

    /// Preprocess raw data
    Preprocess {
        /// Input directory
        #[arg(long)]
        input: Option<PathBuf>,

        /// Output directory
        #[arg(long)]
        output: Option<PathBuf>,

        /// Chunk size for streaming
        #[arg(long, default_value = "10000")]
        chunk_size: usize,
    },

    /// Download datasets
    Download {
        /// Dataset name: tsrd, jcwise, or all
        #[arg(long, default_value = "all")]
        dataset: String,

        /// Download mode: scan, stare
        #[arg(long, default_value = "scan")]
        mode: String,

        /// Data split: train, validation, test, all
        #[arg(long, default_value = "train")]
        split: String,

        /// Subset fraction (0.0 to 1.0)
        #[arg(long, default_value = "0.1")]
        subset: f32,
    },
}

#[derive(Subcommand, Debug)]
pub enum TrainCommand {
    /// Train deep learning model
    Dl {
        /// Number of epochs
        #[arg(long)]
        epochs: Option<usize>,

        /// Learning rate
        #[arg(long)]
        lr: Option<f32>,

        /// Batch size
        #[arg(long)]
        batch_size: Option<usize>,

        /// Save checkpoint
        #[arg(long)]
        checkpoint: Option<PathBuf>,
    },

    /// Train PPO scheduler
    Ppo {
        /// Number of training steps
        #[arg(long, default_value = "100000")]
        steps: usize,

        /// Learning rate
        #[arg(long)]
        lr: Option<f32>,

        /// Save checkpoint
        #[arg(long)]
        checkpoint: Option<PathBuf>,
    },

    /// Train all models (DL + PPO)
    All {
        /// Quick mode with smaller dataset
        #[arg(long)]
        quick: bool,
    },
}

impl Args {
    pub fn parse_args() -> Self {
        Parser::parse()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_command_parsing() {
        let args = Args {
            config: None,
            log_level: "info".to_string(),
            command: Command::Demo {
                steps: 100,
                visualize: false,
            },
        };
        assert_eq!(args.log_level, "info");
    }
}
