use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum ConfigError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("TOML deserialize error: {0}")]
    TomlDeError(#[from] toml::de::Error),

    #[error("TOML serialize error: {0}")]
    TomlSerError(toml::ser::Error),

    #[error("Configuration validation error: {0}")]
    ValidationError(String),
}

/// Global application configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    pub app: AppConfig,
    pub simulator: SimulatorConfig,
    pub training: TrainingConfig,
    pub evaluation: EvaluationConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppConfig {
    pub name: String,
    pub version: String,
    pub log_level: String,
    pub data_dir: String,
    pub models_dir: String,
    pub results_dir: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimulatorConfig {
    pub num_bands: usize,
    pub time_horizon: usize,
    pub num_emitters: usize,
    pub receiver_bandwidth_fraction: f32,
    pub detection_probability: f32,
    pub false_alarm_probability: f32,
    pub seed: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingConfig {
    pub batch_size: usize,
    pub learning_rate: f32,
    pub epochs: usize,
    pub validation_split: f32,
    pub checkpoint_dir: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvaluationConfig {
    pub num_test_episodes: usize,
    pub metrics_output: String,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            app: AppConfig {
                name: "SMARTSCAN".to_string(),
                version: "0.1.0".to_string(),
                log_level: "info".to_string(),
                data_dir: "data".to_string(),
                models_dir: "models".to_string(),
                results_dir: "results".to_string(),
            },
            simulator: SimulatorConfig {
                num_bands: 32,
                time_horizon: 1000,
                num_emitters: 5,
                receiver_bandwidth_fraction: 0.1,
                detection_probability: 0.95,
                false_alarm_probability: 0.01,
                seed: 42,
            },
            training: TrainingConfig {
                batch_size: 32,
                learning_rate: 0.001,
                epochs: 100,
                validation_split: 0.2,
                checkpoint_dir: "models".to_string(),
            },
            evaluation: EvaluationConfig {
                num_test_episodes: 100,
                metrics_output: "results/metrics".to_string(),
            },
        }
    }
}

impl Config {
    /// Load configuration from a TOML file
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self, ConfigError> {
        let content = fs::read_to_string(path)?;
        let config = toml::from_str(&content)?;
        Ok(config)
    }

    /// Save configuration to a TOML file
    pub fn to_file<P: AsRef<Path>>(&self, path: P) -> Result<(), ConfigError> {
        let content = toml::to_string_pretty(self).map_err(ConfigError::TomlSerError)?;
        fs::write(path, content)?;
        Ok(())
    }

    /// Validate configuration values
    pub fn validate(&self) -> Result<(), ConfigError> {
        if self.simulator.num_bands < 1 {
            return Err(ConfigError::ValidationError(
                "num_bands must be >= 1".to_string(),
            ));
        }

        if self.simulator.time_horizon < 1 {
            return Err(ConfigError::ValidationError(
                "time_horizon must be >= 1".to_string(),
            ));
        }

        if !(0.0..=1.0).contains(&self.simulator.detection_probability) {
            return Err(ConfigError::ValidationError(
                "detection_probability must be in [0, 1]".to_string(),
            ));
        }

        if !(0.0..=1.0).contains(&self.simulator.false_alarm_probability) {
            return Err(ConfigError::ValidationError(
                "false_alarm_probability must be in [0, 1]".to_string(),
            ));
        }

        if self.training.batch_size < 1 {
            return Err(ConfigError::ValidationError(
                "batch_size must be >= 1".to_string(),
            ));
        }

        if self.training.epochs < 1 {
            return Err(ConfigError::ValidationError(
                "epochs must be >= 1".to_string(),
            ));
        }

        if !(0.0..=1.0).contains(&self.training.validation_split) {
            return Err(ConfigError::ValidationError(
                "validation_split must be in [0, 1]".to_string(),
            ));
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = Config::default();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_config_validation() {
        let mut config = Config::default();
        config.simulator.num_bands = 0;
        assert!(config.validate().is_err());

        config.simulator.num_bands = 32;
        config.simulator.detection_probability = 1.5;
        assert!(config.validate().is_err());
    }
}
