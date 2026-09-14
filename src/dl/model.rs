use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;
use thiserror::Error;

use crate::data::FeatureVector;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GruConfig {
    pub input_size: usize,
    pub hidden_size: usize,
    pub num_layers: usize,
    pub output_size: usize,
    pub sequence_length: usize,
    pub dropout: f32,
}

impl GruConfig {
    pub fn new(
        input_size: usize,
        hidden_size: usize,
        num_layers: usize,
        output_size: usize,
        sequence_length: usize,
        dropout: f32,
    ) -> Self {
        assert!(input_size > 0, "input_size must be positive");
        assert!(hidden_size > 0, "hidden_size must be positive");
        assert!(num_layers > 0, "num_layers must be positive");
        assert!(output_size > 0, "output_size must be positive");
        assert!(sequence_length > 0, "sequence_length must be positive");
        assert!((0.0..1.0).contains(&dropout), "dropout must be in [0, 1)");
        Self {
            input_size,
            hidden_size,
            num_layers,
            output_size,
            sequence_length,
            dropout,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct GruLayer {
    update_input: Vec<f32>,
    update_hidden: Vec<f32>,
    update_bias: Vec<f32>,
    reset_input: Vec<f32>,
    reset_hidden: Vec<f32>,
    reset_bias: Vec<f32>,
    candidate_input: Vec<f32>,
    candidate_hidden: Vec<f32>,
    candidate_bias: Vec<f32>,
    input_size: usize,
    hidden_size: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GruActivityPredictor {
    config: GruConfig,
    layers: Vec<GruLayer>,
    output_weights: Vec<f32>,
    output_bias: Vec<f32>,
}

#[derive(Debug, Error)]
pub enum ModelError {
    #[error("model I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("model serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
}

impl GruActivityPredictor {
    pub fn new(config: GruConfig, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut layers = Vec::with_capacity(config.num_layers);
        for layer_index in 0..config.num_layers {
            let input_size = if layer_index == 0 {
                config.input_size
            } else {
                config.hidden_size
            };
            layers.push(GruLayer::new(input_size, config.hidden_size, &mut rng));
        }
        let output_weights = random_matrix(config.output_size, config.hidden_size, &mut rng);
        let output_size = config.output_size;
        Self {
            config,
            layers,
            output_weights,
            output_bias: vec![0.0; output_size],
        }
    }

    pub fn config(&self) -> &GruConfig {
        &self.config
    }

    pub fn save(&self, path: impl AsRef<Path>) -> Result<(), ModelError> {
        let path = path.as_ref();
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(path, serde_json::to_vec(self)?)?;
        Ok(())
    }

    pub fn load(path: impl AsRef<Path>) -> Result<Self, ModelError> {
        Ok(serde_json::from_slice(&fs::read(path)?)?)
    }

    /// Produces P(activity | band, future_time) for each candidate output band.
    pub fn predict(&self, history: &[FeatureVector]) -> Vec<f32> {
        let sequence = history
            .iter()
            .map(|features| features.values[..self.config.input_size].to_vec())
            .collect::<Vec<_>>();
        self.forward(&sequence)
    }

    pub fn forward(&self, sequence: &[Vec<f32>]) -> Vec<f32> {
        self.output_probabilities(&self.encode(sequence))
    }

    /// Updates only the output activity head. The recurrent encoder remains
    /// fixed until full BPTT is introduced in a later training backend.
    pub fn train_output_head(
        &mut self,
        history: &[FeatureVector],
        target_activity: &[f32],
        learning_rate: f32,
    ) -> f32 {
        assert_eq!(target_activity.len(), self.config.output_size);
        assert!(learning_rate > 0.0, "learning_rate must be positive");
        let sequence = history
            .iter()
            .map(|features| features.values[..self.config.input_size].to_vec())
            .collect::<Vec<_>>();
        let hidden = self.encode(&sequence);
        let predictions = self.output_probabilities(&hidden);

        for (band, (&prediction, &target)) in predictions.iter().zip(target_activity).enumerate() {
            assert!(
                (0.0..=1.0).contains(&target),
                "target activity must be in [0, 1]"
            );
            let error = prediction - target;
            let weights = &mut self.output_weights
                [band * self.config.hidden_size..(band + 1) * self.config.hidden_size];
            for (weight, hidden_value) in weights.iter_mut().zip(&hidden) {
                *weight -= learning_rate * error * hidden_value;
            }
            self.output_bias[band] -= learning_rate * error;
        }
        binary_cross_entropy(&predictions, target_activity)
    }

    fn encode(&self, sequence: &[Vec<f32>]) -> Vec<f32> {
        assert!(!sequence.is_empty(), "history must not be empty");
        assert!(
            sequence.len() <= self.config.sequence_length,
            "history exceeds configured sequence_length"
        );
        for input in sequence {
            assert_eq!(
                input.len(),
                self.config.input_size,
                "input feature width differs from config"
            );
        }

        let mut hidden_states = vec![vec![0.0; self.config.hidden_size]; self.config.num_layers];
        for input in sequence {
            let mut layer_input = input.clone();
            for (layer_index, layer) in self.layers.iter().enumerate() {
                hidden_states[layer_index] = layer.step(&layer_input, &hidden_states[layer_index]);
                layer_input = hidden_states[layer_index].clone();
            }
        }

        hidden_states[self.config.num_layers - 1].clone()
    }

    fn output_probabilities(&self, hidden: &[f32]) -> Vec<f32> {
        matrix_vector_product(
            &self.output_weights,
            self.config.output_size,
            self.config.hidden_size,
            hidden,
            &self.output_bias,
        )
        .into_iter()
        .map(sigmoid)
        .collect()
    }
}

pub fn binary_cross_entropy(predictions: &[f32], targets: &[f32]) -> f32 {
    assert_eq!(predictions.len(), targets.len());
    let epsilon = 1e-7;
    predictions
        .iter()
        .zip(targets)
        .map(|(prediction, target)| {
            let probability = prediction.clamp(epsilon, 1.0 - epsilon);
            -(target * probability.ln() + (1.0 - target) * (1.0 - probability).ln())
        })
        .sum::<f32>()
        / predictions.len().max(1) as f32
}

impl GruLayer {
    fn new(input_size: usize, hidden_size: usize, rng: &mut impl Rng) -> Self {
        Self {
            update_input: random_matrix(hidden_size, input_size, rng),
            update_hidden: random_matrix(hidden_size, hidden_size, rng),
            update_bias: vec![0.0; hidden_size],
            reset_input: random_matrix(hidden_size, input_size, rng),
            reset_hidden: random_matrix(hidden_size, hidden_size, rng),
            reset_bias: vec![0.0; hidden_size],
            candidate_input: random_matrix(hidden_size, input_size, rng),
            candidate_hidden: random_matrix(hidden_size, hidden_size, rng),
            candidate_bias: vec![0.0; hidden_size],
            input_size,
            hidden_size,
        }
    }

    fn step(&self, input: &[f32], previous_hidden: &[f32]) -> Vec<f32> {
        let update = combine_gate(
            matrix_vector_product(
                &self.update_input,
                self.hidden_size,
                self.input_size,
                input,
                &self.update_bias,
            ),
            matrix_vector_product(
                &self.update_hidden,
                self.hidden_size,
                self.hidden_size,
                previous_hidden,
                &vec![0.0; self.hidden_size],
            ),
            sigmoid,
        );
        let reset = combine_gate(
            matrix_vector_product(
                &self.reset_input,
                self.hidden_size,
                self.input_size,
                input,
                &self.reset_bias,
            ),
            matrix_vector_product(
                &self.reset_hidden,
                self.hidden_size,
                self.hidden_size,
                previous_hidden,
                &vec![0.0; self.hidden_size],
            ),
            sigmoid,
        );
        let reset_hidden = previous_hidden
            .iter()
            .zip(&reset)
            .map(|(hidden, reset)| hidden * reset)
            .collect::<Vec<_>>();
        let candidate = combine_gate(
            matrix_vector_product(
                &self.candidate_input,
                self.hidden_size,
                self.input_size,
                input,
                &self.candidate_bias,
            ),
            matrix_vector_product(
                &self.candidate_hidden,
                self.hidden_size,
                self.hidden_size,
                &reset_hidden,
                &vec![0.0; self.hidden_size],
            ),
            f32::tanh,
        );
        update
            .iter()
            .zip(previous_hidden)
            .zip(candidate)
            .map(|((update, previous), candidate)| update * previous + (1.0 - update) * candidate)
            .collect()
    }
}

fn random_matrix(rows: usize, columns: usize, rng: &mut impl Rng) -> Vec<f32> {
    let bound = (6.0_f32 / (rows + columns) as f32).sqrt();
    (0..rows * columns)
        .map(|_| rng.gen_range(-bound..bound))
        .collect()
}

fn matrix_vector_product(
    matrix: &[f32],
    rows: usize,
    columns: usize,
    vector: &[f32],
    bias: &[f32],
) -> Vec<f32> {
    assert_eq!(matrix.len(), rows * columns);
    assert_eq!(vector.len(), columns);
    assert_eq!(bias.len(), rows);
    matrix
        .chunks_exact(columns)
        .zip(bias)
        .map(|(row, bias)| {
            row.iter()
                .zip(vector)
                .map(|(weight, value)| weight * value)
                .sum::<f32>()
                + bias
        })
        .collect()
}

fn combine_gate(
    input_projection: Vec<f32>,
    hidden_projection: Vec<f32>,
    activation: impl Fn(f32) -> f32,
) -> Vec<f32> {
    input_projection
        .into_iter()
        .zip(hidden_projection)
        .map(|(input, hidden)| activation(input + hidden))
        .collect()
}

fn sigmoid(value: f32) -> f32 {
    1.0 / (1.0 + (-value).exp())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::features::FEATURE_COUNT;

    fn feature(value: f32) -> FeatureVector {
        FeatureVector {
            values: [value; FEATURE_COUNT],
        }
    }

    #[test]
    fn predictions_match_the_configured_band_count_and_probability_range() {
        let model = GruActivityPredictor::new(GruConfig::new(10, 8, 2, 4, 3, 0.0), 42);
        let predictions = model.predict(&[feature(0.1), feature(0.2), feature(0.3)]);
        assert_eq!(predictions.len(), 4);
        assert!(predictions
            .iter()
            .all(|probability| (0.0..=1.0).contains(probability)));
    }

    #[test]
    fn seeded_models_produce_identical_predictions() {
        let config = GruConfig::new(10, 8, 1, 2, 2, 0.0);
        let first = GruActivityPredictor::new(config.clone(), 7);
        let second = GruActivityPredictor::new(config, 7);
        let history = [feature(0.25), feature(0.5)];
        assert_eq!(first.predict(&history), second.predict(&history));
    }

    #[test]
    fn saved_model_reloads_with_identical_predictions() {
        let model = GruActivityPredictor::new(GruConfig::new(10, 4, 1, 2, 2, 0.0), 11);
        let path = std::env::temp_dir().join(format!("smartscan-gru-{}.json", std::process::id()));
        model.save(&path).expect("model should save");
        let reloaded = GruActivityPredictor::load(&path).expect("model should load");
        assert_eq!(
            model.predict(&[feature(0.1)]),
            reloaded.predict(&[feature(0.1)])
        );
        fs::remove_file(path).expect("checkpoint should be removed");
    }
}
