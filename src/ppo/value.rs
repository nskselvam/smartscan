use rand::Rng;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValueNetwork {
    weights: Vec<f32>,
    bias: f32,
}

impl ValueNetwork {
    pub fn new(state_size: usize, rng: &mut impl Rng) -> Self {
        assert!(state_size > 0);
        let bound = (6.0 / (state_size + 1) as f32).sqrt();
        Self {
            weights: (0..state_size)
                .map(|_| rng.gen_range(-bound..bound))
                .collect(),
            bias: 0.0,
        }
    }

    pub fn predict(&self, state: &[f32]) -> f32 {
        assert_eq!(state.len(), self.weights.len());
        self.weights
            .iter()
            .zip(state)
            .map(|(weight, value)| weight * value)
            .sum::<f32>()
            + self.bias
    }

    pub fn update(
        &mut self,
        state: &[f32],
        target: f32,
        learning_rate: f32,
        gradient_clip: f32,
    ) -> f32 {
        let error = self.predict(state) - target;
        for (weight, feature) in self.weights.iter_mut().zip(state) {
            *weight -= learning_rate * (error * feature).clamp(-gradient_clip, gradient_clip);
        }
        self.bias -= learning_rate * error.clamp(-gradient_clip, gradient_clip);
        0.5 * error * error
    }
}
