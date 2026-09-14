use rand::Rng;

use super::rollout::clipped_surrogate_objective;
use super::trainer::PpoConfig;

#[derive(Debug, Clone, PartialEq)]
pub struct PolicyDecision {
    pub action: usize,
    pub probabilities: Vec<f32>,
    pub log_probability: f32,
}

#[derive(Debug, Clone)]
pub struct CategoricalPolicy {
    action_count: usize,
    state_size: usize,
    weights: Vec<f32>,
    bias: Vec<f32>,
}

impl CategoricalPolicy {
    pub fn new(action_count: usize, state_size: usize, rng: &mut impl Rng) -> Self {
        assert!(action_count > 0 && state_size > 0);
        let bound = (6.0 / (action_count + state_size) as f32).sqrt();
        Self {
            action_count,
            state_size,
            weights: (0..action_count * state_size)
                .map(|_| rng.gen_range(-bound..bound))
                .collect(),
            bias: vec![0.0; action_count],
        }
    }

    pub fn probabilities(&self, state: &[f32]) -> Vec<f32> {
        assert_eq!(state.len(), self.state_size);
        softmax(
            self.weights
                .chunks_exact(self.state_size)
                .zip(&self.bias)
                .map(|(weights, bias)| dot(weights, state) + bias)
                .collect(),
        )
    }

    pub fn decide(&self, state: &[f32], rng: &mut impl Rng) -> PolicyDecision {
        let probabilities = self.probabilities(state);
        let draw = rng.gen::<f32>();
        let mut cumulative = 0.0;
        let mut action = self.action_count - 1;
        for (index, probability) in probabilities.iter().enumerate() {
            cumulative += probability;
            if draw <= cumulative {
                action = index;
                break;
            }
        }
        PolicyDecision {
            action,
            log_probability: probabilities[action].max(1e-8).ln(),
            probabilities,
        }
    }

    pub fn update(
        &mut self,
        state: &[f32],
        action: usize,
        old_log_probability: f32,
        advantage: f32,
        config: &PpoConfig,
    ) -> (f32, f32) {
        let probabilities = self.probabilities(state);
        let new_log_probability = probabilities[action].max(1e-8).ln();
        let ratio = (new_log_probability - old_log_probability).exp();
        let clipped = (advantage >= 0.0 && ratio > 1.0 + config.clip_epsilon)
            || (advantage < 0.0 && ratio < 1.0 - config.clip_epsilon);
        let policy_scale = if clipped { 0.0 } else { ratio * advantage };
        let entropy = -probabilities
            .iter()
            .map(|probability| probability * probability.max(1e-8).ln())
            .sum::<f32>();

        for (band, probability) in probabilities.iter().enumerate() {
            let log_probability_gradient =
                if band == action { 1.0 } else { 0.0 } - probabilities[band];
            let entropy_gradient = -*probability * (probability.max(1e-8).ln() + entropy);
            let logit_gradient = policy_scale * log_probability_gradient
                + config.entropy_coefficient * entropy_gradient;
            let weights = &mut self.weights[band * self.state_size..(band + 1) * self.state_size];
            for (weight, feature) in weights.iter_mut().zip(state) {
                *weight += config.learning_rate
                    * (logit_gradient * feature).clamp(-config.gradient_clip, config.gradient_clip);
            }
            self.bias[band] += config.learning_rate
                * logit_gradient.clamp(-config.gradient_clip, config.gradient_clip);
        }
        (
            clipped_surrogate_objective(
                new_log_probability,
                old_log_probability,
                advantage,
                config.clip_epsilon,
            ),
            entropy,
        )
    }
}

fn dot(left: &[f32], right: &[f32]) -> f32 {
    left.iter()
        .zip(right)
        .map(|(left, right)| left * right)
        .sum()
}

fn softmax(logits: Vec<f32>) -> Vec<f32> {
    let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exponentials = logits
        .iter()
        .map(|logit| (logit - max_logit).exp())
        .collect::<Vec<_>>();
    let sum = exponentials.iter().sum::<f32>();
    exponentials.into_iter().map(|value| value / sum).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    #[test]
    fn categorical_policy_returns_a_probability_distribution() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(7);
        let policy = CategoricalPolicy::new(3, 2, &mut rng);
        let probabilities = policy.probabilities(&[0.5, -0.5]);
        assert_eq!(probabilities.len(), 3);
        assert!((probabilities.iter().sum::<f32>() - 1.0).abs() < 1e-6);
    }
}
