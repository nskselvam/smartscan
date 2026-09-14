#[derive(Debug, Clone, PartialEq)]
pub struct RolloutStep {
    pub state: Vec<f32>,
    pub action: usize,
    pub reward: f32,
    pub value: f32,
    pub old_log_probability: f32,
    pub done: bool,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AdvantageEstimate {
    pub advantage: f32,
    pub return_value: f32,
}

pub fn compute_gae(
    steps: &[RolloutStep],
    bootstrap_value: f32,
    gamma: f32,
    gae_lambda: f32,
) -> Vec<AdvantageEstimate> {
    assert!((0.0..=1.0).contains(&gamma));
    assert!((0.0..=1.0).contains(&gae_lambda));
    let mut estimates = vec![
        AdvantageEstimate {
            advantage: 0.0,
            return_value: 0.0
        };
        steps.len()
    ];
    let mut next_value = bootstrap_value;
    let mut gae = 0.0;

    for index in (0..steps.len()).rev() {
        let step = &steps[index];
        let nonterminal = if step.done { 0.0 } else { 1.0 };
        let delta = step.reward + gamma * next_value * nonterminal - step.value;
        gae = delta + gamma * gae_lambda * nonterminal * gae;
        estimates[index] = AdvantageEstimate {
            advantage: gae,
            return_value: gae + step.value,
        };
        next_value = step.value;
    }
    estimates
}

pub fn normalize_advantages(estimates: &[AdvantageEstimate]) -> Vec<f32> {
    if estimates.is_empty() {
        return Vec::new();
    }
    let mean = estimates
        .iter()
        .map(|estimate| estimate.advantage)
        .sum::<f32>()
        / estimates.len() as f32;
    let variance = estimates
        .iter()
        .map(|estimate| (estimate.advantage - mean).powi(2))
        .sum::<f32>()
        / estimates.len() as f32;
    let standard_deviation = variance.sqrt().max(1e-8);
    estimates
        .iter()
        .map(|estimate| (estimate.advantage - mean) / standard_deviation)
        .collect()
}

pub fn clipped_surrogate_objective(
    new_log_probability: f32,
    old_log_probability: f32,
    advantage: f32,
    clip_epsilon: f32,
) -> f32 {
    assert!(clip_epsilon >= 0.0);
    let ratio = (new_log_probability - old_log_probability).exp();
    let clipped_ratio = ratio.clamp(1.0 - clip_epsilon, 1.0 + clip_epsilon);
    (ratio * advantage).min(clipped_ratio * advantage)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gae_respects_terminal_transitions() {
        let steps = vec![
            RolloutStep {
                state: vec![],
                action: 0,
                reward: 1.0,
                value: 0.5,
                old_log_probability: 0.0,
                done: false,
            },
            RolloutStep {
                state: vec![],
                action: 0,
                reward: 2.0,
                value: 0.25,
                old_log_probability: 0.0,
                done: true,
            },
        ];
        let estimates = compute_gae(&steps, 10.0, 1.0, 1.0);
        assert_eq!(estimates[1].advantage, 1.75);
        assert_eq!(estimates[0].advantage, 2.5);
        assert_eq!(estimates[0].return_value, 3.0);
    }

    #[test]
    fn clipped_objective_limits_positive_advantage_gain() {
        let objective = clipped_surrogate_objective((1.5_f32).ln(), 0.0, 2.0, 0.2);
        assert_eq!(objective, 2.4);
    }
}
