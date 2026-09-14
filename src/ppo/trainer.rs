use rand::rngs::StdRng;
use rand::SeedableRng;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;
use thiserror::Error;

use super::{
    compute_gae, normalize_advantages, CategoricalPolicy, PpoEnvironment, PpoError, RolloutStep,
    ValueNetwork,
};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PpoConfig {
    pub state_size: usize,
    pub action_count: usize,
    pub learning_rate: f32,
    pub gamma: f32,
    pub gae_lambda: f32,
    pub clip_epsilon: f32,
    pub entropy_coefficient: f32,
    pub value_loss_coefficient: f32,
    pub gradient_clip: f32,
    pub update_epochs: usize,
    pub seed: u64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PpoTrainingReport {
    pub rollout_steps: usize,
    pub total_reward: f32,
    pub policy_objective: f32,
    pub value_loss: f32,
    pub entropy: f32,
}

pub struct PpoAgent {
    config: PpoConfig,
    actor: CategoricalPolicy,
    critic: ValueNetwork,
    rng: StdRng,
}

#[derive(Debug, Serialize, Deserialize)]
struct PpoCheckpoint {
    config: PpoConfig,
    actor: CategoricalPolicy,
    critic: ValueNetwork,
}

#[derive(Debug, Error)]
pub enum PpoCheckpointError {
    #[error("PPO checkpoint I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("PPO checkpoint serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
}

impl PpoAgent {
    pub fn new(config: PpoConfig) -> Self {
        assert!(config.state_size > 0 && config.action_count > 0);
        assert!(config.learning_rate > 0.0 && config.gradient_clip > 0.0);
        assert!((0.0..=1.0).contains(&config.gamma));
        assert!((0.0..=1.0).contains(&config.gae_lambda));
        assert!(config.update_epochs > 0);
        let mut rng = StdRng::seed_from_u64(config.seed);
        let actor = CategoricalPolicy::new(config.action_count, config.state_size, &mut rng);
        let critic = ValueNetwork::new(config.state_size, &mut rng);
        Self {
            config,
            actor,
            critic,
            rng,
        }
    }

    pub fn action_probabilities(&self, state: &[f32]) -> Vec<f32> {
        self.actor.probabilities(state)
    }

    pub fn save(&self, path: impl AsRef<Path>) -> Result<(), PpoCheckpointError> {
        let path = path.as_ref();
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        let checkpoint = PpoCheckpoint {
            config: self.config.clone(),
            actor: self.actor.clone(),
            critic: self.critic.clone(),
        };
        fs::write(path, serde_json::to_vec(&checkpoint)?)?;
        Ok(())
    }

    pub fn load(path: impl AsRef<Path>) -> Result<Self, PpoCheckpointError> {
        let checkpoint: PpoCheckpoint = serde_json::from_slice(&fs::read(path)?)?;
        Ok(Self {
            rng: StdRng::seed_from_u64(checkpoint.config.seed),
            config: checkpoint.config,
            actor: checkpoint.actor,
            critic: checkpoint.critic,
        })
    }

    pub fn is_compatible(&self, state_size: usize, action_count: usize) -> bool {
        self.config.state_size == state_size && self.config.action_count == action_count
    }

    pub fn train_episode(
        &mut self,
        environment: &mut PpoEnvironment,
        max_steps: usize,
    ) -> Result<PpoTrainingReport, PpoError> {
        assert_eq!(environment.num_actions(), self.config.action_count);
        let mut state = environment.reset().to_vector();
        assert_eq!(state.len(), self.config.state_size);
        let mut rollout = Vec::new();
        let mut total_reward = 0.0;

        for _ in 0..max_steps {
            let decision = self.actor.decide(&state, &mut self.rng);
            let value = self.critic.predict(&state);
            let transition = environment.step(decision.action)?;
            total_reward += transition.reward;
            rollout.push(RolloutStep {
                state: state.clone(),
                action: decision.action,
                reward: transition.reward,
                value,
                old_log_probability: decision.log_probability,
                done: transition.done,
            });
            state = transition.observation.to_vector();
            if transition.done {
                break;
            }
        }

        let bootstrap_value = if rollout.last().is_some_and(|step| step.done) {
            0.0
        } else {
            self.critic.predict(&state)
        };
        let estimates = compute_gae(
            &rollout,
            bootstrap_value,
            self.config.gamma,
            self.config.gae_lambda,
        );
        let advantages = normalize_advantages(&estimates);
        let mut policy_objective = 0.0;
        let mut value_loss = 0.0;
        let mut entropy = 0.0;

        for _ in 0..self.config.update_epochs {
            for (index, step) in rollout.iter().enumerate() {
                let (objective, step_entropy) = self.actor.update(
                    &step.state,
                    step.action,
                    step.old_log_probability,
                    advantages[index],
                    &self.config,
                );
                policy_objective += objective;
                entropy += step_entropy;
                value_loss += self.critic.update(
                    &step.state,
                    estimates[index].return_value,
                    self.config.learning_rate * self.config.value_loss_coefficient,
                    self.config.gradient_clip,
                );
            }
        }
        let update_count = (rollout.len() * self.config.update_epochs).max(1) as f32;
        Ok(PpoTrainingReport {
            rollout_steps: rollout.len(),
            total_reward,
            policy_objective: policy_objective / update_count,
            value_loss: value_loss / update_count,
            entropy: entropy / update_count,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simulator::Scenario;

    fn config(bands: usize) -> PpoConfig {
        PpoConfig {
            state_size: 3 + 5 * bands,
            action_count: bands,
            learning_rate: 0.01,
            gamma: 0.99,
            gae_lambda: 0.95,
            clip_epsilon: 0.2,
            entropy_coefficient: 0.01,
            value_loss_coefficient: 0.5,
            gradient_clip: 1.0,
            update_epochs: 2,
            seed: 42,
        }
    }

    #[test]
    fn agent_collects_and_updates_from_a_rollout() {
        let scenario = Scenario::periodic();
        let bands = scenario.num_bands;
        let mut environment = PpoEnvironment::new(scenario.into_environment());
        let mut agent = PpoAgent::new(config(bands));
        let report = agent
            .train_episode(&mut environment, 20)
            .expect("training should succeed");
        assert_eq!(report.rollout_steps, 20);
        assert!(report.value_loss.is_finite());
        assert!(report.entropy.is_finite());
    }

    #[test]
    fn checkpoint_restores_policy_probabilities() {
        let path = std::env::temp_dir().join(format!("smartscan-ppo-{}.json", std::process::id()));
        let agent = PpoAgent::new(config(2));
        let state = vec![0.25; 13];
        let expected = agent.action_probabilities(&state);
        agent.save(&path).expect("checkpoint should save");
        let loaded = PpoAgent::load(&path).expect("checkpoint should load");
        assert_eq!(loaded.action_probabilities(&state), expected);
        std::fs::remove_file(path).expect("checkpoint should be removed");
    }
}
