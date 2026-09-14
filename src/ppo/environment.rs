use thiserror::Error;

use crate::simulator::RfEnvironment;

#[derive(Debug, Clone, PartialEq)]
pub struct PpoObservation {
    pub normalized_time: f32,
    pub predicted_activity: Vec<f32>,
    pub time_since_scan: Vec<f32>,
    pub time_since_detection: Vec<f32>,
    pub observed_activity_rate: Vec<f32>,
    pub exploration_score: Vec<f32>,
    pub previous_action: Option<usize>,
    pub previous_reward: f32,
}

impl PpoObservation {
    pub fn to_vector(&self) -> Vec<f32> {
        let band_count = self.predicted_activity.len();
        let mut values = Vec::with_capacity(3 + 5 * band_count);
        values.push(self.normalized_time);
        values.push(
            self.previous_action
                .map(|action| action as f32 / band_count.max(1) as f32)
                .unwrap_or(0.0),
        );
        values.push(self.previous_reward);
        for band in 0..band_count {
            values.extend([
                self.predicted_activity[band],
                self.time_since_scan[band],
                self.time_since_detection[band],
                self.observed_activity_rate[band],
                self.exploration_score[band],
            ]);
        }
        values
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct PpoTransition {
    pub observation: PpoObservation,
    pub reward: f32,
    pub done: bool,
    pub action: usize,
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum PpoError {
    #[error("action {action} is outside the {num_bands}-band action space")]
    InvalidAction { action: usize, num_bands: usize },
    #[error("predicted activity width does not match the action space")]
    PredictionWidthMismatch,
    #[error("episode is already complete")]
    EpisodeComplete,
}

pub struct PpoEnvironment {
    environment: RfEnvironment,
    predicted_activity: Vec<f32>,
    last_scan: Vec<Option<usize>>,
    last_detection: Vec<Option<usize>>,
    observations: Vec<usize>,
    detections: Vec<usize>,
    visits: Vec<usize>,
    previous_action: Option<usize>,
    previous_reward: f32,
}

impl PpoEnvironment {
    pub fn new(environment: RfEnvironment) -> Self {
        let num_bands = environment.num_bands;
        Self {
            environment,
            predicted_activity: vec![0.5; num_bands],
            last_scan: vec![None; num_bands],
            last_detection: vec![None; num_bands],
            observations: vec![0; num_bands],
            detections: vec![0; num_bands],
            visits: vec![0; num_bands],
            previous_action: None,
            previous_reward: 0.0,
        }
    }

    pub fn num_actions(&self) -> usize {
        self.environment.num_bands
    }

    pub fn set_predicted_activity(&mut self, predictions: Vec<f32>) -> Result<(), PpoError> {
        if predictions.len() != self.num_actions()
            || predictions.iter().any(|value| !(0.0..=1.0).contains(value))
        {
            return Err(PpoError::PredictionWidthMismatch);
        }
        self.predicted_activity = predictions;
        Ok(())
    }

    pub fn observation(&self) -> PpoObservation {
        let current_time = self.environment.current_time;
        let horizon = self.environment.time_horizon.max(1);
        let normalized_age = |last_time: Option<usize>| match last_time {
            Some(time) => (current_time - time).min(horizon) as f32 / horizon as f32,
            None => 1.0,
        };
        PpoObservation {
            normalized_time: current_time as f32 / horizon as f32,
            predicted_activity: self.predicted_activity.clone(),
            time_since_scan: self.last_scan.iter().copied().map(normalized_age).collect(),
            time_since_detection: self
                .last_detection
                .iter()
                .copied()
                .map(normalized_age)
                .collect(),
            observed_activity_rate: self
                .detections
                .iter()
                .zip(&self.observations)
                .map(|(detections, observations)| {
                    *detections as f32 / (*observations).max(1) as f32
                })
                .collect(),
            exploration_score: self
                .visits
                .iter()
                .map(|visits| 1.0 / (*visits as f32 + 1.0).sqrt())
                .collect(),
            previous_action: self.previous_action,
            previous_reward: self.previous_reward,
        }
    }

    pub fn step(&mut self, action: usize) -> Result<PpoTransition, PpoError> {
        if action >= self.num_actions() {
            return Err(PpoError::InvalidAction {
                action,
                num_bands: self.num_actions(),
            });
        }
        if self.environment.current_time >= self.environment.time_horizon {
            return Err(PpoError::EpisodeComplete);
        }

        let time = self.environment.current_time;
        let result = self.environment.step(action);
        for band in 0..self.num_actions() {
            if result.monitored_bands[band] {
                self.last_scan[band] = Some(time);
                self.observations[band] += 1;
                if result.observation[band] {
                    self.last_detection[band] = Some(time);
                    self.detections[band] += 1;
                }
            }
        }
        self.visits[action] += 1;
        self.previous_action = Some(action);
        self.previous_reward = result.reward;

        Ok(PpoTransition {
            observation: self.observation(),
            reward: result.reward,
            done: result.done,
            action,
        })
    }

    pub fn reset(&mut self) -> PpoObservation {
        self.environment.reset();
        self.last_scan.fill(None);
        self.last_detection.fill(None);
        self.observations.fill(0);
        self.detections.fill(0);
        self.visits.fill(0);
        self.previous_action = None;
        self.previous_reward = 0.0;
        self.observation()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simulator::Scenario;

    #[test]
    fn observation_has_only_state_features_and_expected_width() {
        let environment = Scenario::periodic().into_environment();
        let ppo_environment = PpoEnvironment::new(environment);
        let observation = ppo_environment.observation();
        assert_eq!(
            observation.to_vector().len(),
            3 + 5 * ppo_environment.num_actions()
        );
        assert!(observation
            .predicted_activity
            .iter()
            .all(|value| *value == 0.5));
    }

    #[test]
    fn valid_action_updates_receiver_observation_state() {
        let environment = Scenario::fixed_frequency().into_environment();
        let mut ppo_environment = PpoEnvironment::new(environment);
        let transition = ppo_environment.step(16).expect("action should be accepted");
        assert_eq!(transition.action, 16);
        assert_eq!(transition.observation.previous_action, Some(16));
        assert_eq!(transition.observation.time_since_scan[16], 1.0 / 1_000.0);
    }

    #[test]
    fn invalid_actions_are_rejected_without_stepping_the_simulator() {
        let environment = Scenario::fixed_frequency().into_environment();
        let mut ppo_environment = PpoEnvironment::new(environment);
        let error = ppo_environment
            .step(32)
            .expect_err("action should be rejected");
        assert_eq!(
            error,
            PpoError::InvalidAction {
                action: 32,
                num_bands: 32
            }
        );
        assert_eq!(ppo_environment.observation().normalized_time, 0.0);
    }
}
