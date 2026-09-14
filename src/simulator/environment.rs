use rand::rngs::StdRng;
use rand::SeedableRng;

use super::emitter::Emitter;
use super::ground_truth::GroundTruth;
use super::receiver::Receiver;

#[derive(Debug, Clone)]
pub struct RewardConfig {
    pub detection_reward: f32,
    pub missed_transmission_penalty: f32,
    pub false_alarm_penalty: f32,
    pub inactive_scan_penalty: f32,
}

impl Default for RewardConfig {
    fn default() -> Self {
        Self {
            detection_reward: 1.0,
            missed_transmission_penalty: 0.25,
            false_alarm_penalty: 0.5,
            inactive_scan_penalty: 0.05,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct StepInfo {
    pub active_bands: usize,
    pub monitored_bands: usize,
    pub monitored_active_bands: usize,
    pub monitored_inactive_bands: usize,
    pub transmission_starts: usize,
    pub true_positives: usize,
    pub false_positives: usize,
    pub missed_active_bands: usize,
    pub interceptions: usize,
    pub intercept_delay_sum: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct StepResult {
    pub observation: Vec<bool>,
    pub reward: f32,
    pub done: bool,
    pub selected_band: usize,
    pub info: StepInfo,
}

pub struct RfEnvironment {
    pub num_bands: usize,
    pub time_horizon: usize,
    pub current_time: usize,
    pub ground_truth: GroundTruth,
    receiver: Receiver,
    emitters: Vec<Emitter>,
    reward_config: RewardConfig,
    seed: u64,
    observation_rng: StdRng,
    active_start: Vec<Option<usize>>,
    detected_current_window: Vec<bool>,
}

impl RfEnvironment {
    pub fn new(
        num_bands: usize,
        time_horizon: usize,
        receiver: Receiver,
        emitters: Vec<Emitter>,
        seed: u64,
    ) -> Self {
        Self::with_reward_config(
            num_bands,
            time_horizon,
            receiver,
            emitters,
            seed,
            RewardConfig::default(),
        )
    }

    pub fn with_reward_config(
        num_bands: usize,
        time_horizon: usize,
        receiver: Receiver,
        emitters: Vec<Emitter>,
        seed: u64,
        reward_config: RewardConfig,
    ) -> Self {
        assert!(num_bands > 0, "num_bands must be positive");
        assert!(time_horizon > 0, "time_horizon must be positive");

        let ground_truth = Self::build_ground_truth(num_bands, time_horizon, &emitters, seed);
        Self {
            num_bands,
            time_horizon,
            current_time: 0,
            ground_truth,
            receiver,
            emitters,
            reward_config,
            seed,
            observation_rng: StdRng::seed_from_u64(seed.wrapping_add(1)),
            active_start: vec![None; num_bands],
            detected_current_window: vec![false; num_bands],
        }
    }

    fn build_ground_truth(
        num_bands: usize,
        time_horizon: usize,
        emitters: &[Emitter],
        seed: u64,
    ) -> GroundTruth {
        let mut ground_truth = GroundTruth::new(num_bands, time_horizon);
        let mut rng = StdRng::seed_from_u64(seed);

        for time in 0..time_horizon {
            for emitter in emitters {
                if emitter.is_transmitting(time, &mut rng) {
                    let band = emitter.band_at(time, num_bands, &mut rng);
                    ground_truth.set_transmission(time, band, true);
                }
            }
        }
        ground_truth
    }

    pub fn step(&mut self, selected_band: usize) -> StepResult {
        assert!(
            selected_band < self.num_bands,
            "selected band is out of range"
        );
        assert!(
            self.current_time < self.time_horizon,
            "episode has already ended"
        );

        let monitored = self.receiver.monitored_bands(self.num_bands, selected_band);
        let mut observation = vec![false; self.num_bands];
        let mut true_positives = 0;
        let mut false_positives = 0;
        let mut active_bands = 0;
        let mut observed_active_bands = 0;
        let mut transmission_starts = 0;
        let mut interceptions = 0;
        let mut intercept_delay_sum = 0;

        for band in 0..self.num_bands {
            let transmitting = self.ground_truth.is_transmitting(self.current_time, band);
            if transmitting {
                active_bands += 1;
                if self.active_start[band].is_none() {
                    self.active_start[band] = Some(self.current_time);
                    self.detected_current_window[band] = false;
                    transmission_starts += 1;
                }
            } else {
                self.active_start[band] = None;
                self.detected_current_window[band] = false;
            }
            if !monitored[band] {
                continue;
            }
            if transmitting {
                observed_active_bands += 1;
            }

            let (detected, false_alarm) = self
                .receiver
                .observe(transmitting, &mut self.observation_rng);
            observation[band] = detected;
            if detected && transmitting {
                true_positives += 1;
                if !self.detected_current_window[band] {
                    interceptions += 1;
                    intercept_delay_sum +=
                        self.current_time - self.active_start[band].unwrap_or(self.current_time);
                    self.detected_current_window[band] = true;
                }
            }
            if false_alarm {
                false_positives += 1;
            }
        }

        let missed_active_bands = active_bands - true_positives;
        let monitored_bands = monitored.iter().filter(|monitored| **monitored).count();
        let reward = self.reward_config.detection_reward * true_positives as f32
            - self.reward_config.missed_transmission_penalty * missed_active_bands as f32
            - self.reward_config.false_alarm_penalty * false_positives as f32
            - if observed_active_bands == 0 {
                self.reward_config.inactive_scan_penalty
            } else {
                0.0
            };

        self.current_time += 1;
        StepResult {
            observation,
            reward,
            done: self.current_time == self.time_horizon,
            selected_band,
            info: StepInfo {
                active_bands,
                monitored_bands,
                monitored_active_bands: observed_active_bands,
                monitored_inactive_bands: monitored_bands - observed_active_bands,
                transmission_starts,
                true_positives,
                false_positives,
                missed_active_bands,
                interceptions,
                intercept_delay_sum,
            },
        }
    }

    pub fn reset(&mut self) {
        self.current_time = 0;
        self.ground_truth =
            Self::build_ground_truth(self.num_bands, self.time_horizon, &self.emitters, self.seed);
        self.observation_rng = StdRng::seed_from_u64(self.seed.wrapping_add(1));
        self.active_start.fill(None);
        self.detected_current_window.fill(false);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simulator::emitter::EmitterType;

    fn test_environment(seed: u64) -> RfEnvironment {
        RfEnvironment::new(
            8,
            4,
            Receiver::new(1.0, 1.0, 0.0),
            vec![Emitter::new(0, 1.0e9, EmitterType::FixedFrequency)],
            seed,
        )
    }

    #[test]
    fn step_detects_a_signal_with_a_perfect_receiver() {
        let mut environment = test_environment(42);
        let result = environment.step(4);
        assert_eq!(result.info.true_positives, 1);
        assert_eq!(result.info.false_positives, 0);
        assert_eq!(result.info.missed_active_bands, 0);
        assert_eq!(result.info.interceptions, 1);
        assert_eq!(result.info.intercept_delay_sum, 0);
        assert!(result.reward > 0.0);
    }

    #[test]
    fn same_seed_produces_equal_truth_and_observations() {
        let receiver = Receiver::new(0.25, 0.8, 0.1);
        let emitter = Emitter::new(0, 1.0e9, EmitterType::Intermittent).with_period(1, 0.5);
        let mut first = RfEnvironment::new(16, 10, receiver.clone(), vec![emitter.clone()], 9);
        let mut second = RfEnvironment::new(16, 10, receiver, vec![emitter], 9);

        assert_eq!(first.ground_truth.data, second.ground_truth.data);
        for _ in 0..10 {
            assert_eq!(first.step(8), second.step(8));
        }
    }

    #[test]
    fn reset_replays_the_same_episode() {
        let mut environment = test_environment(42);
        let first = environment.step(4);
        environment.reset();
        assert_eq!(first, environment.step(4));
    }
}
