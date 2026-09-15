use super::emitter::{Emitter, EmitterType};
use super::environment::RfEnvironment;
use super::receiver::Receiver;

/// Reproducible RF simulator configuration and emitter scenario.
#[derive(Debug, Clone)]
pub struct Scenario {
    pub name: String,
    pub description: String,
    pub num_bands: usize,
    pub time_horizon: usize,
    pub receiver: Receiver,
    pub emitters: Vec<Emitter>,
    pub seed: u64,
}

impl Scenario {
    pub fn new(name: String, description: String) -> Self {
        Self {
            name,
            description,
            num_bands: 32,
            time_horizon: 1_000,
            receiver: Receiver::default(),
            emitters: Vec::new(),
            seed: 42,
        }
    }

    pub fn into_environment(self) -> RfEnvironment {
        RfEnvironment::new(
            self.num_bands,
            self.time_horizon,
            self.receiver,
            self.emitters,
            self.seed,
        )
    }

    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    pub fn with_num_bands(mut self, num_bands: usize) -> Self {
        assert!(num_bands > 0, "num_bands must be positive");
        self.num_bands = num_bands;
        self
    }

    pub fn with_time_horizon(mut self, time_horizon: usize) -> Self {
        assert!(time_horizon > 0, "time_horizon must be positive");
        self.time_horizon = time_horizon;
        self
    }

    pub fn with_receiver(mut self, receiver: Receiver) -> Self {
        self.receiver = receiver;
        self
    }

    pub fn fixed_frequency() -> Self {
        Self {
            name: "fixed-frequency".to_string(),
            description: "One continuously transmitting fixed-frequency emitter.".to_string(),
            num_bands: 32,
            time_horizon: 1_000,
            receiver: Receiver::default(),
            emitters: vec![Emitter::new(0, 1.0e9, EmitterType::FixedFrequency)],
            seed: 42,
        }
    }

    pub fn periodic() -> Self {
        Self {
            name: "periodic".to_string(),
            description: "Short periodic transmissions with a 25-slot period.".to_string(),
            num_bands: 32,
            time_horizon: 1_000,
            receiver: Receiver::default(),
            emitters: vec![
                Emitter::new(0, 0.85e9, EmitterType::Periodic).with_period(25, 0.16),
                Emitter::new(1, 1.15e9, EmitterType::Intermittent).with_period(1, 0.15),
            ],
            seed: 42,
        }
    }

    pub fn frequency_hopping() -> Self {
        Self {
            name: "frequency-hopping".to_string(),
            description: "A hopping emitter spread across the scan spectrum.".to_string(),
            num_bands: 64,
            time_horizon: 1_000,
            receiver: Receiver::new(0.1, 0.9, 0.01),
            emitters: vec![Emitter::new(0, 1.0e9, EmitterType::FrequencyHopping)
                .with_frequency_range(0.65e9, 1.35e9)],
            seed: 42,
        }
    }

    pub fn mixed() -> Self {
        Self {
            name: "mixed".to_string(),
            description: "Fixed, periodic, intermittent, agile, and hopping emitters.".to_string(),
            num_bands: 64,
            time_horizon: 1_000,
            receiver: Receiver::default(),
            emitters: vec![
                Emitter::new(0, 0.7e9, EmitterType::FixedFrequency),
                Emitter::new(1, 0.85e9, EmitterType::Periodic).with_period(20, 0.2),
                Emitter::new(2, 1.0e9, EmitterType::Intermittent).with_period(1, 0.25),
                Emitter::new(3, 1.15e9, EmitterType::FrequencyAgile),
                Emitter::new(4, 1.25e9, EmitterType::FrequencyHopping)
                    .with_frequency_range(1.1e9, 1.4e9),
            ],
            seed: 42,
        }
    }

    /// Long-duration live display scenario. Ground truth is bit-packed, so the
    /// 30 × 10,000,000 state matrix remains bounded at roughly 36 MiB.
    pub fn long_running_mixed() -> Self {
        Self::mixed()
            .with_num_bands(30)
            .with_time_horizon(10_000_000)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scenario_creation() {
        let scenario = Scenario::new("basic".to_string(), "Basic test scenario".to_string());
        assert_eq!(scenario.name, "basic");
        assert_eq!(scenario.num_bands, 32);
    }

    #[test]
    fn periodic_scenario_has_short_duration_emitter() {
        let scenario = Scenario::periodic();
        assert_eq!(scenario.emitters.len(), 2);
        assert_eq!(scenario.emitters[0].emitter_type, EmitterType::Periodic);
    }

    #[test]
    fn scenario_constructs_a_runnable_environment() {
        let mut environment = Scenario::mixed().into_environment();
        let result = environment.step(0);
        assert_eq!(result.observation.len(), 64);
    }

    #[test]
    fn scenario_band_count_can_be_configured() {
        let scenario = Scenario::mixed().with_num_bands(30);
        assert_eq!(scenario.num_bands, 30);
    }

    #[test]
    fn long_running_scenario_uses_the_requested_horizon() {
        assert_eq!(Scenario::long_running_mixed().time_horizon, 10_000_000);
    }
}
