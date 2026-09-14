/// Emitter Models
///
/// Supports various emitter types:
/// - Fixed-frequency emitters
/// - Periodic emitters
/// - Intermittent emitters
/// - Frequency-agile emitters
/// - Frequency-hopping emitters

use rand::Rng;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EmitterType {
    FixedFrequency,
    Periodic,
    Intermittent,
    FrequencyAgile,
    FrequencyHopping,
}

#[derive(Debug, Clone)]
pub struct Emitter {
    pub id: usize,
    pub center_frequency: f32,
    pub emitter_type: EmitterType,
    pub period: usize,
    pub duty_cycle: f32,
    pub frequency_range: (f32, f32),
}

impl Emitter {
    pub fn new(id: usize, center_frequency: f32, emitter_type: EmitterType) -> Self {
        Emitter {
            id,
            center_frequency,
            emitter_type,
            period: 1,
            duty_cycle: 1.0,
            frequency_range: (0.5e9, 1.5e9),
        }
    }

    pub fn with_period(mut self, period: usize, duty_cycle: f32) -> Self {
        self.period = period.max(1);
        self.duty_cycle = duty_cycle.clamp(0.0, 1.0);
        self
    }

    pub fn with_frequency_range(mut self, minimum: f32, maximum: f32) -> Self {
        assert!(minimum < maximum, "frequency range must be increasing");
        self.frequency_range = (minimum, maximum);
        self
    }

    pub fn is_transmitting(&self, time: usize, rng: &mut impl Rng) -> bool {
        match self.emitter_type {
            EmitterType::FixedFrequency | EmitterType::FrequencyAgile | EmitterType::FrequencyHopping => true,
            EmitterType::Periodic => {
                let active_slots = (self.period as f32 * self.duty_cycle).ceil() as usize;
                time % self.period < active_slots
            }
            EmitterType::Intermittent => rng.gen::<f32>() < self.duty_cycle,
        }
    }

    pub fn frequency_at(&self, time: usize, rng: &mut impl Rng) -> f32 {
        match self.emitter_type {
            EmitterType::FrequencyAgile => {
                let span = (self.frequency_range.1 - self.frequency_range.0) * 0.25;
                self.center_frequency + span * (time as f32 / 20.0).sin()
            }
            EmitterType::FrequencyHopping => {
                rng.gen_range(self.frequency_range.0..self.frequency_range.1)
            }
            _ => self.center_frequency,
        }
    }

    pub fn band_at(&self, time: usize, num_bands: usize, rng: &mut impl Rng) -> usize {
        assert!(num_bands > 0, "num_bands must be positive");
        let frequency = self.frequency_at(time, rng);
        let normalized = ((frequency - self.frequency_range.0)
            / (self.frequency_range.1 - self.frequency_range.0))
            .clamp(0.0, 1.0);
        ((normalized * num_bands as f32).floor() as usize).min(num_bands - 1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    #[test]
    fn test_emitter_creation() {
        let emitter = Emitter::new(0, 1.0e9, EmitterType::FixedFrequency);
        assert_eq!(emitter.id, 0);
        assert_eq!(emitter.emitter_type, EmitterType::FixedFrequency);
    }

    #[test]
    fn periodic_emitter_respects_its_duty_cycle() {
        let emitter = Emitter::new(0, 1.0e9, EmitterType::Periodic).with_period(10, 0.3);
        let mut rng = rand::rngs::StdRng::seed_from_u64(7);
        let active = (0..10)
            .filter(|time| emitter.is_transmitting(*time, &mut rng))
            .count();
        assert_eq!(active, 3);
    }

    #[test]
    fn hopping_emitter_stays_within_its_band_range() {
        let emitter = Emitter::new(0, 1.0e9, EmitterType::FrequencyHopping)
            .with_frequency_range(0.8e9, 1.2e9);
        let mut rng = rand::rngs::StdRng::seed_from_u64(7);
        for time in 0..32 {
            assert!(emitter.band_at(time, 32, &mut rng) < 32);
        }
    }

    #[test]
    fn intermittent_activity_is_seed_reproducible() {
        let emitter = Emitter::new(0, 1.0e9, EmitterType::Intermittent).with_period(1, 0.4);
        let mut first_rng = rand::rngs::StdRng::seed_from_u64(3);
        let mut second_rng = rand::rngs::StdRng::seed_from_u64(3);
        let first: Vec<_> = (0..20)
            .map(|time| emitter.is_transmitting(time, &mut first_rng))
            .collect();
        let second: Vec<_> = (0..20)
            .map(|time| emitter.is_transmitting(time, &mut second_rng))
            .collect();
        assert_eq!(first, second);
    }
}
