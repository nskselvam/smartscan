/// Receiver Model
///
/// Represents the RF receiver with:
/// - Limited instantaneous bandwidth
/// - Scanning constraints
/// - Detection/false-alarm probabilities
/// - Noisy observations
use rand::Rng;

#[derive(Debug, Clone)]
pub struct Receiver {
    pub bandwidth_fraction: f32,
    pub detection_probability: f32,
    pub false_alarm_probability: f32,
}

impl Receiver {
    pub fn new(
        bandwidth_fraction: f32,
        detection_probability: f32,
        false_alarm_probability: f32,
    ) -> Self {
        Self {
            bandwidth_fraction: bandwidth_fraction.clamp(0.01, 1.0),
            detection_probability: detection_probability.clamp(0.0, 1.0),
            false_alarm_probability: false_alarm_probability.clamp(0.0, 1.0),
        }
    }

    pub fn monitored_bands(&self, num_bands: usize, selected_band: usize) -> Vec<bool> {
        assert!(num_bands > 0, "num_bands must be positive");
        assert!(selected_band < num_bands, "selected band is out of range");

        let width = (num_bands as f32 * self.bandwidth_fraction).ceil().max(1.0) as usize;
        let start = selected_band.saturating_sub(width / 2);
        let end = (start + width).min(num_bands);
        let mut monitored = vec![false; num_bands];
        monitored[start..end].fill(true);
        monitored
    }

    /// Returns `(reported_detection, false_alarm)` for a monitored band.
    pub fn observe(&self, transmitting: bool, rng: &mut impl Rng) -> (bool, bool) {
        if transmitting {
            (rng.gen::<f32>() < self.detection_probability, false)
        } else {
            let false_alarm = rng.gen::<f32>() < self.false_alarm_probability;
            (false_alarm, false_alarm)
        }
    }
}

impl Default for Receiver {
    fn default() -> Self {
        Self::new(0.1, 0.95, 0.01)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    #[test]
    fn test_receiver_creation() {
        let receiver = Receiver::new(0.1, 0.95, 0.01);
        assert_eq!(receiver.bandwidth_fraction, 0.1);
        assert_eq!(receiver.detection_probability, 0.95);
    }

    #[test]
    fn monitored_band_count_matches_receiver_bandwidth() {
        let receiver = Receiver::new(0.25, 1.0, 0.0);
        let monitored = receiver.monitored_bands(32, 16);
        assert_eq!(
            monitored.into_iter().filter(|monitored| *monitored).count(),
            8
        );
    }

    #[test]
    fn perfect_receiver_has_no_false_negative_or_false_alarm() {
        let receiver = Receiver::new(0.1, 1.0, 0.0);
        let mut rng = rand::rngs::StdRng::seed_from_u64(1);
        assert_eq!(receiver.observe(true, &mut rng), (true, false));
        assert_eq!(receiver.observe(false, &mut rng), (false, false));
    }
}
