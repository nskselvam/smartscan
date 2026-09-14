/// Ground Truth
///
/// Stores and manages ground-truth transmission data for evaluation.
/// Each cell is (time, frequency_band) with transmission state.
pub struct GroundTruth {
    pub num_bands: usize,
    pub time_horizon: usize,
    pub data: Vec<Vec<bool>>, // [time][band]
}

impl GroundTruth {
    pub fn new(num_bands: usize, time_horizon: usize) -> Self {
        GroundTruth {
            num_bands,
            time_horizon,
            data: vec![vec![false; num_bands]; time_horizon],
        }
    }

    pub fn is_transmitting(&self, time: usize, band: usize) -> bool {
        if time >= self.time_horizon || band >= self.num_bands {
            false
        } else {
            self.data[time][band]
        }
    }

    pub fn set_transmission(&mut self, time: usize, band: usize, transmitting: bool) {
        if time < self.time_horizon && band < self.num_bands {
            self.data[time][band] = transmitting;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ground_truth_creation() {
        let gt = GroundTruth::new(32, 1000);
        assert_eq!(gt.num_bands, 32);
        assert_eq!(gt.time_horizon, 1000);
    }

    #[test]
    fn test_ground_truth_transmission() {
        let mut gt = GroundTruth::new(32, 1000);
        gt.set_transmission(100, 5, true);
        assert!(gt.is_transmitting(100, 5));
        assert!(!gt.is_transmitting(100, 6));
    }
}
