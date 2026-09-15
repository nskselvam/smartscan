/// Ground Truth
///
/// Stores ground truth as a compact bit-packed time × frequency grid.
/// Each cell is (time, frequency_band) with transmission state.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GroundTruth {
    pub num_bands: usize,
    pub time_horizon: usize,
    bits: Vec<u64>,
}

impl GroundTruth {
    pub fn new(num_bands: usize, time_horizon: usize) -> Self {
        let cell_count = num_bands
            .checked_mul(time_horizon)
            .expect("ground truth dimensions overflow");
        GroundTruth {
            num_bands,
            time_horizon,
            bits: vec![0; cell_count.div_ceil(64)],
        }
    }

    pub fn is_transmitting(&self, time: usize, band: usize) -> bool {
        if time >= self.time_horizon || band >= self.num_bands {
            false
        } else {
            let (word, bit) = self.location(time, band);
            self.bits[word] & (1_u64 << bit) != 0
        }
    }

    pub fn set_transmission(&mut self, time: usize, band: usize, transmitting: bool) {
        if time < self.time_horizon && band < self.num_bands {
            let (word, bit) = self.location(time, band);
            let mask = 1_u64 << bit;
            if transmitting {
                self.bits[word] |= mask;
            } else {
                self.bits[word] &= !mask;
            }
        }
    }

    fn location(&self, time: usize, band: usize) -> (usize, usize) {
        let index = time * self.num_bands + band;
        (index / 64, index % 64)
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

    #[test]
    fn bit_packing_scales_to_long_horizons() {
        let gt = GroundTruth::new(30, 10_000_000);
        assert_eq!(gt.bits.len(), 4_687_500);
    }
}
