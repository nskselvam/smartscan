use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

/// Chooses a frequency band for the next receiver scan.
pub trait Scheduler {
    fn select_band(&mut self) -> usize;
    fn update(&mut self, reward: f32);
    fn reset(&mut self);
    fn name(&self) -> &'static str;
}

pub struct RandomScheduler {
    num_bands: usize,
    rng: StdRng,
}

impl RandomScheduler {
    pub fn new(num_bands: usize, seed: u64) -> Self {
        assert!(num_bands > 0, "num_bands must be positive");
        Self {
            num_bands,
            rng: StdRng::seed_from_u64(seed),
        }
    }
}

impl Scheduler for RandomScheduler {
    fn select_band(&mut self) -> usize {
        self.rng.gen_range(0..self.num_bands)
    }

    fn update(&mut self, _reward: f32) {}

    fn reset(&mut self) {}

    fn name(&self) -> &'static str {
        "random"
    }
}

pub struct RoundRobinScheduler {
    num_bands: usize,
    next_band: usize,
}

impl RoundRobinScheduler {
    pub fn new(num_bands: usize) -> Self {
        assert!(num_bands > 0, "num_bands must be positive");
        Self {
            num_bands,
            next_band: 0,
        }
    }
}

impl Scheduler for RoundRobinScheduler {
    fn select_band(&mut self) -> usize {
        let selected_band = self.next_band;
        self.next_band = (self.next_band + 1) % self.num_bands;
        selected_band
    }

    fn update(&mut self, _reward: f32) {}

    fn reset(&mut self) {
        self.next_band = 0;
    }

    fn name(&self) -> &'static str {
        "round-robin"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn random_scheduler_stays_within_the_action_space() {
        let mut scheduler = RandomScheduler::new(4, 42);
        for _ in 0..100 {
            assert!(scheduler.select_band() < 4);
        }
    }

    #[test]
    fn round_robin_scans_every_band_in_order() {
        let mut scheduler = RoundRobinScheduler::new(3);
        assert_eq!(scheduler.select_band(), 0);
        assert_eq!(scheduler.select_band(), 1);
        assert_eq!(scheduler.select_band(), 2);
        assert_eq!(scheduler.select_band(), 0);
        scheduler.reset();
        assert_eq!(scheduler.select_band(), 0);
    }
}
