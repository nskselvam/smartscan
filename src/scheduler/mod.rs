use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

pub mod dl_ppo;

pub use dl_ppo::{DlPpoScheduler, HybridDecision, HybridError};

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

pub struct EpsilonGreedyScheduler {
    epsilon: f32,
    estimates: Vec<f32>,
    visits: Vec<usize>,
    last_action: Option<usize>,
    rng: StdRng,
}

impl EpsilonGreedyScheduler {
    pub fn new(num_bands: usize, epsilon: f32, seed: u64) -> Self {
        assert!(num_bands > 0);
        assert!((0.0..=1.0).contains(&epsilon));
        Self {
            epsilon,
            estimates: vec![0.0; num_bands],
            visits: vec![0; num_bands],
            last_action: None,
            rng: StdRng::seed_from_u64(seed),
        }
    }
}

impl Scheduler for EpsilonGreedyScheduler {
    fn select_band(&mut self) -> usize {
        let action = if let Some(unvisited) = self.visits.iter().position(|visits| *visits == 0) {
            unvisited
        } else if self.rng.gen::<f32>() < self.epsilon {
            self.rng.gen_range(0..self.estimates.len())
        } else {
            self.estimates
                .iter()
                .enumerate()
                .max_by(|(_, left), (_, right)| left.total_cmp(right))
                .map(|(band, _)| band)
                .expect("scheduler has at least one band")
        };
        self.last_action = Some(action);
        action
    }

    fn update(&mut self, reward: f32) {
        if let Some(action) = self.last_action {
            self.visits[action] += 1;
            let count = self.visits[action] as f32;
            self.estimates[action] += (reward - self.estimates[action]) / count;
        }
    }

    fn reset(&mut self) {
        self.last_action = None;
    }

    fn name(&self) -> &'static str {
        "epsilon-greedy"
    }
}

pub struct Ucb1Scheduler {
    estimates: Vec<f32>,
    visits: Vec<usize>,
    total_visits: usize,
    last_action: Option<usize>,
}

impl Ucb1Scheduler {
    pub fn new(num_bands: usize) -> Self {
        assert!(num_bands > 0);
        Self {
            estimates: vec![0.0; num_bands],
            visits: vec![0; num_bands],
            total_visits: 0,
            last_action: None,
        }
    }
}

impl Scheduler for Ucb1Scheduler {
    fn select_band(&mut self) -> usize {
        let action = if let Some(unvisited) = self.visits.iter().position(|visits| *visits == 0) {
            unvisited
        } else {
            let total = self.total_visits.max(1) as f32;
            self.estimates
                .iter()
                .enumerate()
                .map(|(band, estimate)| {
                    let bonus = (2.0 * total.ln() / self.visits[band] as f32).sqrt();
                    (band, estimate + bonus)
                })
                .max_by(|(_, left), (_, right)| left.total_cmp(right))
                .map(|(band, _)| band)
                .expect("scheduler has at least one band")
        };
        self.last_action = Some(action);
        action
    }

    fn update(&mut self, reward: f32) {
        if let Some(action) = self.last_action {
            self.visits[action] += 1;
            self.total_visits += 1;
            let count = self.visits[action] as f32;
            self.estimates[action] += (reward - self.estimates[action]) / count;
        }
    }

    fn reset(&mut self) {
        self.last_action = None;
    }

    fn name(&self) -> &'static str {
        "ucb1"
    }
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

    #[test]
    fn epsilon_greedy_visits_each_band_before_exploitation() {
        let mut scheduler = EpsilonGreedyScheduler::new(3, 0.0, 42);
        assert_eq!(scheduler.select_band(), 0);
        scheduler.update(1.0);
        assert_eq!(scheduler.select_band(), 1);
        scheduler.update(2.0);
        assert_eq!(scheduler.select_band(), 2);
    }

    #[test]
    fn ucb1_visits_each_band_before_confidence_selection() {
        let mut scheduler = Ucb1Scheduler::new(2);
        assert_eq!(scheduler.select_band(), 0);
        scheduler.update(1.0);
        assert_eq!(scheduler.select_band(), 1);
        scheduler.update(1.0);
        assert!(scheduler.select_band() < 2);
    }
}
