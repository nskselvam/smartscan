use super::estimator::PeriodicityEstimate;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PredictedWindow {
    pub start: usize,
    pub end_exclusive: usize,
}

pub struct PeriodicPredictor {
    window_duration: usize,
    minimum_confidence: f32,
}

impl PeriodicPredictor {
    pub fn new(window_duration: usize, minimum_confidence: f32) -> Self {
        assert!(window_duration > 0);
        assert!((0.0..=1.0).contains(&minimum_confidence));
        Self {
            window_duration,
            minimum_confidence,
        }
    }

    pub fn predict_next(
        &self,
        estimate: PeriodicityEstimate,
        current_time: usize,
    ) -> Option<PredictedWindow> {
        if estimate.confidence < self.minimum_confidence {
            return None;
        }
        let start = if current_time < estimate.phase {
            estimate.phase
        } else {
            estimate.phase
                + ((current_time - estimate.phase) / estimate.period + 1) * estimate.period
        };
        Some(PredictedWindow {
            start,
            end_exclusive: start + self.window_duration,
        })
    }
}

impl Default for PeriodicPredictor {
    fn default() -> Self {
        Self::new(1, 0.4)
    }
}

/// Reserves a deterministic fraction of scan decisions for exploration.
#[derive(Debug, Clone, Copy)]
pub struct ExplorationReserve {
    fraction: f32,
}

impl ExplorationReserve {
    pub fn new(fraction: f32) -> Self {
        assert!((0.0..=1.0).contains(&fraction));
        Self { fraction }
    }

    pub fn uses_exploration(&self, decision_index: usize) -> bool {
        if self.fraction == 0.0 {
            return false;
        }
        let interval = (1.0 / self.fraction).round().max(1.0) as usize;
        decision_index.is_multiple_of(interval)
    }

    pub fn select_band(
        &self,
        decision_index: usize,
        exploit_band: usize,
        explore_band: usize,
    ) -> usize {
        if self.uses_exploration(decision_index) {
            explore_band
        } else {
            exploit_band
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn predictor_returns_the_next_periodic_window() {
        let predictor = PeriodicPredictor::default();
        let estimate = PeriodicityEstimate {
            period: 10,
            phase: 2,
            confidence: 0.8,
            mean_interval_error: 0.0,
            samples: 5,
        };
        assert_eq!(
            predictor.predict_next(estimate, 13),
            Some(PredictedWindow {
                start: 22,
                end_exclusive: 23
            })
        );
    }

    #[test]
    fn exploration_reserve_preserves_regular_exploration() {
        let reserve = ExplorationReserve::new(0.2);
        assert_eq!(reserve.select_band(0, 1, 7), 7);
        assert_eq!(reserve.select_band(1, 1, 7), 1);
        assert_eq!(reserve.select_band(5, 1, 7), 7);
    }
}
