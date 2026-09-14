#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PeriodicityEstimate {
    pub period: usize,
    pub phase: usize,
    pub confidence: f32,
    pub mean_interval_error: f32,
    pub samples: usize,
}

#[derive(Debug, Clone)]
pub struct PeriodicityEstimator {
    minimum_arrivals: usize,
}

impl PeriodicityEstimator {
    pub fn new(minimum_arrivals: usize) -> Self {
        assert!(minimum_arrivals >= 2, "at least two arrivals are required");
        Self { minimum_arrivals }
    }

    pub fn estimate(&self, arrivals: &[usize]) -> Option<PeriodicityEstimate> {
        if arrivals.len() < self.minimum_arrivals
            || !arrivals.windows(2).all(|pair| pair[0] < pair[1])
        {
            return None;
        }
        let mut intervals = arrivals
            .windows(2)
            .map(|pair| pair[1] - pair[0])
            .collect::<Vec<_>>();
        intervals.sort_unstable();
        let period = intervals[intervals.len() / 2];
        if period == 0 {
            return None;
        }
        let mean_interval_error = intervals
            .iter()
            .map(|interval| interval.abs_diff(period) as f32)
            .sum::<f32>()
            / intervals.len() as f32;
        let phase = most_common_phase(arrivals, period);
        let regularity = (1.0 - mean_interval_error / period as f32).clamp(0.0, 1.0);
        let sample_confidence = (arrivals.len() as f32 / 10.0).min(1.0);
        Some(PeriodicityEstimate {
            period,
            phase,
            confidence: regularity * sample_confidence,
            mean_interval_error,
            samples: arrivals.len(),
        })
    }
}

impl Default for PeriodicityEstimator {
    fn default() -> Self {
        Self::new(3)
    }
}

pub struct PeriodicTracker {
    arrivals: Vec<Vec<usize>>,
    estimator: PeriodicityEstimator,
}

impl PeriodicTracker {
    pub fn new(num_bands: usize, estimator: PeriodicityEstimator) -> Self {
        Self {
            arrivals: vec![Vec::new(); num_bands],
            estimator,
        }
    }

    pub fn record_detection(&mut self, band: usize, time: usize) {
        let arrivals = &mut self.arrivals[band];
        if arrivals.last().copied() != Some(time) {
            arrivals.push(time);
        }
    }

    pub fn estimate_band(&self, band: usize) -> Option<PeriodicityEstimate> {
        self.arrivals
            .get(band)
            .and_then(|arrivals| self.estimator.estimate(arrivals))
    }
}

fn most_common_phase(arrivals: &[usize], period: usize) -> usize {
    let mut counts = vec![0_usize; period];
    for arrival in arrivals {
        counts[arrival % period] += 1;
    }
    counts
        .iter()
        .enumerate()
        .max_by_key(|(_, count)| *count)
        .map(|(phase, _)| phase)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn estimator_recovers_regular_period_and_phase() {
        let estimate = PeriodicityEstimator::default()
            .estimate(&[2, 12, 22, 32, 42])
            .expect("regular arrivals should yield an estimate");
        assert_eq!(estimate.period, 10);
        assert_eq!(estimate.phase, 2);
        assert_eq!(estimate.confidence, 0.5);
    }

    #[test]
    fn tracker_only_uses_recorded_detections() {
        let mut tracker = PeriodicTracker::new(2, PeriodicityEstimator::default());
        for time in [1, 11, 21] {
            tracker.record_detection(1, time);
        }
        assert!(tracker.estimate_band(0).is_none());
        assert_eq!(
            tracker
                .estimate_band(1)
                .expect("estimate should exist")
                .period,
            10
        );
    }
}
