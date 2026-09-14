/// Evaluation Metrics and Benchmarking
///
/// Calculates performance metrics:
/// - Probability of Detection (Pd)
/// - Probability of False Alarm (Pfa)
/// - Intercept Rate
/// - Intercept Time
/// - Reward metrics
pub struct Metrics {
    pub pd: f32,  // Probability of Detection
    pub pfa: f32, // Probability of False Alarm
    pub intercept_rate: f32,
    pub avg_intercept_time: f32,
    pub avg_reward: f32,
}

impl Default for Metrics {
    fn default() -> Self {
        Metrics {
            pd: 0.0,
            pfa: 0.0,
            intercept_rate: 0.0,
            avg_intercept_time: 0.0,
            avg_reward: 0.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metrics_creation() {
        let metrics = Metrics::default();
        assert_eq!(metrics.pd, 0.0);
    }
}
