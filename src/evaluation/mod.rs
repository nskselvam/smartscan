pub mod benchmark;
pub mod runner;

pub use benchmark::{
    run_baseline_benchmark, BenchmarkConfig, BenchmarkError, BenchmarkReport, SchedulerBenchmark,
};
pub use runner::{run_episode, EpisodeSummary};

/// Measured evaluation metrics derived from episode outcomes.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Metrics {
    pub pd: f32,
    pub pfa: f32,
    pub intercept_rate: f32,
    pub avg_intercept_time: f32,
    pub avg_reward: f32,
    pub miss_rate: f32,
    pub total_true_positives: usize,
    pub total_false_positives: usize,
    pub total_active_bands: usize,
    pub total_monitored_inactive_bands: usize,
}

impl Default for Metrics {
    fn default() -> Self {
        Metrics {
            pd: 0.0,
            pfa: 0.0,
            intercept_rate: 0.0,
            avg_intercept_time: 0.0,
            avg_reward: 0.0,
            miss_rate: 0.0,
            total_true_positives: 0,
            total_false_positives: 0,
            total_active_bands: 0,
            total_monitored_inactive_bands: 0,
        }
    }
}

impl Metrics {
    pub fn from_episodes(episodes: &[EpisodeSummary]) -> Self {
        if episodes.is_empty() {
            return Self::default();
        }

        let total_steps: usize = episodes.iter().map(|episode| episode.steps).sum();
        let total_reward: f32 = episodes.iter().map(|episode| episode.total_reward).sum();
        let total_active_bands: usize = episodes.iter().map(|episode| episode.active_bands).sum();
        let total_true_positives: usize =
            episodes.iter().map(|episode| episode.true_positives).sum();
        let total_false_positives: usize =
            episodes.iter().map(|episode| episode.false_positives).sum();
        let total_monitored_inactive_bands: usize = episodes
            .iter()
            .map(|episode| episode.monitored_inactive_bands)
            .sum();
        let transmission_starts: usize = episodes
            .iter()
            .map(|episode| episode.transmission_starts)
            .sum();
        let interceptions: usize = episodes.iter().map(|episode| episode.interceptions).sum();
        let intercept_delay_sum: usize = episodes
            .iter()
            .map(|episode| episode.intercept_delay_sum)
            .sum();

        let pd = ratio(total_true_positives, total_active_bands);
        Self {
            pd,
            pfa: ratio(total_false_positives, total_monitored_inactive_bands),
            intercept_rate: ratio(interceptions, transmission_starts),
            avg_intercept_time: ratio(intercept_delay_sum, interceptions),
            avg_reward: total_reward / total_steps.max(1) as f32,
            miss_rate: 1.0 - pd,
            total_true_positives,
            total_false_positives,
            total_active_bands,
            total_monitored_inactive_bands,
        }
    }
}

fn ratio(numerator: usize, denominator: usize) -> f32 {
    if denominator == 0 {
        0.0
    } else {
        numerator as f32 / denominator as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn metrics_are_calculated_from_recorded_counts() {
        let episode = EpisodeSummary {
            scheduler: "test",
            steps: 10,
            total_reward: 5.0,
            active_bands: 8,
            monitored_bands: 20,
            monitored_active_bands: 5,
            monitored_inactive_bands: 15,
            transmission_starts: 4,
            true_positives: 6,
            false_positives: 3,
            missed_active_bands: 2,
            interceptions: 3,
            intercept_delay_sum: 6,
        };
        let metrics = Metrics::from_episodes(&[episode]);
        assert_eq!(metrics.pd, 0.75);
        assert_eq!(metrics.pfa, 0.2);
        assert_eq!(metrics.miss_rate, 0.25);
        assert_eq!(metrics.avg_reward, 0.5);
        assert_eq!(metrics.intercept_rate, 0.75);
        assert_eq!(metrics.avg_intercept_time, 2.0);
    }
}
