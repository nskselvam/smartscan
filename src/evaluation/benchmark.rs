use std::fs;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::evaluation::{run_episode, EpisodeSummary, Metrics};
use crate::scheduler::{
    EpsilonGreedyScheduler, RandomScheduler, RoundRobinScheduler, Scheduler, Ucb1Scheduler,
};
use crate::simulator::Scenario;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkConfig {
    pub scenario: String,
    pub episodes: usize,
    pub max_steps: usize,
    pub seed: u64,
    pub output_dir: PathBuf,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchedulerBenchmark {
    pub scheduler: String,
    pub episodes: usize,
    pub metrics: Metrics,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkReport {
    pub config: BenchmarkConfig,
    pub results: Vec<SchedulerBenchmark>,
}

#[derive(Debug, Error)]
pub enum BenchmarkError {
    #[error("episodes and max_steps must both be positive")]
    InvalidConfig,
    #[error("unsupported scenario: {0}")]
    UnsupportedScenario(String),
    #[error("artifact I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("artifact serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
    #[error("CSV artifact error: {0}")]
    Csv(#[from] csv::Error),
}

pub fn run_baseline_benchmark(config: BenchmarkConfig) -> Result<BenchmarkReport, BenchmarkError> {
    if config.episodes == 0 || config.max_steps == 0 {
        return Err(BenchmarkError::InvalidConfig);
    }
    let template = scenario_by_name(&config.scenario)?;
    let num_bands = template.num_bands;
    let definitions: [(&str, Box<dyn Scheduler>); 4] = [
        (
            "random",
            Box::new(RandomScheduler::new(
                num_bands,
                config.seed.wrapping_add(10),
            )),
        ),
        ("round-robin", Box::new(RoundRobinScheduler::new(num_bands))),
        (
            "epsilon-greedy",
            Box::new(EpsilonGreedyScheduler::new(
                num_bands,
                0.1,
                config.seed.wrapping_add(11),
            )),
        ),
        ("ucb1", Box::new(Ucb1Scheduler::new(num_bands))),
    ];

    let mut results = Vec::with_capacity(definitions.len());
    for (name, mut scheduler) in definitions {
        let summaries = (0..config.episodes)
            .map(|episode| {
                let mut environment = template
                    .clone()
                    .with_seed(config.seed.wrapping_add(episode as u64))
                    .into_environment();
                run_episode(&mut environment, scheduler.as_mut(), config.max_steps)
            })
            .collect::<Vec<EpisodeSummary>>();
        results.push(SchedulerBenchmark {
            scheduler: name.to_string(),
            episodes: summaries.len(),
            metrics: Metrics::from_episodes(&summaries),
        });
    }
    let report = BenchmarkReport { config, results };
    write_artifacts(&report)?;
    Ok(report)
}

fn scenario_by_name(name: &str) -> Result<Scenario, BenchmarkError> {
    match name {
        "fixed-frequency" => Ok(Scenario::fixed_frequency()),
        "periodic" => Ok(Scenario::periodic()),
        "frequency-hopping" => Ok(Scenario::frequency_hopping()),
        "mixed" => Ok(Scenario::mixed()),
        _ => Err(BenchmarkError::UnsupportedScenario(name.to_string())),
    }
}

fn write_artifacts(report: &BenchmarkReport) -> Result<(), BenchmarkError> {
    fs::create_dir_all(&report.config.output_dir)?;
    fs::write(
        report.config.output_dir.join("config.json"),
        serde_json::to_vec_pretty(&report.config)?,
    )?;
    fs::write(
        report.config.output_dir.join("metrics.json"),
        serde_json::to_vec_pretty(&report.results)?,
    )?;
    write_csv(
        &report.config.output_dir.join("metrics.csv"),
        &report.results,
    )?;
    Ok(())
}

fn write_csv(path: &Path, results: &[SchedulerBenchmark]) -> Result<(), BenchmarkError> {
    let mut writer = csv::Writer::from_path(path)?;
    writer.write_record([
        "scheduler",
        "episodes",
        "pd",
        "pfa",
        "intercept_rate",
        "avg_intercept_time",
        "avg_reward",
        "miss_rate",
    ])?;
    for result in results {
        writer.write_record([
            result.scheduler.clone(),
            result.episodes.to_string(),
            result.metrics.pd.to_string(),
            result.metrics.pfa.to_string(),
            result.metrics.intercept_rate.to_string(),
            result.metrics.avg_intercept_time.to_string(),
            result.metrics.avg_reward.to_string(),
            result.metrics.miss_rate.to_string(),
        ])?;
    }
    writer.flush()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn benchmark_writes_measured_artifacts_for_all_baselines() {
        let output_dir =
            std::env::temp_dir().join(format!("smartscan-benchmark-{}", std::process::id()));
        let report = run_baseline_benchmark(BenchmarkConfig {
            scenario: "periodic".to_string(),
            episodes: 2,
            max_steps: 20,
            seed: 42,
            output_dir: output_dir.clone(),
        })
        .expect("benchmark should run");
        assert_eq!(report.results.len(), 4);
        assert!(output_dir.join("config.json").is_file());
        assert!(output_dir.join("metrics.json").is_file());
        assert!(output_dir.join("metrics.csv").is_file());
        fs::remove_dir_all(output_dir).expect("artifacts should be removed");
    }
}
