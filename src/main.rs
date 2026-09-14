use anyhow::{bail, Result};
use smartscan::cli::{Args, Command, DataCommand, TrainCommand};
use smartscan::data::{
    inspect_directory, preprocess_csv_file, preprocess_tsrd_hdf5, write_processed_index,
};
use smartscan::evaluation::{run_baseline_benchmark, run_episode, BenchmarkConfig, Metrics};
use smartscan::gui::launch;
use smartscan::ppo::{PpoAgent, PpoConfig, PpoEnvironment};
use smartscan::scheduler::{RandomScheduler, RoundRobinScheduler};
use smartscan::simulator::Scenario;
use smartscan::storage::PostgresExperimentStore;
use smartscan::{init_logging, Config, PROJECT_DESC, PROJECT_NAME, VERSION};
use tracing::{error, info, warn};

fn main() -> Result<()> {
    let args = Args::parse_args();

    // Initialize logging
    init_logging(&args.log_level)?;

    info!("{} v{}", PROJECT_NAME, VERSION);
    info!("{}", PROJECT_DESC);

    // Load or create config
    let config = match &args.config {
        Some(path) => {
            info!("Loading config from {:?}", path);
            match Config::from_file(path) {
                Ok(c) => c,
                Err(e) => {
                    error!("Failed to load config: {}", e);
                    return Err(e.into());
                }
            }
        }
        None => {
            info!("Using default configuration");
            Config::default()
        }
    };

    // Validate config
    config.validate()?;

    // Dispatch commands
    match args.command {
        Command::Setup { mode, force } => {
            cmd_setup(&config, &mode, force)?;
        }
        Command::Data { subcommand } => {
            cmd_data(&config, subcommand)?;
        }
        Command::Train { subcommand } => {
            cmd_train(&config, subcommand)?;
        }
        Command::Evaluate {
            checkpoint,
            episodes,
        } => {
            cmd_evaluate(&config, checkpoint, episodes)?;
        }
        Command::Benchmark {
            benchmark_type,
            output_format,
        } => {
            cmd_benchmark(&config, &benchmark_type, &output_format)?;
        }
        Command::Demo { steps, visualize } => {
            cmd_demo(&config, steps, visualize)?;
        }
        Command::Gui => {
            cmd_gui(&config)?;
        }
    }

    info!("Finished successfully");
    Ok(())
}

fn cmd_setup(_config: &Config, mode: &str, _force: bool) -> Result<()> {
    info!("Setting up project (mode: {})", mode);
    info!("Creating directories...");

    // This will be implemented in Phase 1
    warn!("Setup not yet fully implemented");

    Ok(())
}

fn cmd_data(config: &Config, subcommand: DataCommand) -> Result<()> {
    match subcommand {
        DataCommand::Inspect { dataset } => {
            let dataset_path = dataset
                .map(|name| {
                    std::path::Path::new(&config.app.data_dir)
                        .join("raw")
                        .join(name)
                })
                .unwrap_or_else(|| std::path::Path::new(&config.app.data_dir).join("raw"));
            let inspection = inspect_directory(&dataset_path)?;
            info!(
                path = %inspection.root.display(),
                file_count = inspection.file_count,
                csv_files = inspection.csv_files,
                total_bytes = inspection.total_bytes,
                "Local dataset inspection"
            );
        }
        DataCommand::Preprocess {
            input,
            output,
            chunk_size,
        } => {
            let input_path = input.unwrap_or_else(|| {
                std::path::Path::new(&config.app.data_dir)
                    .join("raw")
                    .join("pulses.csv")
            });
            if !input_path.is_file() {
                bail!("input CSV does not exist: {}", input_path.display());
            }
            let output_path = output.unwrap_or_else(|| {
                std::path::Path::new(&config.app.data_dir)
                    .join("processed")
                    .join("pulses.ssp")
            });
            let index = match input_path
                .extension()
                .and_then(|extension| extension.to_str())
            {
                Some("h5") | Some("hdf5") => {
                    preprocess_tsrd_hdf5(&input_path, &output_path, chunk_size)?
                }
                _ => preprocess_csv_file(&input_path, &output_path, chunk_size)?,
            };
            let index_path = std::path::Path::new(&config.app.data_dir)
                .join("cache")
                .join("processed_index.json");
            write_processed_index(&index_path, &index)?;
            info!(
                input = %input_path.display(),
                output = %output_path.display(),
                pulse_count = index.pulse_count,
                index = %index_path.display(),
                "Preprocessed pulse dataset"
            );
        }
        DataCommand::Download {
            dataset,
            mode,
            split,
            subset,
        } => {
            info!(
                "Downloading dataset: {} (mode: {}, split: {}, subset: {})",
                dataset, mode, split, subset
            );
            warn!("Data download not yet implemented");
        }
    }
    Ok(())
}

fn cmd_train(config: &Config, subcommand: TrainCommand) -> Result<()> {
    match subcommand {
        TrainCommand::Dl {
            epochs,
            lr,
            batch_size,
            checkpoint,
        } => {
            info!("Training DL model");
            if let Some(e) = epochs {
                info!("Epochs: {}", e);
            }
            if let Some(l) = lr {
                info!("Learning rate: {}", l);
            }
            if let Some(b) = batch_size {
                info!("Batch size: {}", b);
            }
            if let Some(c) = checkpoint {
                info!("Checkpoint: {:?}", c);
            }
            warn!("DL training not yet implemented");
        }
        TrainCommand::Ppo {
            steps,
            lr,
            checkpoint,
        } => {
            if steps == 0 {
                bail!("PPO training steps must be positive");
            }
            let scenario = Scenario::mixed().with_num_bands(30);
            let action_count = scenario.num_bands;
            let mut environment = PpoEnvironment::new(scenario.into_environment());
            let state_size = environment.observation().to_vector().len();
            let mut agent = PpoAgent::new(PpoConfig {
                state_size,
                action_count,
                learning_rate: lr.unwrap_or(config.training.learning_rate),
                gamma: 0.99,
                gae_lambda: 0.95,
                clip_epsilon: 0.2,
                entropy_coefficient: 0.01,
                value_loss_coefficient: 0.5,
                gradient_clip: 1.0,
                update_epochs: 4,
                seed: config.simulator.seed,
            });

            let mut remaining_steps = steps;
            let mut episodes = 0;
            let mut total_reward = 0.0;
            let mut last_report = None;
            while remaining_steps > 0 {
                let episode_steps = remaining_steps.min(config.simulator.time_horizon);
                let report = agent.train_episode(&mut environment, episode_steps)?;
                remaining_steps -= report.rollout_steps;
                total_reward += report.total_reward;
                episodes += 1;
                last_report = Some(report);
            }
            if let Some(report) = last_report {
                info!(
                    episodes,
                    total_reward,
                    policy_objective = report.policy_objective,
                    value_loss = report.value_loss,
                    entropy = report.entropy,
                    "Measured PPO training result"
                );
            }
            let checkpoint_path = checkpoint.unwrap_or_else(|| {
                std::path::Path::new(&config.app.models_dir)
                    .join("ppo")
                    .join("latest.json")
            });
            agent.save(&checkpoint_path)?;
            info!(path = %checkpoint_path.display(), "Saved trained PPO checkpoint");
        }
        TrainCommand::All { quick } => {
            info!("Training all models (quick: {})", quick);
            warn!("Full training not yet implemented");
        }
    }
    Ok(())
}

fn cmd_evaluate(
    _config: &Config,
    checkpoint: Option<std::path::PathBuf>,
    episodes: usize,
) -> Result<()> {
    info!("Evaluating model (episodes: {})", episodes);
    if let Some(c) = checkpoint {
        info!("Checkpoint: {:?}", c);
    }
    warn!("Evaluation not yet implemented");
    Ok(())
}

fn cmd_benchmark(config: &Config, benchmark_type: &str, output_format: &str) -> Result<()> {
    if benchmark_type != "all" {
        bail!("only --benchmark-type all is currently supported");
    }
    if !matches!(output_format, "text" | "json" | "csv") {
        bail!("output format must be text, json, or csv");
    }
    let output_dir = std::path::Path::new(&config.app.results_dir)
        .join("experiments")
        .join(format!("baseline_mixed_seed_{}", config.simulator.seed));
    let report = run_baseline_benchmark(BenchmarkConfig {
        scenario: "mixed".to_string(),
        episodes: config.evaluation.num_test_episodes,
        max_steps: config.simulator.time_horizon,
        seed: config.simulator.seed,
        output_dir: output_dir.clone(),
    })?;
    if std::env::var_os("DATABASE_URL").is_some() {
        let experiment_id =
            PostgresExperimentStore::connect_from_environment()?.save_benchmark(&report)?;
        info!(experiment_id, "Persisted measured benchmark to PostgreSQL");
    }
    for result in report.results {
        info!(
            scheduler = result.scheduler,
            episodes = result.episodes,
            pd = result.metrics.pd,
            pfa = result.metrics.pfa,
            intercept_rate = result.metrics.intercept_rate,
            avg_intercept_time = result.metrics.avg_intercept_time,
            avg_reward = result.metrics.avg_reward,
            "Measured benchmark result"
        );
    }
    info!(path = %output_dir.display(), "Benchmark artifacts written");
    Ok(())
}

fn cmd_demo(_config: &Config, steps: usize, visualize: bool) -> Result<()> {
    info!(
        "Running demo simulation (steps: {}, visualize: {})",
        steps, visualize
    );

    let scenario = Scenario::mixed();
    let num_bands = scenario.num_bands;
    let seed = scenario.seed;
    let mut random_environment = scenario.into_environment();
    let mut random_scheduler = RandomScheduler::new(num_bands, seed.wrapping_add(2));
    let random_result = run_episode(&mut random_environment, &mut random_scheduler, steps);
    let random_metrics = Metrics::from_episodes(std::slice::from_ref(&random_result));

    let scenario = Scenario::mixed();
    let mut round_robin_environment = scenario.into_environment();
    let mut round_robin_scheduler = RoundRobinScheduler::new(num_bands);
    let round_robin_result = run_episode(
        &mut round_robin_environment,
        &mut round_robin_scheduler,
        steps,
    );
    let round_robin_metrics = Metrics::from_episodes(std::slice::from_ref(&round_robin_result));

    info!(
        scheduler = random_result.scheduler,
        steps = random_result.steps,
        total_reward = random_result.total_reward,
        true_positives = random_result.true_positives,
        false_positives = random_result.false_positives,
        missed_active_bands = random_result.missed_active_bands,
        pd = random_metrics.pd,
        pfa = random_metrics.pfa,
        intercept_rate = random_metrics.intercept_rate,
        avg_intercept_time = random_metrics.avg_intercept_time,
        "Measured baseline result"
    );
    info!(
        scheduler = round_robin_result.scheduler,
        steps = round_robin_result.steps,
        total_reward = round_robin_result.total_reward,
        true_positives = round_robin_result.true_positives,
        false_positives = round_robin_result.false_positives,
        missed_active_bands = round_robin_result.missed_active_bands,
        pd = round_robin_metrics.pd,
        pfa = round_robin_metrics.pfa,
        intercept_rate = round_robin_metrics.intercept_rate,
        avg_intercept_time = round_robin_metrics.avg_intercept_time,
        "Measured baseline result"
    );

    if visualize {
        warn!("GUI visualization is scheduled for Phase 14; the measured CLI demo completed");
    }
    Ok(())
}

fn cmd_gui(_config: &Config) -> Result<()> {
    info!("Launching GUI");
    launch().map_err(|error| anyhow::anyhow!(error.to_string()))?;
    Ok(())
}
