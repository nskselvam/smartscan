use std::path::PathBuf;

use thiserror::Error;

use super::{
    evaluate_predictions, GruActivityPredictor, LabeledSequence, ModelError, PredictionMetrics,
};

#[derive(Debug, Clone)]
pub struct DlTrainingConfig {
    pub epochs: usize,
    pub batch_size: usize,
    pub learning_rate: f32,
    pub learning_rate_decay: f32,
    pub early_stopping_patience: usize,
    pub checkpoint_path: PathBuf,
}

#[derive(Debug, Clone)]
pub struct TrainingReport {
    pub epochs_run: usize,
    pub best_epoch: usize,
    pub training_loss: Vec<f32>,
    pub validation_loss: Vec<f32>,
    pub validation_metrics: PredictionMetrics,
}

#[derive(Debug, Error)]
pub enum TrainingError {
    #[error("training and validation datasets must both be non-empty")]
    EmptyDataset,
    #[error("sequence target width does not match model output width")]
    OutputWidthMismatch,
    #[error(transparent)]
    Model(#[from] ModelError),
}

pub fn train_model(
    model: &mut GruActivityPredictor,
    training: &[LabeledSequence],
    validation: &[LabeledSequence],
    config: &DlTrainingConfig,
) -> Result<TrainingReport, TrainingError> {
    if training.is_empty() || validation.is_empty() {
        return Err(TrainingError::EmptyDataset);
    }
    if training
        .iter()
        .chain(validation)
        .any(|sample| sample.target_activity.len() != model.config().output_size)
    {
        return Err(TrainingError::OutputWidthMismatch);
    }
    assert!(config.epochs > 0, "epochs must be positive");
    assert!(config.batch_size > 0, "batch_size must be positive");
    assert!(config.learning_rate > 0.0, "learning_rate must be positive");
    assert!((0.0..=1.0).contains(&config.learning_rate_decay));

    let mut best_model = model.clone();
    let mut best_metrics = evaluate(model, validation);
    let mut best_epoch = 0;
    let mut patience = 0;
    let mut learning_rate = config.learning_rate;
    let mut training_loss = Vec::new();
    let mut validation_loss = Vec::new();

    for epoch in 0..config.epochs {
        let mut loss_sum = 0.0;
        let mut samples = 0;
        for batch in training.chunks(config.batch_size) {
            for sample in batch {
                loss_sum += model.train_output_head(
                    &sample.history,
                    &sample.target_activity,
                    learning_rate,
                );
                samples += 1;
            }
        }
        let train_loss = loss_sum / samples.max(1) as f32;
        let metrics = evaluate(model, validation);
        training_loss.push(train_loss);
        validation_loss.push(metrics.bce_loss);

        if metrics.bce_loss < best_metrics.bce_loss {
            best_model = model.clone();
            best_metrics = metrics;
            best_epoch = epoch + 1;
            patience = 0;
        } else {
            patience += 1;
            learning_rate *= config.learning_rate_decay;
            if patience >= config.early_stopping_patience {
                break;
            }
        }
    }
    best_model.save(&config.checkpoint_path)?;
    *model = best_model;

    Ok(TrainingReport {
        epochs_run: training_loss.len(),
        best_epoch,
        training_loss,
        validation_loss,
        validation_metrics: best_metrics,
    })
}

fn evaluate(model: &GruActivityPredictor, validation: &[LabeledSequence]) -> PredictionMetrics {
    let predictions = validation
        .iter()
        .map(|sample| model.predict(&sample.history))
        .collect::<Vec<_>>();
    let targets = validation
        .iter()
        .map(|sample| sample.target_activity.clone())
        .collect::<Vec<_>>();
    evaluate_predictions(&predictions, &targets)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::features::FEATURE_COUNT;
    use crate::data::FeatureVector;
    use crate::dl::GruConfig;

    fn sample(target: f32) -> LabeledSequence {
        LabeledSequence::new(
            vec![FeatureVector {
                values: [0.5; FEATURE_COUNT],
            }],
            vec![target],
        )
    }

    #[test]
    fn training_saves_best_checkpoint_and_reports_validation_metrics() {
        let path =
            std::env::temp_dir().join(format!("smartscan-training-{}.json", std::process::id()));
        let mut model = GruActivityPredictor::new(GruConfig::new(10, 4, 1, 1, 1, 0.0), 42);
        let config = DlTrainingConfig {
            epochs: 10,
            batch_size: 2,
            learning_rate: 0.1,
            learning_rate_decay: 0.5,
            early_stopping_patience: 3,
            checkpoint_path: path.clone(),
        };
        let report = train_model(
            &mut model,
            &[sample(1.0), sample(1.0)],
            &[sample(1.0)],
            &config,
        )
        .expect("training should succeed");

        assert!(path.is_file());
        assert!(report.epochs_run > 0);
        assert!(report.validation_metrics.bce_loss.is_finite());
        std::fs::remove_file(path).expect("checkpoint should be removed");
    }
}
