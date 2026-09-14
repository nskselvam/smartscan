use thiserror::Error;

use crate::data::FeatureVector;
use crate::dl::GruActivityPredictor;
use crate::ppo::{PpoAgent, PpoEnvironment, PpoError};

#[derive(Debug, Clone, PartialEq)]
pub struct HybridDecision {
    pub selected_band: usize,
    pub activity_predictions: Vec<f32>,
    pub action_probabilities: Vec<f32>,
}

#[derive(Debug, Error)]
pub enum HybridError {
    #[error("GRU output width ({predictions}) does not match PPO action count ({actions})")]
    BandCountMismatch { predictions: usize, actions: usize },
    #[error(transparent)]
    Ppo(#[from] PpoError),
}

/// Hybrid inference scheduler. The GRU predicts future activity; PPO selects
/// the scan action using that prediction plus receiver-observable history.
pub struct DlPpoScheduler {
    predictor: GruActivityPredictor,
    ppo_agent: PpoAgent,
}

impl DlPpoScheduler {
    pub fn new(predictor: GruActivityPredictor, ppo_agent: PpoAgent) -> Self {
        Self {
            predictor,
            ppo_agent,
        }
    }

    pub fn decide(
        &self,
        environment: &mut PpoEnvironment,
        feature_history: &[FeatureVector],
    ) -> Result<HybridDecision, HybridError> {
        let activity_predictions = self.predictor.predict(feature_history);
        let action_count = environment.num_actions();
        if activity_predictions.len() != action_count {
            return Err(HybridError::BandCountMismatch {
                predictions: activity_predictions.len(),
                actions: action_count,
            });
        }
        environment.set_predicted_activity(activity_predictions.clone())?;
        let action_probabilities = self
            .ppo_agent
            .action_probabilities(&environment.observation().to_vector());
        let selected_band = action_probabilities
            .iter()
            .enumerate()
            .max_by(|(_, left), (_, right)| left.total_cmp(right))
            .map(|(band, _)| band)
            .expect("PPO action space is guaranteed non-empty");

        Ok(HybridDecision {
            selected_band,
            activity_predictions,
            action_probabilities,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::features::FEATURE_COUNT;
    use crate::dl::GruConfig;
    use crate::ppo::PpoConfig;
    use crate::simulator::Scenario;

    fn config(bands: usize) -> PpoConfig {
        PpoConfig {
            state_size: 3 + 5 * bands,
            action_count: bands,
            learning_rate: 0.01,
            gamma: 0.99,
            gae_lambda: 0.95,
            clip_epsilon: 0.2,
            entropy_coefficient: 0.01,
            value_loss_coefficient: 0.5,
            gradient_clip: 1.0,
            update_epochs: 1,
            seed: 7,
        }
    }

    #[test]
    fn hybrid_scheduler_passes_gru_predictions_to_ppo() {
        let scenario = Scenario::periodic();
        let bands = scenario.num_bands;
        let mut environment = PpoEnvironment::new(scenario.into_environment());
        let predictor = GruActivityPredictor::new(GruConfig::new(10, 8, 1, bands, 2, 0.0), 7);
        let agent = PpoAgent::new(config(bands));
        let scheduler = DlPpoScheduler::new(predictor, agent);
        let feature_history = vec![FeatureVector {
            values: [0.0; FEATURE_COUNT],
        }];

        let decision = scheduler
            .decide(&mut environment, &feature_history)
            .expect("compatible hybrid decision should succeed");

        assert_eq!(decision.activity_predictions.len(), bands);
        assert_eq!(decision.action_probabilities.len(), bands);
        assert_eq!(
            environment.observation().predicted_activity,
            decision.activity_predictions
        );
        assert!(decision.selected_band < bands);
    }
}
