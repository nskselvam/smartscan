use crate::data::FeatureVector;

/// One chronological training example. `target_activity` belongs strictly after
/// `history`, as produced by the temporal-window generator.
#[derive(Debug, Clone, PartialEq)]
pub struct LabeledSequence {
    pub history: Vec<FeatureVector>,
    pub target_activity: Vec<f32>,
}

impl LabeledSequence {
    pub fn new(history: Vec<FeatureVector>, target_activity: Vec<f32>) -> Self {
        assert!(!history.is_empty(), "history must not be empty");
        assert!(
            !target_activity.is_empty(),
            "target_activity must not be empty"
        );
        assert!(
            target_activity
                .iter()
                .all(|target| (0.0..=1.0).contains(target)),
            "target activity must be in [0, 1]"
        );
        Self {
            history,
            target_activity,
        }
    }
}
