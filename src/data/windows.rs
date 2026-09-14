use crate::data::FeatureVector;

#[derive(Debug, Clone, PartialEq)]
pub struct TemporalWindow {
    pub history: Vec<FeatureVector>,
    pub target: FeatureVector,
    pub target_index: usize,
}

/// Builds chronological windows. The target begins after the final history item,
/// separated by `prediction_horizon - 1` unobserved future positions.
pub fn generate_windows(
    features: &[FeatureVector],
    sequence_length: usize,
    prediction_horizon: usize,
) -> Vec<TemporalWindow> {
    assert!(sequence_length > 0, "sequence_length must be positive");
    assert!(
        prediction_horizon > 0,
        "prediction_horizon must be positive"
    );

    let target_offset = sequence_length + prediction_horizon - 1;
    if features.len() <= target_offset {
        return Vec::new();
    }
    (0..features.len() - target_offset)
        .map(|start| {
            let target_index = start + target_offset;
            TemporalWindow {
                history: features[start..start + sequence_length].to_vec(),
                target: features[target_index].clone(),
                target_index,
            }
        })
        .collect()
}

/// Splits whole scenarios, never individual temporal windows, to prevent a
/// scenario's future behavior from appearing in both train and validation data.
pub fn split_scenario_ids(
    scenario_ids: &[String],
    validation_fraction: f32,
) -> (Vec<String>, Vec<String>) {
    assert!((0.0..1.0).contains(&validation_fraction));
    let validation_start =
        ((scenario_ids.len() as f32) * (1.0 - validation_fraction)).floor() as usize;
    (
        scenario_ids[..validation_start].to_vec(),
        scenario_ids[validation_start..].to_vec(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::features::FEATURE_COUNT;

    fn feature(value: f32) -> FeatureVector {
        FeatureVector {
            values: [value; FEATURE_COUNT],
        }
    }

    #[test]
    fn windows_keep_future_targets_out_of_history() {
        let features = (0..6)
            .map(|index| feature(index as f32))
            .collect::<Vec<_>>();
        let windows = generate_windows(&features, 2, 2);

        assert_eq!(windows.len(), 3);
        assert_eq!(windows[0].history[0].values[0], 0.0);
        assert_eq!(windows[0].history[1].values[0], 1.0);
        assert_eq!(windows[0].target.values[0], 3.0);
        assert_eq!(windows[0].target_index, 3);
    }

    #[test]
    fn splits_do_not_break_scenarios_apart() {
        let ids = ["scenario-a", "scenario-b", "scenario-c", "scenario-d"]
            .into_iter()
            .map(str::to_string)
            .collect::<Vec<_>>();
        let (training, validation) = split_scenario_ids(&ids, 0.25);
        assert_eq!(training, vec!["scenario-a", "scenario-b", "scenario-c"]);
        assert_eq!(validation, vec!["scenario-d"]);
    }
}
