use super::model::binary_cross_entropy;

#[derive(Debug, Clone, PartialEq)]
pub struct PredictionMetrics {
    pub bce_loss: f32,
    pub mae: f32,
    pub rmse: f32,
    pub precision: f32,
    pub recall: f32,
    pub f1: f32,
}

pub fn evaluate_predictions(predictions: &[Vec<f32>], targets: &[Vec<f32>]) -> PredictionMetrics {
    assert_eq!(predictions.len(), targets.len());
    let mut bce_sum = 0.0;
    let mut absolute_error_sum = 0.0;
    let mut squared_error_sum = 0.0;
    let mut count = 0_usize;
    let mut true_positives = 0_usize;
    let mut false_positives = 0_usize;
    let mut false_negatives = 0_usize;

    for (prediction, target) in predictions.iter().zip(targets) {
        assert_eq!(prediction.len(), target.len());
        bce_sum += binary_cross_entropy(prediction, target) * prediction.len() as f32;
        for (&probability, &label) in prediction.iter().zip(target) {
            let error = probability - label;
            absolute_error_sum += error.abs();
            squared_error_sum += error * error;
            count += 1;
            match (probability >= 0.5, label >= 0.5) {
                (true, true) => true_positives += 1,
                (true, false) => false_positives += 1,
                (false, true) => false_negatives += 1,
                (false, false) => {}
            }
        }
    }

    let denominator = count.max(1) as f32;
    let precision = count_ratio(true_positives, true_positives + false_positives);
    let recall = count_ratio(true_positives, true_positives + false_negatives);
    PredictionMetrics {
        bce_loss: bce_sum / denominator,
        mae: absolute_error_sum / denominator,
        rmse: (squared_error_sum / denominator).sqrt(),
        precision,
        recall,
        f1: float_ratio(2.0 * precision * recall, precision + recall),
    }
}

fn count_ratio(numerator: usize, denominator: usize) -> f32 {
    if denominator == 0 {
        0.0
    } else {
        numerator as f32 / denominator as f32
    }
}

fn float_ratio(numerator: f32, denominator: f32) -> f32 {
    if denominator == 0.0 {
        0.0
    } else {
        numerator / denominator
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn metrics_are_calculated_from_predictions() {
        let metrics = evaluate_predictions(&[vec![0.9, 0.2]], &[vec![1.0, 0.0]]);
        assert_eq!(metrics.precision, 1.0);
        assert_eq!(metrics.recall, 1.0);
        assert_eq!(metrics.f1, 1.0);
        assert!(metrics.bce_loss < 0.2);
    }
}
