use crate::data::PulseRecord;

pub const FEATURE_COUNT: usize = 10;

#[derive(Debug, Clone, PartialEq)]
pub struct FeatureVector {
    pub values: [f32; FEATURE_COUNT],
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NormalizationConfig {
    pub max_abs_zscore: f64,
}

impl Default for NormalizationConfig {
    fn default() -> Self {
        Self {
            max_abs_zscore: 5.0,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NormalizationStats {
    pub mean: f64,
    pub standard_deviation: f64,
}

#[derive(Debug, Clone)]
pub struct FeatureNormalizer {
    stats: [NormalizationStats; 5],
    config: NormalizationConfig,
}

impl FeatureNormalizer {
    /// Fits only the training-scenario records supplied by the caller.
    pub fn fit(training_records: &[PulseRecord], config: NormalizationConfig) -> Self {
        assert!(
            !training_records.is_empty(),
            "training records must not be empty"
        );
        let columns = [
            training_records
                .iter()
                .map(|record| Some(record.toa))
                .collect::<Vec<_>>(),
            training_records
                .iter()
                .map(|record| record.center_frequency)
                .collect(),
            training_records
                .iter()
                .map(|record| record.pulse_width)
                .collect(),
            training_records
                .iter()
                .map(|record| record.amplitude)
                .collect(),
            training_records.iter().map(|record| record.aoa).collect(),
        ];
        let stats = columns.map(|column| calculate_stats(&column));
        Self { stats, config }
    }

    pub fn transform(&self, record: &PulseRecord) -> FeatureVector {
        let fields = [
            Some(record.toa),
            record.center_frequency,
            record.pulse_width,
            record.amplitude,
            record.aoa,
        ];
        let mut values = [0.0; FEATURE_COUNT];
        for (index, field) in fields.iter().enumerate() {
            if let Some(value) = field.filter(|value| value.is_finite()) {
                let stat = self.stats[index];
                let zscore = ((value - stat.mean) / stat.standard_deviation)
                    .clamp(-self.config.max_abs_zscore, self.config.max_abs_zscore);
                values[index] = zscore as f32;
            } else {
                values[index] = 0.0;
                values[index + 5] = 1.0;
            }
        }
        FeatureVector { values }
    }

    pub fn stats(&self) -> &[NormalizationStats; 5] {
        &self.stats
    }
}

fn calculate_stats(values: &[Option<f64>]) -> NormalizationStats {
    let valid: Vec<f64> = values
        .iter()
        .flatten()
        .copied()
        .filter(|value| value.is_finite())
        .collect();
    if valid.is_empty() {
        return NormalizationStats {
            mean: 0.0,
            standard_deviation: 1.0,
        };
    }
    let mean = valid.iter().sum::<f64>() / valid.len() as f64;
    let variance = valid
        .iter()
        .map(|value| (value - mean).powi(2))
        .sum::<f64>()
        / valid.len() as f64;
    NormalizationStats {
        mean,
        standard_deviation: variance.sqrt().max(f64::EPSILON),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pulse(toa: f64, frequency: Option<f64>) -> PulseRecord {
        PulseRecord {
            toa,
            center_frequency: frequency,
            pulse_width: Some(2.0),
            amplitude: Some(-30.0),
            aoa: None,
        }
    }

    #[test]
    fn missing_features_are_imputed_and_indicated() {
        let normalizer =
            FeatureNormalizer::fit(&[pulse(1.0, Some(100.0))], NormalizationConfig::default());
        let vector = normalizer.transform(&pulse(2.0, None));
        assert_eq!(vector.values[1], 0.0);
        assert_eq!(vector.values[6], 1.0);
        assert_eq!(vector.values[9], 1.0);
    }

    #[test]
    fn training_statistics_do_not_include_held_out_values() {
        let normalizer = FeatureNormalizer::fit(
            &[pulse(0.0, Some(0.0)), pulse(2.0, Some(2.0))],
            NormalizationConfig::default(),
        );
        assert_eq!(normalizer.stats()[0].mean, 1.0);
        let held_out = normalizer.transform(&pulse(100.0, Some(100.0)));
        assert_eq!(held_out.values[0], 5.0);
    }
}
