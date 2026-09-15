use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;

use thiserror::Error;

use crate::data::{FeatureNormalizer, PulseRecord};
use crate::dl::LabeledSequence;

const RECORD_BYTES: usize = 5 * std::mem::size_of::<f64>();

#[derive(Debug, Error)]
pub enum ActivityDataError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("invalid SMARTSCAN pulse file header")]
    InvalidHeader,
    #[error("compact pulse file ends mid-record")]
    TruncatedRecord,
}

pub fn read_compact_pulses(
    path: impl AsRef<Path>,
    max_records: usize,
) -> Result<Vec<PulseRecord>, ActivityDataError> {
    let mut reader = BufReader::new(File::open(path)?);
    let mut header = [0_u8; 4];
    reader.read_exact(&mut header)?;
    if &header != b"SSP1" {
        return Err(ActivityDataError::InvalidHeader);
    }
    let mut pulses = Vec::with_capacity(max_records);
    let mut record = [0_u8; RECORD_BYTES];
    while pulses.len() < max_records {
        match reader.read_exact(&mut record) {
            Ok(()) => pulses.push(PulseRecord {
                toa: read_f64(&record[0..8]),
                center_frequency: optional(read_f64(&record[8..16])),
                pulse_width: optional(read_f64(&record[16..24])),
                amplitude: optional(read_f64(&record[24..32])),
                aoa: optional(read_f64(&record[32..40])),
            }),
            Err(error) if error.kind() == std::io::ErrorKind::UnexpectedEof => {
                let bytes_read = reader.read(&mut [0_u8; 1])?;
                if bytes_read == 0 {
                    break;
                }
                return Err(ActivityDataError::TruncatedRecord);
            }
            Err(error) => return Err(error.into()),
        }
    }
    Ok(pulses)
}

/// Creates chronological next-pulse frequency-band activity labels. Every
/// target describes the pulse immediately after its history; no future PDWs
/// are included in its feature sequence.
pub fn build_activity_sequences(
    pulses: &[PulseRecord],
    normalizer: &FeatureNormalizer,
    num_bands: usize,
    sequence_length: usize,
    max_samples: usize,
) -> Vec<LabeledSequence> {
    assert!(num_bands > 0 && sequence_length > 0);
    let features = pulses
        .iter()
        .map(|pulse| normalizer.transform(pulse))
        .collect::<Vec<_>>();
    if features.len() <= sequence_length {
        return Vec::new();
    }
    let min_frequency = pulses
        .iter()
        .filter_map(|pulse| pulse.center_frequency)
        .fold(f64::INFINITY, f64::min);
    let max_frequency = pulses
        .iter()
        .filter_map(|pulse| pulse.center_frequency)
        .fold(f64::NEG_INFINITY, f64::max);
    if !min_frequency.is_finite() || min_frequency >= max_frequency {
        return Vec::new();
    }
    (0..features.len() - sequence_length)
        .take(max_samples)
        .filter_map(|start| {
            let target_frequency = pulses[start + sequence_length].center_frequency?;
            let band = (((target_frequency - min_frequency) / (max_frequency - min_frequency))
                .clamp(0.0, 1.0)
                * num_bands as f64)
                .floor() as usize;
            let mut target_activity = vec![0.0; num_bands];
            target_activity[band.min(num_bands - 1)] = 1.0;
            Some(LabeledSequence::new(
                features[start..start + sequence_length].to_vec(),
                target_activity,
            ))
        })
        .collect()
}

fn read_f64(bytes: &[u8]) -> f64 {
    f64::from_le_bytes(
        bytes
            .try_into()
            .expect("compact record fields are eight bytes"),
    )
}

fn optional(value: f64) -> Option<f64> {
    value.is_finite().then_some(value)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::{NormalizationConfig, PulseRecord};

    fn pulse(toa: f64, frequency: f64) -> PulseRecord {
        PulseRecord {
            toa,
            center_frequency: Some(frequency),
            pulse_width: Some(1.0),
            amplitude: Some(-30.0),
            aoa: Some(0.0),
        }
    }

    #[test]
    fn sequences_target_the_next_unseen_pulse_band() {
        let pulses = vec![pulse(0.0, 100.0), pulse(1.0, 200.0), pulse(2.0, 300.0)];
        let normalizer = FeatureNormalizer::fit(&pulses, NormalizationConfig::default());
        let sequences = build_activity_sequences(&pulses, &normalizer, 3, 2, 10);
        assert_eq!(sequences.len(), 1);
        assert_eq!(sequences[0].target_activity, vec![0.0, 0.0, 1.0]);
    }
}
