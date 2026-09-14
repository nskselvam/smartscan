use std::path::Path;
use std::{fs, io::BufWriter, io::Write};

use hdf5::File;
use thiserror::Error;

use crate::data::{ProcessedFileIndex, PulseRecord};

const PDW_WIDTH: usize = 5;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TsrdLoadReport {
    pub source_rows: usize,
    pub loaded_rows: usize,
    pub feature_count: usize,
}

#[derive(Debug, Error)]
pub enum TsrdLoadError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("HDF5 error: {0}")]
    Hdf5(#[from] hdf5::Error),
    #[error("TSRD /data must be a two-dimensional PDW dataset")]
    InvalidRank,
    #[error("TSRD /data has {actual} PDW columns; expected {expected}")]
    InvalidWidth { actual: usize, expected: usize },
    #[error("max_records must be positive")]
    ZeroLimit,
}

/// Reads at most `max_records` PDWs from TSRD's `/data` dataset. The expected
/// column order is ToA, centre frequency, pulse width, AoA, and amplitude.
pub fn load_tsrd_pulses(
    path: impl AsRef<Path>,
    max_records: usize,
) -> Result<(Vec<PulseRecord>, TsrdLoadReport), TsrdLoadError> {
    if max_records == 0 {
        return Err(TsrdLoadError::ZeroLimit);
    }
    configure_plugin_path()?;
    let file = File::open(path)?;
    let dataset = file.dataset("data")?;
    let shape = dataset.shape();
    if shape.len() != 2 {
        return Err(TsrdLoadError::InvalidRank);
    }
    if shape[1] != PDW_WIDTH {
        return Err(TsrdLoadError::InvalidWidth {
            actual: shape[1],
            expected: PDW_WIDTH,
        });
    }
    let loaded_rows = shape[0].min(max_records);
    let values = dataset.read_slice_2d::<f64, _>(ndarray::s![0..loaded_rows, 0..PDW_WIDTH])?;
    let pulses = values
        .outer_iter()
        .filter_map(|row| {
            let toa = row[0];
            toa.is_finite().then_some(PulseRecord {
                toa,
                center_frequency: optional_value(row[1]),
                pulse_width: optional_value(row[2]),
                aoa: optional_value(row[3]),
                amplitude: optional_value(row[4]),
            })
        })
        .collect();
    Ok((
        pulses,
        TsrdLoadReport {
            source_rows: shape[0],
            loaded_rows,
            feature_count: shape[1],
        },
    ))
}

/// Reads a bounded TSVD PDW prefix and writes SMARTSCAN's compact pulse format.
pub fn preprocess_tsrd_hdf5(
    input_path: impl AsRef<Path>,
    output_path: impl AsRef<Path>,
    max_records: usize,
) -> Result<ProcessedFileIndex, TsrdLoadError> {
    let input_path = input_path.as_ref();
    let output_path = output_path.as_ref();
    let (pulses, _) = load_tsrd_pulses(input_path, max_records)?;
    if let Some(parent) = output_path.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut writer = BufWriter::new(fs::File::create(output_path)?);
    writer.write_all(b"SSP1")?;
    for pulse in &pulses {
        for value in [
            pulse.toa,
            pulse.center_frequency.unwrap_or(f64::NAN),
            pulse.pulse_width.unwrap_or(f64::NAN),
            pulse.amplitude.unwrap_or(f64::NAN),
            pulse.aoa.unwrap_or(f64::NAN),
        ] {
            writer.write_all(&value.to_le_bytes())?;
        }
    }
    writer.flush()?;
    Ok(ProcessedFileIndex {
        format: "smartscan-pulse-v1".to_string(),
        source_path: input_path.to_path_buf(),
        output_path: output_path.to_path_buf(),
        pulse_count: pulses.len() as u64,
        record_size_bytes: PDW_WIDTH * std::mem::size_of::<f64>(),
    })
}

fn optional_value(value: f64) -> Option<f64> {
    value.is_finite().then_some(value)
}

fn configure_plugin_path() -> Result<(), std::io::Error> {
    if std::env::var_os("HDF5_PLUGIN_PATH").is_none() {
        let path = std::env::temp_dir().join("smartscan-hdf5-plugins");
        fs::create_dir_all(&path)?;
        std::env::set_var("HDF5_PLUGIN_PATH", path);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_a_zero_record_limit() {
        let error = load_tsrd_pulses("missing.h5", 0).expect_err("zero limit should be rejected first");
        assert!(matches!(error, TsrdLoadError::ZeroLimit));
    }
}