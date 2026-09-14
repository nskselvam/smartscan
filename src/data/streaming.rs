use std::collections::HashMap;
use std::fs;
use std::fs::File;
use std::io::{BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};

use csv::{Reader, StringRecord};
use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum StreamingError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("CSV error: {0}")]
    Csv(#[from] csv::Error),
    #[error("missing required time-of-arrival column; accepted names: toa, time_of_arrival, time")]
    MissingToa,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PulseRecord {
    pub toa: f64,
    pub center_frequency: Option<f64>,
    pub pulse_width: Option<f64>,
    pub amplitude: Option<f64>,
    pub aoa: Option<f64>,
}

/// Metadata stored separately from the compact fixed-width pulse file.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProcessedFileIndex {
    pub format: String,
    pub source_path: PathBuf,
    pub output_path: PathBuf,
    pub pulse_count: u64,
    pub record_size_bytes: usize,
}

pub struct CsvPulseReader {
    reader: Reader<BufReader<File>>,
    columns: HashMap<String, usize>,
}

impl CsvPulseReader {
    pub fn open(path: impl AsRef<Path>) -> Result<Self, StreamingError> {
        let file = File::open(path)?;
        let mut reader = csv::ReaderBuilder::new()
            .flexible(true)
            .from_reader(BufReader::new(file));
        let headers = reader.headers()?.clone();
        let columns = headers
            .iter()
            .enumerate()
            .map(|(index, name)| (normalize_header(name), index))
            .collect::<HashMap<_, _>>();
        if column_index(&columns, &["toa", "time_of_arrival", "time"]).is_none() {
            return Err(StreamingError::MissingToa);
        }
        Ok(Self { reader, columns })
    }

    /// Reads no more than `chunk_size` records. No complete dataset is retained.
    pub fn next_chunk(&mut self, chunk_size: usize) -> Result<Vec<PulseRecord>, StreamingError> {
        assert!(chunk_size > 0, "chunk_size must be positive");
        let mut chunk = Vec::with_capacity(chunk_size);
        let columns = &self.columns;
        for row in self.reader.records().take(chunk_size) {
            let row = row?;
            if let Some(pulse) = parse_row(&row, columns) {
                chunk.push(pulse);
            }
        }
        Ok(chunk)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DatasetInspection {
    pub root: PathBuf,
    pub file_count: usize,
    pub total_bytes: u64,
    pub csv_files: usize,
}

pub fn inspect_directory(path: impl AsRef<Path>) -> Result<DatasetInspection, StreamingError> {
    let root = path.as_ref().to_path_buf();
    let mut inspection = DatasetInspection {
        root: root.clone(),
        file_count: 0,
        total_bytes: 0,
        csv_files: 0,
    };
    if !root.exists() {
        return Ok(inspection);
    }
    inspect_recursive(&root, &mut inspection)?;
    Ok(inspection)
}

/// Stream one CSV source into a compact, fixed-width binary pulse file.
/// Each record uses five IEEE 754 f64 values in this order: ToA, frequency,
/// pulse width, amplitude, and AoA. Missing optional values are encoded as NaN.
pub fn preprocess_csv_file(
    input_path: impl AsRef<Path>,
    output_path: impl AsRef<Path>,
    chunk_size: usize,
) -> Result<ProcessedFileIndex, StreamingError> {
    let input_path = input_path.as_ref();
    let output_path = output_path.as_ref();
    if let Some(parent) = output_path.parent() {
        fs::create_dir_all(parent)?;
    }

    let mut reader = CsvPulseReader::open(input_path)?;
    let output_file = File::create(output_path)?;
    let mut writer = BufWriter::new(output_file);
    writer.write_all(b"SSP1")?;

    let mut pulse_count = 0_u64;
    loop {
        let chunk = reader.next_chunk(chunk_size)?;
        if chunk.is_empty() {
            break;
        }
        for pulse in chunk {
            for value in [
                pulse.toa,
                pulse.center_frequency.unwrap_or(f64::NAN),
                pulse.pulse_width.unwrap_or(f64::NAN),
                pulse.amplitude.unwrap_or(f64::NAN),
                pulse.aoa.unwrap_or(f64::NAN),
            ] {
                writer.write_all(&value.to_le_bytes())?;
            }
            pulse_count += 1;
        }
    }
    writer.flush()?;

    Ok(ProcessedFileIndex {
        format: "smartscan-pulse-v1".to_string(),
        source_path: input_path.to_path_buf(),
        output_path: output_path.to_path_buf(),
        pulse_count,
        record_size_bytes: 5 * std::mem::size_of::<f64>(),
    })
}

pub fn write_processed_index(
    index_path: impl AsRef<Path>,
    index: &ProcessedFileIndex,
) -> Result<(), StreamingError> {
    let index_path = index_path.as_ref();
    if let Some(parent) = index_path.parent() {
        fs::create_dir_all(parent)?;
    }
    let json = serde_json::to_vec_pretty(index)
        .map_err(|error| std::io::Error::other(error.to_string()))?;
    fs::write(index_path, json)?;
    Ok(())
}

fn inspect_recursive(
    path: &Path,
    inspection: &mut DatasetInspection,
) -> Result<(), StreamingError> {
    for entry in fs::read_dir(path)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() {
            inspect_recursive(&path, inspection)?;
        } else {
            inspection.file_count += 1;
            inspection.total_bytes += entry.metadata()?.len();
            if path
                .extension()
                .is_some_and(|extension| extension.eq_ignore_ascii_case("csv"))
            {
                inspection.csv_files += 1;
            }
        }
    }
    Ok(())
}

fn normalize_header(header: &str) -> String {
    header.trim().to_ascii_lowercase().replace([' ', '-'], "_")
}

fn column_index(columns: &HashMap<String, usize>, aliases: &[&str]) -> Option<usize> {
    aliases
        .iter()
        .find_map(|alias| columns.get(*alias).copied())
}

fn parse_row(row: &StringRecord, columns: &HashMap<String, usize>) -> Option<PulseRecord> {
    let toa = value(row, columns, &["toa", "time_of_arrival", "time"])?;
    Some(PulseRecord {
        toa,
        center_frequency: value(row, columns, &["center_frequency", "frequency", "freq"]),
        pulse_width: value(row, columns, &["pulse_width", "pw"]),
        amplitude: value(row, columns, &["amplitude", "amp"]),
        aoa: value(row, columns, &["aoa", "angle_of_arrival"]),
    })
}

fn value(row: &StringRecord, columns: &HashMap<String, usize>, aliases: &[&str]) -> Option<f64> {
    column_index(columns, aliases)
        .and_then(|index| row.get(index))
        .and_then(|raw| raw.trim().parse::<f64>().ok())
        .filter(|value| value.is_finite())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn reader_processes_bounded_chunks_and_skips_invalid_toa() {
        let path =
            std::env::temp_dir().join(format!("smartscan-pulses-{}.csv", std::process::id()));
        let mut file = File::create(&path).expect("fixture file should be created");
        writeln!(file, "ToA,Frequency,PW,Amplitude,AoA").expect("fixture header should be written");
        writeln!(file, "1.0,1000,2.0,-30,40").expect("fixture row should be written");
        writeln!(file, "invalid,1001,2.0,-31,41").expect("fixture row should be written");
        writeln!(file, "3.0,1002,2.5,-32,42").expect("fixture row should be written");

        let mut reader = CsvPulseReader::open(&path).expect("reader should open");
        let first_chunk = reader.next_chunk(2).expect("chunk should parse");
        let second_chunk = reader.next_chunk(2).expect("chunk should parse");

        assert_eq!(first_chunk.len(), 1);
        assert_eq!(second_chunk.len(), 1);
        assert_eq!(second_chunk[0].toa, 3.0);
        fs::remove_file(path).expect("fixture should be removed");
    }

    #[test]
    fn preprocessing_writes_compact_records_without_retaining_source() {
        let base =
            std::env::temp_dir().join(format!("smartscan-preprocess-{}", std::process::id()));
        let input = base.with_extension("csv");
        let output = base.with_extension("ssp");
        fs::write(&input, "toa,frequency\n1.0,1000\n2.0,1001\n")
            .expect("fixture should be written");

        let index = preprocess_csv_file(&input, &output, 1).expect("preprocessing should succeed");

        assert_eq!(index.pulse_count, 2);
        assert_eq!(index.record_size_bytes, 40);
        assert_eq!(
            fs::metadata(&output)
                .expect("output metadata should exist")
                .len(),
            84
        );
        fs::remove_file(input).expect("fixture should be removed");
        fs::remove_file(output).expect("fixture should be removed");
    }
}
