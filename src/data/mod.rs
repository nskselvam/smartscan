pub mod features;
pub mod hdf5_loader;
pub mod streaming;
pub mod windows;

pub use features::{FeatureNormalizer, FeatureVector, NormalizationConfig, NormalizationStats};
pub use hdf5_loader::{load_tsrd_pulses, preprocess_tsrd_hdf5, TsrdLoadError, TsrdLoadReport};
pub use streaming::{
    inspect_directory, preprocess_csv_file, write_processed_index, CsvPulseReader,
    DatasetInspection, ProcessedFileIndex, PulseRecord, StreamingError,
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inspection_reports_a_missing_directory() {
        let inspection = inspect_directory("does-not-exist").expect("inspection should succeed");
        assert_eq!(inspection.file_count, 0);
        assert_eq!(inspection.total_bytes, 0);
    }
}
