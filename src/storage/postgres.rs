use std::env;

use postgres::{Client, NoTls};
use thiserror::Error;

use crate::evaluation::BenchmarkReport;

const DATABASE_URL_ENV: &str = "DATABASE_URL";

#[derive(Debug, Error)]
pub enum StorageError {
    #[error("{DATABASE_URL_ENV} is not set; PostgreSQL experiment storage is disabled")]
    MissingDatabaseUrl,
    #[error("PostgreSQL error: {0}")]
    Postgres(#[from] postgres::Error),
    #[error("experiment serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
}

/// Optional local PostgreSQL persistence for actual experiment results.
/// Credentials remain outside the repository in `DATABASE_URL`.
pub struct PostgresExperimentStore {
    client: Client,
}

impl PostgresExperimentStore {
    pub fn connect_from_environment() -> Result<Self, StorageError> {
        let database_url =
            env::var(DATABASE_URL_ENV).map_err(|_| StorageError::MissingDatabaseUrl)?;
        Self::connect(&database_url)
    }

    pub fn connect(database_url: &str) -> Result<Self, StorageError> {
        let mut store = Self {
            client: Client::connect(database_url, NoTls)?,
        };
        store.initialize_schema()?;
        Ok(store)
    }

    pub fn initialize_schema(&mut self) -> Result<(), StorageError> {
        self.client.batch_execute(
            "
            CREATE TABLE IF NOT EXISTS smartscan_experiments (
                id BIGSERIAL PRIMARY KEY,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                scenario TEXT NOT NULL,
                episodes BIGINT NOT NULL CHECK (episodes > 0),
                max_steps BIGINT NOT NULL CHECK (max_steps > 0),
                seed BIGINT NOT NULL,
                report JSONB NOT NULL
            );
            CREATE INDEX IF NOT EXISTS smartscan_experiments_created_at_idx
                ON smartscan_experiments (created_at DESC);
            ",
        )?;
        Ok(())
    }

    pub fn save_benchmark(&mut self, report: &BenchmarkReport) -> Result<i64, StorageError> {
        let report_json = serde_json::to_value(report)?;
        let row = self.client.query_one(
            "INSERT INTO smartscan_experiments (scenario, episodes, max_steps, seed, report)
             VALUES ($1, $2, $3, $4, $5)
             RETURNING id",
            &[
                &report.config.scenario,
                &(report.config.episodes as i64),
                &(report.config.max_steps as i64),
                &(report.config.seed as i64),
                &report_json,
            ],
        )?;
        Ok(row.get(0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn storage_requires_an_explicit_database_url() {
        let key = DATABASE_URL_ENV;
        let saved = env::var_os(key);
        env::remove_var(key);
        assert!(matches!(
            PostgresExperimentStore::connect_from_environment(),
            Err(StorageError::MissingDatabaseUrl)
        ));
        if let Some(value) = saved {
            env::set_var(key, value);
        }
    }
}
