use anyhow::Result;
use std::io;
use tracing_subscriber::filter::EnvFilter;
use tracing_subscriber::fmt;

pub fn init_logging(level: &str) -> Result<()> {
    let env_filter = EnvFilter::try_from_default_env()
        .or_else(|_| EnvFilter::try_new(level))
        .unwrap_or_else(|_| EnvFilter::new("info"));

    fmt()
        .with_env_filter(env_filter)
        .with_writer(io::stderr)
        .with_target(true)
        .with_level(true)
        .with_thread_ids(false)
        .init();

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_init_logging() {
        let result = init_logging("debug");
        assert!(result.is_ok());
    }
}
