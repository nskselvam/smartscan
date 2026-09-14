// SMARTSCAN - ML-Based Adaptive Scan Strategy for Electronic Warfare
// Smart India Hackathon 2026 - PS 26055

pub mod cli;
pub mod config;
pub mod logging;

// Core modules (to be implemented in subsequent phases)
pub mod data;
pub mod dl;
pub mod evaluation;
pub mod gui;
pub mod periodic;
pub mod ppo;
pub mod scheduler;
pub mod simulator;
pub mod storage;

pub use config::Config;
pub use logging::init_logging;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Version {
    pub major: u8,
    pub minor: u8,
    pub patch: u8,
}

impl Version {
    pub const fn new(major: u8, minor: u8, patch: u8) -> Self {
        Self {
            major,
            minor,
            patch,
        }
    }
}

impl std::fmt::Display for Version {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}.{}.{}", self.major, self.minor, self.patch)
    }
}

pub const VERSION: Version = Version::new(0, 1, 0);
pub const PROJECT_NAME: &str = "SMARTSCAN";
pub const PROJECT_DESC: &str = "ML-Based Adaptive Scan Strategy for Electronic Warfare";
