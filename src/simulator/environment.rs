/// RF Environment
///
/// Represents the RF environment as a time × frequency grid
/// with ground-truth transmission states.
pub struct RfEnvironment {
    pub num_bands: usize,
    pub time_horizon: usize,
}

impl RfEnvironment {
    pub fn new(num_bands: usize, time_horizon: usize) -> Self {
        RfEnvironment {
            num_bands,
            time_horizon,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_environment_creation() {
        let env = RfEnvironment::new(32, 1000);
        assert_eq!(env.num_bands, 32);
        assert_eq!(env.time_horizon, 1000);
    }
}
