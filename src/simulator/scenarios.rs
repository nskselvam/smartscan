/// Simulation Scenarios
///
/// Predefined scenarios for testing and evaluation.
pub struct Scenario {
    pub name: String,
    pub description: String,
}

impl Scenario {
    pub fn new(name: String, description: String) -> Self {
        Scenario { name, description }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scenario_creation() {
        let scenario = Scenario::new("basic".to_string(), "Basic test scenario".to_string());
        assert_eq!(scenario.name, "basic");
    }
}
