/// RF Environment Simulator
///
/// Implements a deterministic/reproducible RF simulation environment.
/// Supports:
/// - Fixed-frequency emitters
/// - Periodic emitters
/// - Intermittent emitters
/// - Frequency-agile emitters
/// - Frequency-hopping emitters
/// - Multiple simultaneous emitters
/// - Short-duration transmissions
/// - Noisy observations
/// - False alarms
pub mod emitter;
pub mod environment;
pub mod ground_truth;
pub mod receiver;
pub mod scenarios;

pub use emitter::{Emitter, EmitterType};
pub use environment::{RewardConfig, RfEnvironment, StepInfo, StepResult};
pub use ground_truth::GroundTruth;
pub use receiver::Receiver;
pub use scenarios::Scenario;
