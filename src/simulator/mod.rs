pub mod emitter;
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
pub mod environment;
pub mod ground_truth;
pub mod receiver;
pub mod scenarios;
