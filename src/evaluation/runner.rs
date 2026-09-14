use crate::scheduler::Scheduler;
use crate::simulator::RfEnvironment;

#[derive(Debug, Clone, PartialEq)]
pub struct EpisodeSummary {
    pub scheduler: &'static str,
    pub steps: usize,
    pub total_reward: f32,
    pub active_bands: usize,
    pub monitored_bands: usize,
    pub monitored_active_bands: usize,
    pub monitored_inactive_bands: usize,
    pub transmission_starts: usize,
    pub true_positives: usize,
    pub false_positives: usize,
    pub missed_active_bands: usize,
    pub interceptions: usize,
    pub intercept_delay_sum: usize,
}

pub fn run_episode(
    environment: &mut RfEnvironment,
    scheduler: &mut impl Scheduler,
    max_steps: usize,
) -> EpisodeSummary {
    scheduler.reset();
    let steps_to_run = max_steps.min(environment.time_horizon - environment.current_time);
    let mut summary = EpisodeSummary {
        scheduler: scheduler.name(),
        steps: 0,
        total_reward: 0.0,
        active_bands: 0,
        monitored_bands: 0,
        monitored_active_bands: 0,
        monitored_inactive_bands: 0,
        transmission_starts: 0,
        true_positives: 0,
        false_positives: 0,
        missed_active_bands: 0,
        interceptions: 0,
        intercept_delay_sum: 0,
    };

    for _ in 0..steps_to_run {
        let selected_band = scheduler.select_band();
        let result = environment.step(selected_band);
        scheduler.update(result.reward);

        summary.steps += 1;
        summary.total_reward += result.reward;
        summary.active_bands += result.info.active_bands;
        summary.monitored_bands += result.info.monitored_bands;
        summary.monitored_active_bands += result.info.monitored_active_bands;
        summary.monitored_inactive_bands += result.info.monitored_inactive_bands;
        summary.transmission_starts += result.info.transmission_starts;
        summary.true_positives += result.info.true_positives;
        summary.false_positives += result.info.false_positives;
        summary.missed_active_bands += result.info.missed_active_bands;
        summary.interceptions += result.info.interceptions;
        summary.intercept_delay_sum += result.info.intercept_delay_sum;

        if result.done {
            break;
        }
    }
    summary
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scheduler::RoundRobinScheduler;
    use crate::simulator::Scenario;

    #[test]
    fn runner_records_the_complete_episode() {
        let mut environment = Scenario::fixed_frequency().into_environment();
        let mut scheduler = RoundRobinScheduler::new(environment.num_bands);
        let summary = run_episode(&mut environment, &mut scheduler, 1_000);

        assert_eq!(summary.scheduler, "round-robin");
        assert_eq!(summary.steps, 1_000);
        assert!(summary.active_bands > 0);
        assert!(summary.true_positives > 0);
    }
}
