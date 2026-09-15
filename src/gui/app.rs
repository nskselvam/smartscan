use eframe::egui::{self, Align2, Color32, FontId, Pos2, Rect, Stroke, StrokeKind, Vec2};
use std::time::{Duration, Instant};

use crate::data::{FeatureVector, FEATURE_COUNT};
use crate::dl::{GruActivityPredictor, GruConfig};
use crate::periodic::ExplorationReserve;
use crate::ppo::{PpoAgent, PpoConfig, PpoEnvironment};
use crate::simulator::{Scenario, StepResult};

const MAX_HISTORY_ROWS: usize = 60;
const MAX_CHART_SAMPLES: usize = 120;
const AUTO_STEP_INTERVAL: Duration = Duration::from_secs(2);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DashboardView {
    Overview,
    Spectrum,
    BandReport,
    Activity,
}

struct HeatmapRow {
    truth: Vec<bool>,
    observation: Vec<bool>,
    selected_band: usize,
}

#[derive(Debug, Clone)]
struct ScanEvent {
    time: usize,
    band: usize,
    hit: bool,
    reward: f32,
}

#[derive(Default)]
struct LiveMetrics {
    total_reward: f32,
    true_positives: usize,
    false_positives: usize,
    active_bands: usize,
    monitored_inactive_bands: usize,
    transmission_starts: usize,
    interceptions: usize,
    intercept_delay_sum: usize,
}

impl LiveMetrics {
    fn record(&mut self, result: &StepResult) {
        self.total_reward += result.reward;
        self.true_positives += result.info.true_positives;
        self.false_positives += result.info.false_positives;
        self.active_bands += result.info.active_bands;
        self.monitored_inactive_bands += result.info.monitored_inactive_bands;
        self.transmission_starts += result.info.transmission_starts;
        self.interceptions += result.info.interceptions;
        self.intercept_delay_sum += result.info.intercept_delay_sum;
    }

    fn ratio(numerator: usize, denominator: usize) -> f32 {
        if denominator == 0 {
            0.0
        } else {
            numerator as f32 / denominator as f32
        }
    }
}

pub struct GuiApp {
    environment: PpoEnvironment,
    predictor: GruActivityPredictor,
    ppo_agent: PpoAgent,
    auto_run: bool,
    history: Vec<HeatmapRow>,
    activity_predictions: Vec<f32>,
    action_probabilities: Vec<f32>,
    feature_history: Vec<FeatureVector>,
    reward_history: Vec<f32>,
    hit_history: Vec<f32>,
    scans_by_band: Vec<usize>,
    hits_by_band: Vec<usize>,
    last_scan_time_by_band: Vec<Option<usize>>,
    recent_events: Vec<ScanEvent>,
    exploration_reserve: ExplorationReserve,
    scan_decisions: usize,
    view: DashboardView,
    last_auto_step: Instant,
    metrics: LiveMetrics,
    last_result: Option<StepResult>,
}

impl GuiApp {
    pub fn new() -> Self {
        let scenario = Scenario::long_running_mixed();
        let num_bands = scenario.num_bands;
        let environment = PpoEnvironment::new(scenario.into_environment());
        let ppo_config = PpoConfig {
            state_size: environment.observation().to_vector().len(),
            action_count: num_bands,
            learning_rate: 0.001,
            gamma: 0.99,
            gae_lambda: 0.95,
            clip_epsilon: 0.2,
            entropy_coefficient: 0.01,
            value_loss_coefficient: 0.5,
            gradient_clip: 1.0,
            update_epochs: 4,
            seed: 42,
        };
        let checkpoint = std::path::Path::new("models/ppo/latest.json");
        let ppo_agent = match PpoAgent::load(checkpoint) {
            Ok(agent) if agent.is_compatible(ppo_config.state_size, ppo_config.action_count) => {
                agent
            }
            _ => PpoAgent::new(ppo_config),
        };
        let dl_config = GruConfig::new(10, 16, 1, num_bands, 8, 0.0);
        let predictor = GruActivityPredictor::load("models/dl/latest.json")
            .ok()
            .filter(|model| model.config() == &dl_config)
            .unwrap_or_else(|| GruActivityPredictor::new(dl_config, 42));
        let mut app = Self {
            environment,
            predictor,
            ppo_agent,
            auto_run: false,
            history: Vec::new(),
            activity_predictions: vec![0.5; num_bands],
            action_probabilities: vec![1.0 / num_bands as f32; num_bands],
            feature_history: vec![FeatureVector {
                values: [0.0; FEATURE_COUNT],
            }],
            reward_history: Vec::new(),
            hit_history: Vec::new(),
            scans_by_band: vec![0; num_bands],
            hits_by_band: vec![0; num_bands],
            last_scan_time_by_band: vec![None; num_bands],
            recent_events: Vec::new(),
            exploration_reserve: ExplorationReserve::new(0.1),
            scan_decisions: 0,
            view: DashboardView::Overview,
            last_auto_step: Instant::now(),
            metrics: LiveMetrics::default(),
            last_result: None,
        };
        for _ in 0..30 {
            app.step();
        }
        app
    }

    fn reset(&mut self) {
        self.environment.reset();
        self.auto_run = false;
        self.history.clear();
        self.activity_predictions.fill(0.5);
        self.action_probabilities
            .fill(1.0 / self.environment.num_actions() as f32);
        self.feature_history = vec![FeatureVector {
            values: [0.0; FEATURE_COUNT],
        }];
        self.reward_history.clear();
        self.hit_history.clear();
        self.scans_by_band.fill(0);
        self.hits_by_band.fill(0);
        self.last_scan_time_by_band.fill(None);
        self.recent_events.clear();
        self.scan_decisions = 0;
        self.last_auto_step = Instant::now();
        self.metrics = LiveMetrics::default();
        self.last_result = None;
    }

    fn step(&mut self) {
        if self.environment.current_time() >= self.environment.time_horizon() {
            self.auto_run = false;
            return;
        }
        let time = self.environment.current_time();
        self.activity_predictions = self.predictor.predict(&self.feature_history);
        self.environment
            .set_predicted_activity(self.activity_predictions.clone())
            .expect("GRU output matches the 30-band PPO action space");
        let action_probabilities = self
            .ppo_agent
            .action_probabilities(&self.environment.observation().to_vector());
        let policy_band = action_probabilities
            .iter()
            .enumerate()
            .max_by(|(_, left), (_, right)| left.total_cmp(right))
            .map(|(band, _)| band)
            .expect("PPO has a non-empty action space");
        let exploration_band = self
            .scans_by_band
            .iter()
            .enumerate()
            .min_by_key(|(_, scans)| *scans)
            .map(|(band, _)| band)
            .expect("PPO has a non-empty action space");
        let selected_band = if let Some(unvisited_band) =
            self.scans_by_band.iter().position(|scans| *scans == 0)
        {
            unvisited_band
        } else {
            self.exploration_reserve
                .select_band(self.scan_decisions, policy_band, exploration_band)
        };
        let truth = (0..self.environment.num_actions())
            .map(|band| self.environment.ground_truth_at(time, band))
            .collect();
        let transition = self
            .environment
            .step(selected_band)
            .expect("PPO selected a valid action before episode completion");
        let next_feature = Self::feature_from_observation(&transition.observation);
        let result = transition.result;
        self.action_probabilities = action_probabilities;
        self.feature_history.push(next_feature);
        if self.feature_history.len() > 8 {
            self.feature_history.remove(0);
        }
        Self::push_chart_value(&mut self.reward_history, result.reward);
        Self::push_chart_value(
            &mut self.hit_history,
            if result.info.true_positives > 0 {
                1.0
            } else {
                0.0
            },
        );
        self.scans_by_band[selected_band] += 1;
        self.scan_decisions += 1;
        self.last_scan_time_by_band[selected_band] = Some(time);
        if result.info.true_positives > 0 {
            self.hits_by_band[selected_band] += 1;
        }
        self.recent_events.push(ScanEvent {
            time,
            band: selected_band,
            hit: result.info.true_positives > 0,
            reward: result.reward,
        });
        if self.recent_events.len() > 24 {
            self.recent_events.remove(0);
        }
        self.metrics.record(&result);
        self.history.push(HeatmapRow {
            truth,
            observation: result.observation.clone(),
            selected_band,
        });
        if self.history.len() > MAX_HISTORY_ROWS {
            self.history.remove(0);
        }
        if result.done {
            self.auto_run = false;
        }
        self.last_result = Some(result);
    }

    fn push_chart_value(values: &mut Vec<f32>, value: f32) {
        values.push(value);
        if values.len() > MAX_CHART_SAMPLES {
            values.remove(0);
        }
    }

    fn feature_from_observation(observation: &crate::ppo::PpoObservation) -> FeatureVector {
        let average = |values: &[f32]| values.iter().sum::<f32>() / values.len().max(1) as f32;
        let maximum = |values: &[f32]| values.iter().copied().fold(0.0_f32, f32::max);
        FeatureVector {
            values: [
                observation.normalized_time,
                observation.previous_reward,
                average(&observation.time_since_scan),
                average(&observation.time_since_detection),
                average(&observation.observed_activity_rate),
                average(&observation.exploration_score),
                maximum(&observation.observed_activity_rate),
                maximum(&observation.exploration_score),
                observation.previous_action.unwrap_or(0) as f32
                    / observation.predicted_activity.len().max(1) as f32,
                average(&observation.predicted_activity),
            ],
        }
    }

    fn draw_heatmap(&self, ui: &mut egui::Ui, height: f32) {
        let (response, painter) = ui.allocate_painter(
            Vec2::new(ui.available_width(), height),
            egui::Sense::hover(),
        );
        let rect = response.rect;
        painter.rect_filled(rect, 4.0, Color32::from_rgb(246, 249, 253));
        let label_width = 46.0;
        let plot_rect = Rect::from_min_max(
            Pos2::new(rect.left() + label_width, rect.top()),
            rect.right_bottom(),
        );
        let time_columns = self.history.len().max(1);
        let cell_width = plot_rect.width() / time_columns as f32;
        let cell_height = plot_rect.height() / self.environment.num_actions() as f32;
        for band in 0..self.environment.num_actions() {
            if band % 5 == 0 || band + 1 == self.environment.num_actions() {
                painter.text(
                    Pos2::new(
                        rect.left() + 2.0,
                        plot_rect.bottom() - (band + 1) as f32 * cell_height,
                    ),
                    Align2::LEFT_TOP,
                    format!("B{band:02}"),
                    FontId::monospace(10.0),
                    Color32::from_rgb(72, 85, 104),
                );
            }
        }
        for (time_index, row) in self.history.iter().enumerate() {
            for band in 0..self.environment.num_actions() {
                let cell = Rect::from_min_size(
                    Pos2::new(
                        plot_rect.left() + time_index as f32 * cell_width,
                        plot_rect.bottom() - (band + 1) as f32 * cell_height,
                    ),
                    Vec2::new(cell_width, cell_height),
                );
                let color = if row.observation[band] {
                    Color32::from_rgb(25, 174, 109)
                } else if row.truth[band] {
                    Color32::from_rgb(255, 122, 65)
                } else {
                    Color32::from_rgb(222, 231, 241)
                };
                painter.rect_filled(cell, 0.0, color);
                if band == row.selected_band {
                    painter.rect_stroke(
                        cell,
                        0.0,
                        Stroke::new(1.0_f32, Color32::from_rgb(28, 91, 214)),
                        StrokeKind::Inside,
                    );
                }
            }
        }
        ui.small("Rows: Band 00 to Band 29. Time moves left to right. Orange: signal active. Green: signal found. Blue outline: selected scan.");
    }

    fn draw_bar_chart(&self, ui: &mut egui::Ui) {
        ui.heading("PPO Band Priorities");
        let (response, painter) =
            ui.allocate_painter(Vec2::new(ui.available_width(), 150.0), egui::Sense::hover());
        let rect = response.rect;
        painter.rect_filled(rect, 4.0, Color32::from_rgb(246, 249, 253));
        let peak = self
            .action_probabilities
            .iter()
            .copied()
            .fold(0.0_f32, f32::max)
            .max(1e-6);
        let bar_width = rect.width() / self.action_probabilities.len() as f32;
        for (band, probability) in self.action_probabilities.iter().enumerate() {
            let height = rect.height() * (*probability / peak);
            let bar = Rect::from_min_size(
                Pos2::new(
                    rect.left() + band as f32 * bar_width + 1.0,
                    rect.bottom() - height,
                ),
                Vec2::new((bar_width - 2.0).max(1.0), height),
            );
            let color = if self
                .last_result
                .as_ref()
                .is_some_and(|result| result.selected_band == band)
            {
                Color32::from_rgb(25, 174, 109)
            } else {
                Color32::from_rgb(43, 105, 214)
            };
            painter.rect_filled(bar, 0.0, color);
        }
        ui.label("Taller bars are bands PPO currently prefers. Green is the chosen band.");
    }

    fn draw_prediction_chart(&self, ui: &mut egui::Ui) {
        ui.heading("Activity Forecast");
        let (response, painter) =
            ui.allocate_painter(Vec2::new(ui.available_width(), 150.0), egui::Sense::hover());
        let rect = response.rect;
        painter.rect_filled(rect, 4.0, Color32::from_rgb(246, 249, 253));
        let bar_width = rect.width() / self.activity_predictions.len() as f32;
        for (band, probability) in self.activity_predictions.iter().enumerate() {
            let height = rect.height() * probability;
            let bar = Rect::from_min_size(
                Pos2::new(
                    rect.left() + band as f32 * bar_width + 1.0,
                    rect.bottom() - height,
                ),
                Vec2::new((bar_width - 2.0).max(1.0), height),
            );
            painter.rect_filled(bar, 0.0, Color32::from_rgb(255, 122, 65));
        }
        ui.label("Taller bars indicate bands the temporal predictor expects to become active.");
    }

    fn draw_line_chart(&self, ui: &mut egui::Ui, title: &str, values: &[f32], color: Color32) {
        ui.heading(title);
        let (response, painter) =
            ui.allocate_painter(Vec2::new(ui.available_width(), 150.0), egui::Sense::hover());
        let rect = response.rect;
        painter.rect_filled(rect, 4.0, Color32::from_rgb(246, 249, 253));
        if values.len() < 2 {
            return;
        }
        let minimum = values.iter().copied().fold(f32::INFINITY, f32::min);
        let maximum = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let range = (maximum - minimum).max(1e-4);
        let points = values
            .iter()
            .enumerate()
            .map(|(index, value)| {
                Pos2::new(
                    rect.left() + index as f32 * rect.width() / (values.len() - 1) as f32,
                    rect.bottom() - ((*value - minimum) / range) * rect.height(),
                )
            })
            .collect::<Vec<_>>();
        painter.add(egui::Shape::line(points, Stroke::new(1.5_f32, color)));
    }

    fn draw_metrics(&self, ui: &mut egui::Ui) {
        let detection_probability =
            LiveMetrics::ratio(self.metrics.true_positives, self.metrics.active_bands);
        let false_alarm_probability = LiveMetrics::ratio(
            self.metrics.false_positives,
            self.metrics.monitored_inactive_bands,
        );
        let intercept_rate =
            LiveMetrics::ratio(self.metrics.interceptions, self.metrics.transmission_starts);
        let intercept_time =
            LiveMetrics::ratio(self.metrics.intercept_delay_sum, self.metrics.interceptions);
        egui::Grid::new("live-metrics")
            .num_columns(2)
            .show(ui, |ui| {
                ui.label("Cumulative reward");
                ui.label(format!("{:.3}", self.metrics.total_reward));
                ui.end_row();
                ui.label("Interceptions");
                ui.label(self.metrics.interceptions.to_string());
                ui.end_row();
                ui.label("Average intercept time");
                ui.label(format!("{intercept_time:.3}"));
                ui.end_row();
                ui.label("Detection rate");
                ui.label(format!("{detection_probability:.3}"));
                ui.end_row();
                ui.label("False alarm rate");
                ui.label(format!("{false_alarm_probability:.3}"));
                ui.end_row();
                ui.label("Window intercept rate");
                ui.label(format!("{intercept_rate:.3}"));
                ui.end_row();
            });
    }

    fn draw_decision_strip(&self, ui: &mut egui::Ui) {
        let forecast_band = self
            .activity_predictions
            .iter()
            .enumerate()
            .max_by(|(_, left), (_, right)| left.total_cmp(right))
            .map(|(band, _)| band)
            .unwrap_or(0);
        let forecast_confidence = self.activity_predictions[forecast_band];
        let (selected_band, result_label, result_color) = match &self.last_result {
            Some(result) if result.info.true_positives > 0 => (
                result.selected_band,
                "SIGNAL FOUND",
                Color32::from_rgb(77, 213, 153),
            ),
            Some(result) => (
                result.selected_band,
                "NO SIGNAL FOUND",
                Color32::from_rgb(242, 178, 65),
            ),
            None => (0, "READY TO SCAN", Color32::from_rgb(71, 138, 196)),
        };
        egui::Frame::NONE
            .fill(Color32::from_rgb(239, 245, 255))
            .inner_margin(egui::Margin::same(10))
            .show(ui, |ui| {
                ui.columns(4, |columns| {
                    columns[0].heading("1. FORECAST");
                    columns[0].label(format!("Watch band {forecast_band}"));
                    columns[0].label(format!("Confidence {:.0}%", forecast_confidence * 100.0));
                    columns[1].heading("2. CHOICE");
                    columns[1].label(format!("Scan band {selected_band}"));
                    columns[1].label("PPO selected this band");
                    columns[2].heading("3. RECEIVER");
                    columns[2].label(result_label);
                    columns[2].colored_label(result_color, "Latest scan result");
                    columns[3].heading("4. LEARNING");
                    columns[3].label("Result feeds the next choice");
                    columns[3].label("Coverage updates live");
                });
            });
    }

    fn draw_scoreboard(&self, ui: &mut egui::Ui) {
        let detection_rate =
            LiveMetrics::ratio(self.metrics.true_positives, self.metrics.active_bands);
        let false_alarm_rate = LiveMetrics::ratio(
            self.metrics.false_positives,
            self.metrics.monitored_inactive_bands,
        );
        let intercept_rate =
            LiveMetrics::ratio(self.metrics.interceptions, self.metrics.transmission_starts);
        let average_delay =
            LiveMetrics::ratio(self.metrics.intercept_delay_sum, self.metrics.interceptions);
        let latest_hit = self
            .last_result
            .as_ref()
            .is_some_and(|result| result.info.true_positives > 0);
        let coverage = LiveMetrics::ratio(
            self.scans_by_band
                .iter()
                .filter(|scans| **scans > 0)
                .count(),
            self.environment.num_actions(),
        );
        let cards = [
            (
                "Latest Scan",
                if latest_hit {
                    "HIT".to_string()
                } else {
                    "WAITING".to_string()
                },
            ),
            ("Signals Found", self.metrics.interceptions.to_string()),
            ("Detection Rate", format!("{:.0}%", detection_rate * 100.0)),
            ("False Alarms", format!("{:.1}%", false_alarm_rate * 100.0)),
            ("Average Delay", format!("{average_delay:.2} slots")),
            ("Total Reward", format!("{:.1}", self.metrics.total_reward)),
            ("Bands Checked", format!("{:.0}%", coverage * 100.0)),
        ];
        egui::Grid::new("scoreboard")
            .num_columns(7)
            .spacing([12.0, 4.0])
            .show(ui, |ui| {
                for (label, value) in cards {
                    ui.vertical(|ui| {
                        ui.small(label);
                        ui.heading(value);
                    });
                }
                ui.end_row();
            });
        ui.small(format!("Windows caught: {:.0}%", intercept_rate * 100.0));
        ui.small("Detection rate counts active band-time slots seen. Windows caught counts distinct transmissions intercepted at least once. Every tenth scan explores the least-checked band.");
    }

    fn draw_band_report(&self, ui: &mut egui::Ui) {
        ui.heading("All 30 Receiver Bands");
        let mut ranked_bands = (0..self.environment.num_actions()).collect::<Vec<_>>();
        ranked_bands.sort_by(|left, right| {
            let left_ratio =
                LiveMetrics::ratio(self.hits_by_band[*left], self.scans_by_band[*left]);
            let right_ratio =
                LiveMetrics::ratio(self.hits_by_band[*right], self.scans_by_band[*right]);
            right_ratio.total_cmp(&left_ratio)
        });
        let best = ranked_bands.first().copied().unwrap_or(0);
        let least = ranked_bands
            .iter()
            .copied()
            .filter(|band| self.scans_by_band[*band] > 0)
            .min_by(|left, right| {
                let left_ratio =
                    LiveMetrics::ratio(self.hits_by_band[*left], self.scans_by_band[*left]);
                let right_ratio =
                    LiveMetrics::ratio(self.hits_by_band[*right], self.scans_by_band[*right]);
                left_ratio.total_cmp(&right_ratio)
            })
            .unwrap_or(0);
        ui.horizontal(|ui| {
            ui.colored_label(
                Color32::from_rgb(77, 213, 153),
                format!("Most successful: Band {best}"),
            );
            ui.separator();
            ui.colored_label(
                Color32::from_rgb(242, 178, 65),
                format!("Needs more evidence: Band {least}"),
            );
        });
        egui::Grid::new("band-report-header")
            .num_columns(5)
            .show(ui, |ui| {
                ui.strong("Receiver band");
                ui.strong("Scans");
                ui.strong("Signals found");
                ui.strong("Hit ratio");
                ui.strong("Last checked");
                ui.end_row();
            });
        egui::ScrollArea::vertical()
            .id_salt("all-band-report")
            .max_height(500.0)
            .show(ui, |ui| {
                egui::Grid::new("band-report-rows")
                    .num_columns(5)
                    .min_col_width(90.0)
                    .striped(true)
                    .show(ui, |ui| {
                        for band in 0..self.environment.num_actions() {
                            let ratio = LiveMetrics::ratio(
                                self.hits_by_band[band],
                                self.scans_by_band[band],
                            );
                            ui.label(format!("Band {band:02}"));
                            ui.label(self.scans_by_band[band].to_string());
                            ui.label(self.hits_by_band[band].to_string());
                            ui.colored_label(
                                if ratio > 0.0 {
                                    Color32::from_rgb(77, 213, 153)
                                } else {
                                    Color32::GRAY
                                },
                                format!("{:.0}%", ratio * 100.0),
                            );
                            ui.label(
                                self.last_scan_time_by_band[band]
                                    .map(|time| time.to_string())
                                    .unwrap_or_else(|| "-".to_string()),
                            );
                            ui.end_row();
                        }
                    });
            });
    }

    fn draw_recent_events(&self, ui: &mut egui::Ui) {
        ui.heading("Latest Scan Decisions");
        egui::ScrollArea::vertical()
            .id_salt("recent-events-scroll")
            .max_height(210.0)
            .show(ui, |ui| {
                for event in self.recent_events.iter().rev() {
                    let label = if event.hit { "FOUND" } else { "NO SIGNAL" };
                    let color = if event.hit {
                        Color32::from_rgb(77, 213, 153)
                    } else {
                        Color32::from_rgb(242, 178, 65)
                    };
                    ui.horizontal(|ui| {
                        ui.label(format!("Time {}", event.time));
                        ui.label(format!("Band {}", event.band));
                        ui.colored_label(color, label);
                        ui.label(format!("Reward {:.2}", event.reward));
                    });
                }
            });
    }

    fn draw_overview(&self, ui: &mut egui::Ui) {
        self.draw_decision_strip(ui);
        ui.add_space(6.0);
        self.draw_scoreboard(ui);
        ui.separator();
        ui.heading("30-Band Spectrum: Recent Activity");
        self.draw_heatmap(ui, 250.0);
        ui.columns(2, |columns| {
            self.draw_prediction_chart(&mut columns[0]);
            self.draw_bar_chart(&mut columns[1]);
        });
    }

    fn draw_spectrum(&self, ui: &mut egui::Ui) {
        ui.heading("30-Band Spectrum Over Time");
        self.draw_heatmap(ui, 520.0);
        ui.columns(2, |columns| {
            self.draw_prediction_chart(&mut columns[0]);
            self.draw_bar_chart(&mut columns[1]);
        });
    }

    fn draw_band_reports(&self, ui: &mut egui::Ui) {
        ui.columns(2, |columns| {
            self.draw_band_report(&mut columns[0]);
            self.draw_recent_events(&mut columns[1]);
        });
    }

    fn draw_activity(&self, ui: &mut egui::Ui) {
        ui.columns(2, |columns| {
            self.draw_line_chart(
                &mut columns[0],
                "Reward From Recent Scans",
                &self.reward_history,
                Color32::from_rgb(242, 178, 65),
            );
            self.draw_line_chart(
                &mut columns[1],
                "Recent Scan Results",
                &self.hit_history,
                Color32::from_rgb(25, 174, 109),
            );
        });
        ui.separator();
        self.draw_scoreboard(ui);
        self.draw_recent_events(ui);
    }
}

impl Default for GuiApp {
    fn default() -> Self {
        Self::new()
    }
}

impl eframe::App for GuiApp {
    fn update(&mut self, context: &egui::Context, _frame: &mut eframe::Frame) {
        if context.style().visuals.dark_mode {
            context.set_visuals(egui::Visuals::light());
        }
        if self.auto_run && self.last_auto_step.elapsed() >= AUTO_STEP_INTERVAL {
            self.step();
            self.last_auto_step = Instant::now();
        }
        context.request_repaint_after(Duration::from_millis(100));
        egui::TopBottomPanel::top("controls").show(context, |ui| {
            ui.horizontal(|ui| {
                ui.heading("SMARTSCAN");
                ui.separator();
                if ui
                    .selectable_label(self.view == DashboardView::Overview, "Overview")
                    .clicked()
                {
                    self.view = DashboardView::Overview;
                }
                if ui
                    .selectable_label(self.view == DashboardView::Spectrum, "Spectrum")
                    .clicked()
                {
                    self.view = DashboardView::Spectrum;
                }
                if ui
                    .selectable_label(self.view == DashboardView::BandReport, "Band Report")
                    .clicked()
                {
                    self.view = DashboardView::BandReport;
                }
                if ui
                    .selectable_label(self.view == DashboardView::Activity, "Activity")
                    .clicked()
                {
                    self.view = DashboardView::Activity;
                }
                ui.separator();
                if ui
                    .button(if self.auto_run { "Pause" } else { "Run" })
                    .clicked()
                {
                    self.auto_run = !self.auto_run;
                }
                if ui.button("Step").clicked() {
                    self.step();
                }
                if ui.button("Reset").clicked() {
                    self.reset();
                }
                ui.separator();
                ui.label(format!(
                    "Time: {}/{}",
                    self.environment.current_time(),
                    self.environment.time_horizon()
                ));
                ui.label("Adaptive scheduler: temporal forecast + PPO policy");
            });
        });
        egui::SidePanel::right("state")
            .min_width(290.0)
            .show(context, |ui| {
                ui.heading("Live Receiver");
                if let Some(result) = &self.last_result {
                    ui.label(format!("Listening on band {}", result.selected_band));
                    ui.label(format!("Latest reward: {:.3}", result.reward));
                    ui.label(if result.info.true_positives > 0 {
                        "Detection: HIT"
                    } else {
                        "Detection: MISS"
                    });
                    ui.label(format!("Signals active now: {}", result.info.active_bands));
                } else {
                    ui.label("No scan executed");
                }
                ui.separator();
                ui.heading("Performance So Far");
                self.draw_metrics(ui);
                ui.separator();
                ui.heading("System Ready");
                ui.label("Activity forecast updated before every scan");
                ui.label("PPO chooses the next scan band");
                ui.label("Live results update after each receiver scan");
            });
        egui::CentralPanel::default().show(context, |ui| match self.view {
            DashboardView::Overview => self.draw_overview(ui),
            DashboardView::Spectrum => self.draw_spectrum(ui),
            DashboardView::BandReport => self.draw_band_reports(ui),
            DashboardView::Activity => self.draw_activity(ui),
        });
    }
}

pub fn launch() -> eframe::Result<()> {
    eframe::run_native(
        "SMARTSCAN",
        eframe::NativeOptions {
            viewport: egui::ViewportBuilder::default().with_inner_size([1200.0, 760.0]),
            ..Default::default()
        },
        Box::new(|_creation_context| Ok(Box::new(GuiApp::new()))),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn initial_sweep_checks_each_receiver_band_once() {
        let app = GuiApp::new();
        assert_eq!(app.scans_by_band.len(), 30);
        assert!(app.scans_by_band.iter().all(|scans| *scans == 1));
    }
}
