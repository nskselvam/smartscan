use eframe::egui::{self, Color32, Pos2, Rect, Stroke, StrokeKind, Vec2};

use crate::ppo::{PpoAgent, PpoConfig, PpoEnvironment};
use crate::simulator::{Scenario, StepResult};

const MAX_HISTORY_ROWS: usize = 60;
const MAX_CHART_SAMPLES: usize = 120;

struct HeatmapRow {
    truth: Vec<bool>,
    observation: Vec<bool>,
    selected_band: usize,
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
    ppo_agent: PpoAgent,
    ppo_checkpoint_loaded: bool,
    auto_run: bool,
    history: Vec<HeatmapRow>,
    action_probabilities: Vec<f32>,
    reward_history: Vec<f32>,
    hit_history: Vec<f32>,
    metrics: LiveMetrics,
    last_result: Option<StepResult>,
}

impl GuiApp {
    pub fn new() -> Self {
        let scenario = Scenario::mixed().with_num_bands(30);
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
        let (ppo_agent, ppo_checkpoint_loaded) = match PpoAgent::load(checkpoint) {
            Ok(agent) if agent.is_compatible(ppo_config.state_size, ppo_config.action_count) => {
                (agent, true)
            }
            _ => (PpoAgent::new(ppo_config), false),
        };
        Self {
            environment,
            ppo_agent,
            ppo_checkpoint_loaded,
            auto_run: false,
            history: Vec::new(),
            action_probabilities: vec![1.0 / num_bands as f32; num_bands],
            reward_history: Vec::new(),
            hit_history: Vec::new(),
            metrics: LiveMetrics::default(),
            last_result: None,
        }
    }

    fn reset(&mut self) {
        self.environment.reset();
        self.auto_run = false;
        self.history.clear();
        self.action_probabilities.fill(1.0 / self.environment.num_actions() as f32);
        self.reward_history.clear();
        self.hit_history.clear();
        self.metrics = LiveMetrics::default();
        self.last_result = None;
    }

    fn step(&mut self) {
        if self.environment.current_time() >= self.environment.time_horizon() {
            self.auto_run = false;
            return;
        }
        let time = self.environment.current_time();
        let action_probabilities = self
            .ppo_agent
            .action_probabilities(&self.environment.observation().to_vector());
        let selected_band = action_probabilities
            .iter()
            .enumerate()
            .max_by(|(_, left), (_, right)| left.total_cmp(right))
            .map(|(band, _)| band)
            .expect("PPO has a non-empty action space");
        let truth = (0..self.environment.num_actions())
            .map(|band| self.environment.ground_truth_at(time, band))
            .collect();
        let transition = self
            .environment
            .step(selected_band)
            .expect("PPO selected a valid action before episode completion");
        let result = transition.result;
        self.action_probabilities = action_probabilities;
        Self::push_chart_value(&mut self.reward_history, result.reward);
        Self::push_chart_value(
            &mut self.hit_history,
            if result.info.true_positives > 0 { 1.0 } else { 0.0 },
        );
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

    fn draw_heatmap(&self, ui: &mut egui::Ui) {
        let (response, painter) = ui.allocate_painter(
            Vec2::new(ui.available_width(), (ui.available_height() * 0.62).clamp(280.0, 430.0)),
            egui::Sense::hover(),
        );
        let rect = response.rect;
        painter.rect_filled(rect, 2.0, Color32::from_rgb(20, 28, 34));
        let rows = self.history.len().max(1);
        let cell_width = rect.width() / self.environment.num_actions() as f32;
        let cell_height = rect.height() / rows as f32;
        for (row_index, row) in self.history.iter().enumerate() {
            for band in 0..self.environment.num_actions() {
                let cell = Rect::from_min_size(
                    Pos2::new(
                        rect.left() + band as f32 * cell_width,
                        rect.top() + row_index as f32 * cell_height,
                    ),
                    Vec2::new(cell_width, cell_height),
                );
                let color = if row.observation[band] {
                    Color32::from_rgb(84, 220, 135)
                } else if row.truth[band] {
                    Color32::from_rgb(224, 116, 74)
                } else {
                    Color32::from_rgb(32, 47, 53)
                };
                painter.rect_filled(cell, 0.0, color);
                if band == row.selected_band {
                    painter.rect_stroke(
                        cell,
                        0.0,
                        Stroke::new(1.0_f32, Color32::WHITE),
                        StrokeKind::Inside,
                    );
                }
            }
        }
        ui.label("Orange: active ground truth. Green: receiver report. White outline: selected scan band.");
    }

    fn draw_bar_chart(&self, ui: &mut egui::Ui) {
        ui.heading("PPO Band Priorities");
        let (response, painter) = ui.allocate_painter(
            Vec2::new(ui.available_width(), 150.0),
            egui::Sense::hover(),
        );
        let rect = response.rect;
        painter.rect_filled(rect, 2.0, Color32::from_rgb(20, 28, 34));
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
                Pos2::new(rect.left() + band as f32 * bar_width + 1.0, rect.bottom() - height),
                Vec2::new((bar_width - 2.0).max(1.0), height),
            );
            let color = if self.last_result.as_ref().is_some_and(|result| result.selected_band == band) {
                Color32::from_rgb(77, 213, 153)
            } else {
                Color32::from_rgb(71, 138, 196)
            };
            painter.rect_filled(bar, 0.0, color);
        }
        ui.label("Taller bars are bands PPO currently prefers. Green is the chosen band.");
    }

    fn draw_line_chart(&self, ui: &mut egui::Ui, title: &str, values: &[f32], color: Color32) {
        ui.heading(title);
        let (response, painter) = ui.allocate_painter(
            Vec2::new(ui.available_width(), 150.0),
            egui::Sense::hover(),
        );
        let rect = response.rect;
        painter.rect_filled(rect, 2.0, Color32::from_rgb(20, 28, 34));
        if values.len() < 2 {
            return;
        }
        let minimum = values.iter().copied().fold(f32::INFINITY, f32::min);
        let maximum = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let range = (maximum - minimum).max(1e-4);
        let points = values
            .iter()
            .enumerate()
            .map(|(index, value)| Pos2::new(
                rect.left() + index as f32 * rect.width() / (values.len() - 1) as f32,
                rect.bottom() - ((*value - minimum) / range) * rect.height(),
            ))
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
}

impl Default for GuiApp {
    fn default() -> Self {
        Self::new()
    }
}

impl eframe::App for GuiApp {
    fn update(&mut self, context: &egui::Context, _frame: &mut eframe::Frame) {
        if self.auto_run {
            self.step();
            context.request_repaint();
        }
        egui::TopBottomPanel::top("controls").show(context, |ui| {
            ui.horizontal(|ui| {
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
                ui.label(if self.ppo_checkpoint_loaded {
                    "Scheduler: PPO policy (trained checkpoint)"
                } else {
                    "Scheduler: PPO policy (untrained)"
                });
            });
        });
        egui::SidePanel::right("state")
            .min_width(290.0)
            .show(context, |ui| {
                ui.heading("Receiver State");
                if let Some(result) = &self.last_result {
                    ui.label(format!("Selected band: {}", result.selected_band));
                    ui.label(format!("Latest reward: {:.3}", result.reward));
                    ui.label(if result.info.true_positives > 0 {
                        "Detection: HIT"
                    } else {
                        "Detection: MISS"
                    });
                    ui.label(format!("Active bands: {}", result.info.active_bands));
                } else {
                    ui.label("No scan executed");
                }
                ui.separator();
                ui.heading("Measured Metrics");
                self.draw_metrics(ui);
                ui.separator();
                ui.heading("Model Status");
                ui.label("DL predictor: not trained / not loaded");
                ui.label(if self.ppo_checkpoint_loaded {
                    "PPO action probabilities: trained policy output"
                } else {
                    "PPO action probabilities: untrained policy output"
                });
                ui.label("PPO training: checkpoint loaded");
            });
        egui::CentralPanel::default().show(context, |ui| {
            ui.heading("30-Band Frequency-Time Environment");
            self.draw_heatmap(ui);
            ui.columns(2, |columns| {
                self.draw_bar_chart(&mut columns[0]);
                self.draw_line_chart(
                    &mut columns[1],
                    "Recent Reward",
                    &self.reward_history,
                    Color32::from_rgb(242, 178, 65),
                );
            });
            self.draw_line_chart(
                ui,
                "Detection History (1 = hit, 0 = miss)",
                &self.hit_history,
                Color32::from_rgb(77, 213, 153),
            );
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
