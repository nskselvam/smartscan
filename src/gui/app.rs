use eframe::egui::{self, Color32, Pos2, Rect, Stroke, StrokeKind, Vec2};

use crate::scheduler::{RoundRobinScheduler, Scheduler};
use crate::simulator::{RfEnvironment, Scenario, StepResult};

const HEATMAP_HEIGHT: f32 = 280.0;
const MAX_HISTORY_ROWS: usize = 80;

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
    environment: RfEnvironment,
    scheduler: RoundRobinScheduler,
    auto_run: bool,
    history: Vec<HeatmapRow>,
    metrics: LiveMetrics,
    last_result: Option<StepResult>,
}

impl GuiApp {
    pub fn new() -> Self {
        let scenario = Scenario::mixed();
        let num_bands = scenario.num_bands;
        Self {
            environment: scenario.into_environment(),
            scheduler: RoundRobinScheduler::new(num_bands),
            auto_run: false,
            history: Vec::new(),
            metrics: LiveMetrics::default(),
            last_result: None,
        }
    }

    fn reset(&mut self) {
        self.environment.reset();
        self.scheduler.reset();
        self.auto_run = false;
        self.history.clear();
        self.metrics = LiveMetrics::default();
        self.last_result = None;
    }

    fn step(&mut self) {
        if self.environment.current_time >= self.environment.time_horizon {
            self.auto_run = false;
            return;
        }
        let time = self.environment.current_time;
        let selected_band = self.scheduler.select_band();
        let truth = (0..self.environment.num_bands)
            .map(|band| self.environment.ground_truth.is_transmitting(time, band))
            .collect();
        let result = self.environment.step(selected_band);
        self.scheduler.update(result.reward);
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

    fn draw_heatmap(&self, ui: &mut egui::Ui) {
        let (response, painter) = ui.allocate_painter(
            Vec2::new(ui.available_width(), HEATMAP_HEIGHT),
            egui::Sense::hover(),
        );
        let rect = response.rect;
        painter.rect_filled(rect, 2.0, Color32::from_rgb(20, 28, 34));
        let rows = self.history.len().max(1);
        let cell_width = rect.width() / self.environment.num_bands as f32;
        let cell_height = rect.height() / rows as f32;
        for (row_index, row) in self.history.iter().enumerate() {
            for band in 0..self.environment.num_bands {
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
                ui.label("Pd");
                ui.label(format!("{detection_probability:.3}"));
                ui.end_row();
                ui.label("Pfa");
                ui.label(format!("{false_alarm_probability:.3}"));
                ui.end_row();
                ui.label("Intercept rate");
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
                    self.environment.current_time, self.environment.time_horizon
                ));
                ui.label(format!("Scheduler: {}", self.scheduler.name()));
            });
        });
        egui::SidePanel::right("state")
            .min_width(250.0)
            .show(context, |ui| {
                ui.heading("Receiver State");
                if let Some(result) = &self.last_result {
                    ui.label(format!("Selected band: {}", result.selected_band));
                    ui.label(format!("Latest reward: {:.3}", result.reward));
                    ui.label(if result.info.true_positives > 0 {
                        "Hit"
                    } else {
                        "Miss"
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
                ui.label("DL prediction probabilities: NOT YET MEASURED");
                ui.label("PPO action probabilities: NOT YET MEASURED");
                ui.label("Training progress: NOT YET MEASURED");
            });
        egui::CentralPanel::default().show(context, |ui| {
            ui.heading("Frequency-Time Environment");
            self.draw_heatmap(ui);
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
