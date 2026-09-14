/// Native GUI using egui
///
/// Provides visualization for:
/// - Frequency-time heatmap
/// - Current receiver band
/// - Active emitter bands
/// - DL prediction probabilities
/// - PPO action probabilities
/// - Hit/miss indicator
/// - Metrics display
/// - Training progress
pub struct GuiApp;

impl GuiApp {
    pub fn new() -> Self {
        GuiApp
    }
}

impl Default for GuiApp {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gui_app_creation() {
        let _app = GuiApp::new();
    }
}
