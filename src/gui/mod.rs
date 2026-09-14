pub mod app;

pub use app::{launch, GuiApp};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gui_app_creation() {
        let _app = GuiApp::new();
    }
}
