use super::activity::Activity;
use super::keymap::KeyMap;
use super::util;
use super::util::KBMcode;

pub struct Handler {
    pub keymap: KeyMap,
    pub activity: Activity,
    pub mouse_sensitivity: f64,
    pub last_mouse_motion: (f64, f64),
}

impl Handler {
    /// push input to the activity queue
    /// if mouse motion, handle immediatley
    pub fn push(&mut self, input: KBMcode, state: util::State) {
        if let Some(cmd) = self.keymap.commands_from_keycodes[input as usize] {
            self.activity.push((cmd, state));
        }
    }

    /// process the keyqueue and mouse motion events
    pub fn process(&mut self, dt: f32) {
        // seconds f32 -> milliseconds u32
        self.activity.run((dt * 1000.0) as u16);
    }
}

impl Default for Handler {
    fn default() -> Self {
        Self {
            keymap: Default::default(),
            activity: Default::default(),
            mouse_sensitivity: 0.004,
            last_mouse_motion: (0.0, 0.0),
        }
    }
}
