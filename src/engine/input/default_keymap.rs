use crate::engine::input::commands::*;
use crate::engine::input::util::KBMcode;

impl std::default::Default for crate::engine::input::keymap::KeyMap {
    fn default() -> Self {
        let mut keycodes_from_commands = [KBMcode::NULLorCOUNT; Command::MAX_COUNT];
        let mut i: usize;

        // Entity Commands
        i = Command::Player(PlayerCommands::MoveForward).index();
        keycodes_from_commands[i] = KBMcode::KeyW;
        i = Command::Player(PlayerCommands::MoveBackward).index();
        keycodes_from_commands[i] = KBMcode::KeyS;
        i = Command::Player(PlayerCommands::MoveLeft).index();
        keycodes_from_commands[i] = KBMcode::KeyA;
        i = Command::Player(PlayerCommands::MoveRight).index();
        keycodes_from_commands[i] = KBMcode::KeyD;
        i = Command::Player(PlayerCommands::Sprint).index();
        keycodes_from_commands[i] = KBMcode::ShiftLeft;
        i = Command::Player(PlayerCommands::Jump).index();
        keycodes_from_commands[i] = KBMcode::Space;
        i = Command::Player(PlayerCommands::Sneak).index();
        keycodes_from_commands[i] = KBMcode::KeyC;
        i = Command::Player(PlayerCommands::Destroy).index();
        keycodes_from_commands[i] = KBMcode::MouseLeft;
        i = Command::Player(PlayerCommands::Interact).index();
        keycodes_from_commands[i] = KBMcode::MouseRight;
        i = Command::Player(PlayerCommands::Select).index();
        keycodes_from_commands[i] = KBMcode::MouseMiddle;
        i = Command::Player(PlayerCommands::Inventory).index();
        keycodes_from_commands[i] = KBMcode::KeyE;
        i = Command::Player(PlayerCommands::BrushGrow).index();
        keycodes_from_commands[i] = KBMcode::MouseScrollUp;
        i = Command::Player(PlayerCommands::BrushShrink).index();
        keycodes_from_commands[i] = KBMcode::MouseScrollDown;

        // Camera Commands
        i = Command::Camera(CameraCommands::ZoomIn).index();
        keycodes_from_commands[i] = KBMcode::Equal;
        i = Command::Camera(CameraCommands::ZoomOut).index();
        keycodes_from_commands[i] = KBMcode::Minus;
        i = Command::Camera(CameraCommands::ChangePerspective).index();
        keycodes_from_commands[i] = KBMcode::F5;

        // Inventory Commands
        // (not on MouseLeft, a key only maps to one command and that one breaks stuff)
        i = Command::Inventory(InventoryCommands::Select).index();
        keycodes_from_commands[i] = KBMcode::Enter;
        i = Command::Inventory(InventoryCommands::Next).index();
        keycodes_from_commands[i] = KBMcode::ArrowRight;
        i = Command::Inventory(InventoryCommands::Back).index();
        keycodes_from_commands[i] = KBMcode::ArrowLeft;
        i = Command::Inventory(InventoryCommands::Split).index();
        keycodes_from_commands[i] = KBMcode::Digit8;
        i = Command::Inventory(InventoryCommands::QuickMove).index();
        keycodes_from_commands[i] = KBMcode::Digit9;

        // Developer Commands
        i = Command::Developer(DeveloperCommands::Pause).index();
        keycodes_from_commands[i] = KBMcode::Escape;
        i = Command::Developer(DeveloperCommands::Back).index();
        keycodes_from_commands[i] = KBMcode::Escape;
        i = Command::Developer(DeveloperCommands::VolumeUp).index();
        keycodes_from_commands[i] = KBMcode::AudioVolumeUp;
        i = Command::Developer(DeveloperCommands::VolumeDown).index();
        keycodes_from_commands[i] = KBMcode::AudioVolumeDown;
        i = Command::Developer(DeveloperCommands::OpenDevMenu).index();
        keycodes_from_commands[i] = KBMcode::Digit3;
        i = Command::Developer(DeveloperCommands::ShowDebugInfoLight).index();
        keycodes_from_commands[i] = KBMcode::Digit1;
        i = Command::Developer(DeveloperCommands::ShowDebugInfoHeavy).index();
        keycodes_from_commands[i] = KBMcode::Digit2;
        i = Command::Developer(DeveloperCommands::ForceMinimize).index();
        keycodes_from_commands[i] = KBMcode::F12;
        i = Command::Developer(DeveloperCommands::ToggleFullscreen).index();
        keycodes_from_commands[i] = KBMcode::Digit0;

        let mut commands_from_keycodes = [None; KBMcode::NULLorCOUNT as usize];

        for i in 0..Command::MAX_COUNT as usize {
            let keycode = keycodes_from_commands[i];
            commands_from_keycodes[keycode as usize] = Some(Command::LUT[i]);
        }

        for i in 0..Command::MAX_COUNT {
            assert!(
                keycodes_from_commands[i] != KBMcode::NULLorCOUNT,
                "invalid default keymap for index {}: {}",
                i,
                Command::LUT[i]
            );
        }

        Self {
            keycodes_from_commands,
            commands_from_keycodes,
        }
    }
}
