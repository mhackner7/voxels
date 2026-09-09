/*
IDEAS:

not allowing chording yet

keeping it simple is important
still keep it setup for variable keymapping though, even if it is not implemented

given that it is command -> enum its is simply storing an array
however enum -> command might require a search + verification check

hashmap with enum as the key and command as the value is the simplest
should i store key and command as values?
no, i can always go from command to key easily with array access



actually test the speed differences to gague whether the hashmap is actually needed
might just be a bunk addition

*/

use crate::engine::input::commands::*;
use crate::engine::input::util::*;

/// maps all commands to their corresponding KBMcode
/// stores all commands stored in order
pub struct KeyMap {
    /// used to get keycodes from commands
    pub keycodes_from_commands: [KBMcode; Command::MAX_COUNT],

    /// used to get commands from keycodes
    pub commands_from_keycodes: [Option<Command>; KBMcode::NULLorCOUNT as usize],
    // stores the characters that are bound to the command
    // useful because I am storing physical key location not printed character
    // pub keymap_characters: [Key; SIZE],
}

impl KeyMap {
    /// change the binding for *action* to *new_key*
    /// true on success, false on overlap / failure
    pub fn change(&mut self, command: Command, new_keycode: KBMcode) -> bool {
        let i = command.index();

        // return early if there would be an overlap conflict
        let old_keycode = self.keycodes_from_commands[i];

        if self.commands_from_keycodes[new_keycode as usize].is_none() {
            return false;
        }

        // change the keycode-storing array
        self.keycodes_from_commands[i] = new_keycode;

        // change the command-storing array
        self.commands_from_keycodes[new_keycode as usize] = Some(command);

        // null out the old keycode's associated command
        self.commands_from_keycodes[old_keycode as usize] = None;

        return true;
    }

    /// returns either a valid command if there is one for that keycode
    /// or else returns Command::Invalid, which is already baked into the array
    pub fn command_for(&self, keycode: KBMcode) -> Option<Command> {
        self.commands_from_keycodes[keycode as usize]
    }

    /// returns keycode for specified command, cannot return an invalid command
    pub fn keycode_for(&self, command: Command) -> KBMcode {
        let i = command.index();
        self.keycodes_from_commands[i]
    }

    /// check if the kbm code is stored
    /// useful for when youre trying to avoid key overlap between diff keymaps
    #[inline(always)]
    pub fn contains(&self, key: KBMcode) -> bool {
        self.commands_from_keycodes[key as usize].is_none()
    }

    /// yields the keycodes for the specified command type
    /// example: map.iter(Command::Entity(_))
    pub fn iter(&self, variant: Command) -> impl Iterator<Item = KBMcode> {
        let std::ops::Range { start, end } = match variant {
            Command::Player(_) => Command::ENTITY_RANGE,
            Command::Camera(_) => Command::CAMERA_RANGE,
            Command::Inventory(_) => Command::INVENTORY_RANGE,
            Command::Developer(_) => Command::DEVELOPER_RANGE,
            Command::NULL => panic!(
                "cant use null to access array, it is an invalid command and not counted in the total number"
            ),
        };

        self.keycodes_from_commands
            .into_iter()
            .skip(start)
            .take(end)
    }
}
