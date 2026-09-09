use std::fmt::Display;

use crate::engine::input::commands::Command;
use crate::engine::input::util::State;

/// CONTAINS INPUT QUEUE AND ACTIVITY LOGGER
pub struct Activity {
    /// keys currently held, and their duration
    held: [(Command, u16); SIZE],
    held_count: usize,

    /// the keys that have been released this frame
    /// includes the duration it was held,
    /// where 0 indicates pressed and released only the current frame
    just_released: [(Command, u16); SIZE],
    released_count: usize,

    /// command input queue
    queue_buffer: [(Command, State); SIZE],
    queue_count: usize,

    /// the keys that have been pressed this frame
    just_pressed: [Command; SIZE],
    pressed_count: usize,

    exists: [BufferState; Command::MAX_COUNT],
}

/// which buffer (if any) the command is in
#[repr(u8)]
#[derive(Clone, Copy)]
enum BufferState {
    None,
    Held,
    HeldAndPresssed,
    Released,
}

/// held size is 16 bc only 10 fingers, plus some padding
/// if someone means to press 16 keys at once anyways they
/// are weird as fuck and should be executed
const SIZE: usize = 10 + 6;

impl Activity {
    pub const DEFAULT: Self = Self {
        held: [(Command::NULL, 0); SIZE],
        held_count: 0,
        just_released: [(Command::NULL, 0); SIZE],
        released_count: 0,
        queue_buffer: [(Command::NULL, State::Released); SIZE],
        queue_count: 0,
        just_pressed: [Command::NULL; SIZE],
        pressed_count: 0,
        exists: [BufferState::None; Command::MAX_COUNT],
    };

    pub fn push(&mut self, entry: (Command, State)) {
        if self.queue_count == (SIZE - 1) {
            // log that the key is dropped
        } else {
            self.queue_buffer[self.queue_count] = entry;
            self.queue_count += 1;
        }
    }

    /// go from queue -> corresponding buffer
    /// incremebt the held buffer by dt
    pub fn run(&mut self, dt: u16) {
        for (cmd, state) in self.queue_buffer.into_iter().take(self.queue_count) {
            match state {
                State::Pressed => self.held_pressed_add(cmd),
                State::Released => self.remove_from_held(cmd),
            }
        }
        for i in 0..self.held_count {
            self.held[i].1 += dt;
        }
    }

    pub fn cleanup(&mut self) {
        let mut temp = Self::DEFAULT;
        temp.held = self.held;
        temp.held_count = self.held_count;
        *self = temp;

        for i in 0..self.held_count {
            let (cmd, _) = self.held[i];
            self.exists[cmd.index()] = BufferState::Held;
        }
    }

    pub fn print_maybe(&self) {
        if self.held_count + self.pressed_count + self.released_count + self.queue_count == 0 {
            return;
        }
        eprintln!("{}", self);
    }

    /// if a key isnt in the held buffer add it
    /// then add it to pressed keys
    fn held_pressed_add(&mut self, cmd: Command) {
        let mut exists = false;

        for i in 0..self.held_count {
            if self.held[i].0 == cmd {
                exists = true
            }
        }

        if !exists && self.held_count < SIZE {
            self.held[self.held_count] = (cmd, 0);
            self.held_count += 1;
            self.exists[cmd.index()] = BufferState::Held;
        }

        if self.pressed_count < SIZE {
            self.just_pressed[self.pressed_count] = cmd;
            self.pressed_count += 1;
            self.exists[cmd.index()] = BufferState::HeldAndPresssed;
        }
    }

    #[inline(always)]
    ///check if command is being held for this frame
    pub fn is_held(&self, cmd: Command) -> bool {
        match self.exists[cmd.index()] {
            BufferState::Held | BufferState::HeldAndPresssed => true,
            _ => false,
        }
    }

    #[inline(always)]
    ///check if command is being held for this frame
    pub fn was_pressed(&self, cmd: Command) -> bool {
        match self.exists[cmd.index()] {
            BufferState::HeldAndPresssed => true,
            _ => false,
        }
    }

    #[inline(always)]
    ///check if command is being held for this frame
    pub fn was_released(&self, cmd: Command) -> bool {
        match self.exists[cmd.index()] {
            BufferState::Released => true,
            _ => false,
        }
    }

    /// remove item from held keys, place into just_released
    /// move held keys buffer down one
    fn remove_from_held(&mut self, cmd: Command) {
        let mut index = SIZE;
        for i in 0..self.held_count {
            if self.held[i].0 == cmd {
                index = i;
            }
        }

        if index != SIZE {
            self.just_released[self.released_count] = self.held[index];
            self.released_count += 1;

            self.held.copy_within((index + 1)..self.held_count, index);
            self.held_count -= 1;

            self.exists[cmd.index()] = BufferState::Released;
        }
    }
}

impl std::default::Default for Activity {
    fn default() -> Self {
        Self::DEFAULT
    }
}

impl Display for Activity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.queue_count == 0 {
            write!(f, "---Key Queue: EMPTY --- \n\n")?;
        } else {
            write!(f, "---Key Queue: {} --- \n[ ", self.queue_count)?;
            for i in 0..self.queue_count {
                write!(
                    f,
                    "({}: {}), ",
                    self.queue_buffer[i].0, self.queue_buffer[i].1
                )?;
            }
            write!(f, "]\n\n")?;
        }

        if self.held_count == 0 {
            write!(f, "--- Held Keys: EMPTY --- \n\n")?;
        } else {
            write!(f, "--- Held Keys: {} --- \n[ ", self.held_count)?;
            for i in 0..self.held_count {
                write!(f, "({}: {}ms), ", self.held[i].0, self.held[i].1)?;
            }
            write!(f, "]\n\n")?;
        }

        if self.pressed_count == 0 {
            write!(f, "--- Pressed Keys: EMPTY --- \n\n")?;
        } else {
            write!(f, "--- Pressed Keys: {} --- \n[ ", self.pressed_count)?;
            for i in 0..self.pressed_count {
                write!(f, "{}, ", self.just_pressed[i])?;
            }
            write!(f, "]\n\n")?;
        }

        if self.released_count == 0 {
            write!(f, "--- Released Keys: EMPTY --- \n\n")?;
        } else {
            write!(f, "--- Released Keys: {} --- \n[ ", self.released_count)?;
            for i in 0..self.released_count {
                write!(
                    f,
                    "({}: {}ms), ",
                    self.just_released[i].0, self.just_released[i].1
                )?;
            }
            write!(f, "]\n\n")?;
        }

        write!(f, "")
    }
}
