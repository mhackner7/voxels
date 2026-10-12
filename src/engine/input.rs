pub mod activity;
pub mod commands;
pub mod default_keymap;
pub mod handler;
pub mod keymap;
pub mod util;

use crate::engine::input::commands::*;
use crate::engine::player::GaitState;

use super::*;

impl Engine {
    /// everything the keys and mouse do this frame:
    /// ui toggles, looking around, then moving the player through the world
    pub fn handle_input(&mut self, dt: f32) {
        self.input.process(dt);
        self.input.activity.print_maybe();

        if self
            .input
            .activity
            .was_pressed(DeveloperCommands::ToggleFullscreen.into())
        {
            self.set_fullscreen = !self.set_fullscreen;
        }

        self.ui_input();

        // the dev menu eats the movement keys and the mouse while it's open, //
        // gravity still runs though

        let (wish, gait, tapped, swim) = if self.ui.menu {
            self.input.last_mouse_motion = (0.0, 0.0);
            (Vec3::DEFAULT, GaitState::Still, false, false)
        } else {
            self.mouse_motion();
            self.wish_move(dt)
        };

        let jump = self.player.tap_jump(dt, tapped);
        let before = self.player.position;

        self.player.update(&self.world, wish, dt, jump, swim, gait);
        self.camera.center = self.player.position;

        // only a real move restarts the image, pushing into a wall doesn't //

        if self.player.position.data != before.data {
            self.frame_accumulation_count = 0;
        }
    }

    /// the move the keys ask for this frame (meters), how it's being asked,
    /// and space tapped / held
    fn wish_move(&self, dt: f32) -> (Vec3, GaitState, bool, bool) {
        let is_held = |cmd| self.input.activity.is_held(cmd) as i32 as f32;

        let base_speed = 3.0 * dt;
        let sprint = is_held(PlayerCommands::Sprint.into());
        let modified_speed = (sprint * base_speed) + base_speed;

        let mut front = self.camera.front;
        front[1] = 0.0;
        if front.length_squared() > 0.0 {
            front = front.normalize();
        }

        let mut right = self.camera.right;
        right[1] = 0.0;
        if right.length_squared() > 0.0 {
            right = right.normalize();
        }

        let z_move = is_held(PlayerCommands::MoveForward.into())
            - is_held(PlayerCommands::MoveBackward.into());
        let x_move =
            is_held(PlayerCommands::MoveRight.into()) - is_held(PlayerCommands::MoveLeft.into());
        let jump = is_held(PlayerCommands::Jump.into());
        let sneak = is_held(PlayerCommands::Sneak.into());

        let mut movement = (front * z_move) + (right * x_move);

        if movement.length_squared() > 0.0 {
            movement = movement.normalize();
        }
        movement *= modified_speed;
        movement[1] = (jump - sneak) * base_speed;

        let gait = if sneak > 0.0 {
            GaitState::Sneaking
        } else if z_move == 0.0 && x_move == 0.0 {
            GaitState::Still
        } else if sprint > 0.0 {
            GaitState::Sprinting
        } else {
            GaitState::Walking
        };

        let tapped = self.input.activity.was_pressed(PlayerCommands::Jump.into());

        (movement, gait, tapped, jump > 0.0)
    }

    pub fn mouse_motion(&mut self) {
        let yaw = self.camera.yaw;
        let pitch = self.camera.pitch;

        let dx = self.input.last_mouse_motion.0 * self.input.mouse_sensitivity;
        let dy = self.input.last_mouse_motion.1 * self.input.mouse_sensitivity;

        if dx.abs() > 0.0001 || dy.abs() > 0.0001 {
            self.frame_accumulation_count = 0;
            self.camera.pitch = pitch - dy as f32;
            self.camera.yaw = yaw + dx as f32;
        }

        self.input.last_mouse_motion = (0.0, 0.0);
    }
}
