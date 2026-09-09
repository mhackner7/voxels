pub mod activity;
pub mod commands;
pub mod default_keymap;
pub mod handler;
pub mod keymap;
pub mod util;

use crate::engine::input::commands::*;

use super::*;

impl Engine {
    pub fn handle_input(&mut self, dt: f32) {
        self.mouse_motion();

        self.input.process(dt);
        self.input.activity.print_maybe();

        let is_held = |cmd| self.input.activity.is_held(cmd) as i32 as f32;
        let was_pressed = |cmd| self.input.activity.was_pressed(cmd);

        if was_pressed(DeveloperCommands::ToggleFullscreen.into()) {
            self.set_fullscreen = !self.set_fullscreen;
        }

        let base_speed = 3.0 * dt;
        let modified_speed = ((is_held(PlayerCommands::Sprint.into()) * base_speed) + base_speed);

        let mut front = self.camera.front;
        front.y = 0.0;
        if front.length_squared() > 0.0 {
            front = front.normalize();
        }

        let mut right = self.camera.right;
        right.y = 0.0;
        if right.length_squared() > 0.0 {
            right = right.normalize();
        }

        let z_move = is_held(PlayerCommands::MoveForward.into())
            - is_held(PlayerCommands::MoveBackward.into());
        let x_move =
            is_held(PlayerCommands::MoveRight.into()) - is_held(PlayerCommands::MoveLeft.into());
        let y_move = is_held(PlayerCommands::Jump.into()) - is_held(PlayerCommands::Sneak.into());

        let mut movement = (front * z_move) + (right * x_move);

        if movement.length_squared() > 0.0 {
            movement = movement.normalize();
        }
        movement *= modified_speed;
        movement.y = y_move * base_speed;

        if movement.length_squared() > 0.0001 {
            let old_center = self.camera.center;
            self.camera.center += movement;

            if (self.camera.center - old_center).length_squared() > 0.000001 {
                self.frame_accumulation_count = 0;
                self.camera.update();
            }
        }
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
