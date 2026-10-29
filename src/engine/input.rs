use crate::engine::player::GaitState;
use crate::world::voxel::{Material, VOXELS_PER_METER};
use winit::event::MouseButton;
use winit::keyboard::KeyCode;

use super::*;

// every action is one bit: held = down now, pressed = went down since last frame

#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Action {
    Forward,
    Back,
    Left,
    Right,
    Sprint,
    Jump,
    Sneak,
    /// middle click picks up what you're looking at
    Break,
    Place,
    Pick,
    BrushGrow,
    BrushShrink,
    HandNext,
    HandBack,
    Overlay,
    Perf,
    Menu,
    Escape,
    Fullscreen,
}

impl Action {
    pub fn from_key(key: KeyCode) -> Option<Self> {
        Some(match key {
            KeyCode::KeyW => Self::Forward,
            KeyCode::KeyS => Self::Back,
            KeyCode::KeyA => Self::Left,
            KeyCode::KeyD => Self::Right,
            KeyCode::ShiftLeft => Self::Sprint,
            KeyCode::Space => Self::Jump,
            KeyCode::KeyC => Self::Sneak,
            KeyCode::ArrowRight => Self::HandNext,
            KeyCode::ArrowLeft => Self::HandBack,
            KeyCode::Digit1 => Self::Overlay,
            KeyCode::Digit2 => Self::Perf,
            KeyCode::Digit3 => Self::Menu,
            KeyCode::Escape => Self::Escape,
            KeyCode::Digit0 => Self::Fullscreen,
            _ => return None,
        })
    }

    pub fn from_button(button: MouseButton) -> Option<Self> {
        Some(match button {
            MouseButton::Left => Self::Break,
            MouseButton::Right => Self::Place,
            MouseButton::Middle => Self::Pick,
            _ => return None,
        })
    }

    #[inline(always)]
    const fn bit(self) -> u32 {
        1 << self as u32
    }
}

pub struct Input {
    held: u32,
    pressed: u32,
    /// raw motion of the last mouse event (device units)
    pub mouse_motion: (f64, f64),
    pub mouse_sensitivity: f64,
}

impl Input {
    pub const fn new() -> Self {
        Self {
            held: 0,
            pressed: 0,
            mouse_motion: (0.0, 0.0),
            mouse_sensitivity: 0.004,
        }
    }

    /// only the first down counts as a press, so os key repeat never does
    pub fn set(&mut self, action: Action, down: bool) {
        let bit = action.bit();

        if down {
            self.pressed |= bit & !self.held;
            self.held |= bit;
        } else {
            self.held &= !bit;
        }
    }

    /// a press without a release, the scroll wheel
    pub fn tap(&mut self, action: Action) {
        self.pressed |= action.bit();
    }

    #[inline(always)]
    pub fn is_held(&self, action: Action) -> bool {
        self.held & action.bit() != 0
    }

    #[inline(always)]
    pub fn was_pressed(&self, action: Action) -> bool {
        self.pressed & action.bit() != 0
    }

    pub fn end_frame(&mut self) {
        self.pressed = 0;
    }
}

impl Engine {
    /// everything the keys and mouse do this frame:
    /// ui toggles, looking around, then moving the player through the world
    pub fn handle_input(&mut self, dt: f32) {
        if self.input.was_pressed(Action::Fullscreen) {
            self.set_fullscreen = !self.set_fullscreen;
        }

        self.ui_input();

        // the dev menu eats the movement keys and the mouse while it's open, //
        // gravity still runs though

        let (wish, gait, tapped, swim) = if self.ui.menu {
            self.input.mouse_motion = (0.0, 0.0);
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

        self.camera.update();
        self.target = self
            .world
            .raycast(self.camera.center, self.camera.front, Player::REACH);

        if !self.ui.menu {
            self.edit_input(dt);
        }
    }

    fn edit_input(&mut self, dt: f32) {
        let input = &self.input;
        let player = &mut self.player;

        if input.was_pressed(Action::BrushGrow) {
            player.brush = (player.brush + 1).min(World::BRUSH_MAX);
        }

        if input.was_pressed(Action::BrushShrink) {
            player.brush = (player.brush - 1).max(1);
        }

        let cycle =
            input.was_pressed(Action::HandNext) as i32 - input.was_pressed(Action::HandBack) as i32;

        if cycle != 0 {
            let at = Player::HAND
                .iter()
                .position(|&m| m == player.hand)
                .unwrap_or(0) as i32;
            player.hand = Player::HAND[(at + cycle).rem_euclid(Player::HAND.len() as i32) as usize];
        }

        let Some(target) = self.target else {
            return;
        };

        if input.was_pressed(Action::Pick) {
            player.hand = target.material;
        }

        // a fresh click goes off right away, holding it repeats every EDIT_REPEAT //

        let carve = input.is_held(Action::Break);
        let place = input.is_held(Action::Place);

        if input.was_pressed(Action::Break) || input.was_pressed(Action::Place) {
            player.edit_cooldown = 0.0;
        }

        player.edit_cooldown -= dt;

        if !(carve || place) || player.edit_cooldown > 0.0 {
            return;
        }

        player.edit_cooldown = Player::EDIT_REPEAT;

        let material = if carve { Material::Air } else { player.hand };
        let center = World::brush_center(&target, player.brush, !carve);

        // never build the player into a wall, unless it can fly through walls anyway //
        let keep_out =
            (!(player.noclip && player.flying())).then(|| Player::hitbox(player.position));

        if self.world.edit(center, player.brush, material, keep_out) {
            // faces around it (plus a meter for shadows / bounce) go back to young //

            let reach = player.brush as f32 * 0.5 + VOXELS_PER_METER;
            let lo = [0, 1, 2].map(|a| (center[a] - reach).floor() as i32);
            let hi = [0, 1, 2].map(|a| (center[a] + reach).ceil() as i32);

            self.lights.invalidate(lo, hi);
            self.frame_accumulation_count = 0;
            self.target = self
                .world
                .raycast(self.camera.center, self.camera.front, Player::REACH);
        }
    }

    /// the move the keys ask for this frame (meters), how it's being asked,
    /// and space tapped / held
    fn wish_move(&self, dt: f32) -> (Vec3, GaitState, bool, bool) {
        let is_held = |action| self.input.is_held(action) as i32 as f32;

        let base_speed = 3.0 * dt;
        let sprint = is_held(Action::Sprint);
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

        let z_move = is_held(Action::Forward) - is_held(Action::Back);
        let x_move = is_held(Action::Right) - is_held(Action::Left);
        let jump = is_held(Action::Jump);
        let sneak = is_held(Action::Sneak);

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

        let tapped = self.input.was_pressed(Action::Jump);

        (movement, gait, tapped, jump > 0.0)
    }

    pub fn mouse_motion(&mut self) {
        let yaw = self.camera.yaw;
        let pitch = self.camera.pitch;

        let dx = self.input.mouse_motion.0 * self.input.mouse_sensitivity;
        let dy = self.input.mouse_motion.1 * self.input.mouse_sensitivity;

        if dx.abs() > 0.0001 || dy.abs() > 0.0001 {
            self.frame_accumulation_count = 0;
            self.camera.pitch = pitch - dy as f32;
            self.camera.yaw = yaw + dx as f32;
        }

        self.input.mouse_motion = (0.0, 0.0);
    }
}
