use crate::math::vec3::Vec3;
use crate::world::voxel::Material;
use winit::keyboard::SmolStr;

pub struct Player {
    pub name: SmolStr,
    pub state: PlayerState,
    pub position: Vec3,
    pub velocity: Vec3,
    pub hand: Material,
}

pub struct PlayerState {
    gait: GaitState,
    y_state: VerticalState,
    air_time: u32,
    run_time: u32,
    medium: MediumState,
}

/// what movement action is the player currently doing
pub enum GaitState {
    Still,
    Sneaking,
    Walking,
    Sprinting,
}

/// what is the player's air-state like
pub enum VerticalState {
    Grounded,
    Rising,
    Falling,
    Flying,
}

/// how does the medium traveled through interact with the player
pub enum MediumState {
    Air,
    Water,
}

impl Player {
    // const SPEED_MULTIPLIERS: [f32; Gait::MaxMoveStates as usize] = [0.0, 0.5, 1.0, 1.5];

    // pub fn move
}

// impl Default for Player {
//     fn default() -> Self {
//         Self {
//             name: "player1".into(),
//             position: Vec3::DEFAULT,
//             state: PlayerState {
//                 gait: GaitState::Still,
//                 y_state: VerticalState::Flying,
//                 air_time: 0,
//                 medium: MediumState::Air,
//             },
//         }
//     }
// }
