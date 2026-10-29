use crate::math::vec3::Vec3;
use crate::world::World;
use crate::world::voxel::{Material, VOXEL_SIZE};

pub struct Player {
    pub gait: GaitState,
    pub y_state: VerticalState,
    pub medium: MediumState,
    /// eye position in meters, the camera just sits here
    pub position: Vec3,
    /// meters per second, only y is simulated (gravity / jumps),
    /// x and z come straight from the keys every frame
    pub velocity: Vec3,
    pub hand: Material,
    /// brush diameter in voxels (1..=World::BRUSH_MAX)
    pub brush: u32,
    pub edit_cooldown: f32,

    /// flying straight through walls
    pub noclip: bool,
    /// multiplier on every move from the keys (dev menu)
    pub move_speed: f32,
    /// flying multiplier, keeps growing the longer a movement key is held
    pub fly_boost: f32,
    /// seconds since space was last tapped, for the double tap
    pub since_jump: f32,
}

/// what movement action is the player currently doing
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GaitState {
    Still,
    Sneaking,
    Walking,
    Sprinting,
}

/// what is the player's air-state like
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VerticalState {
    Grounded,
    Rising,
    Falling,
    Flying,
}

/// how does the medium traveled through interact with the player
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MediumState {
    Air,
    Water,
}

impl Player {
    /// hitbox corners relative to the eye (meters), 0.6 wide and 1.7 tall
    pub const BOX_MIN: Vec3 = Vec3::new(-0.3, -1.5, -0.3);
    pub const BOX_MAX: Vec3 = Vec3::new(0.3, 0.2, 0.3);

    /// ledges up to 4 voxels (0.5m) get climbed instead of blocking,
    /// otherwise every 1 voxel step of a slope would catch you
    const STEP_VOXELS: i32 = 4;

    /// water this far below the eye means you're swimming (meters, about chest height)
    const CHEST: f32 = 0.6;

    /// two space taps closer than this (seconds) flip between flying and walking
    pub const DOUBLE_TAP: f32 = 0.3;

    /// flying boost growth per second while moving, and where it tops out
    pub const FLY_RAMP: f32 = 0.8;
    pub const FLY_BOOST_MAX: f32 = 16.0;

    pub const HAND: [Material; 7] = [
        Material::Stone,
        Material::Dirt,
        Material::Grass,
        Material::Wood,
        Material::Leaves,
        Material::Snow,
        Material::Water,
    ];

    /// how far away you can break / place stuff (meters)
    pub const REACH: f32 = 16.0;

    pub const EDIT_REPEAT: f32 = 0.15;

    /// walking physics (m/s^2, m/s), a jump clears about 1.2m
    pub const GRAVITY: f32 = 25.0;
    pub const JUMP_SPEED: f32 = 7.75;
    pub const TERMINAL_SPEED: f32 = 50.0;

    /// in water gravity gets scaled way down and holding space swims up (m/s)
    pub const WATER_GRAVITY: f32 = 0.2;
    pub const SWIM_SPEED: f32 = 2.0;

    /// starts out flying at position (meters)
    pub fn new(position: Vec3) -> Self {
        Self {
            gait: GaitState::Still,
            y_state: VerticalState::Flying,
            medium: MediumState::Air,
            position,
            velocity: Vec3::DEFAULT,
            hand: Material::Stone,
            brush: 1,
            edit_cooldown: 0.0,
            noclip: false,
            move_speed: 1.0,
            fly_boost: 1.0,
            since_jump: Self::DOUBLE_TAP,
        }
    }

    #[inline(always)]
    pub fn flying(&self) -> bool {
        self.y_state == VerticalState::Flying
    }

    /// double tapping space flips between flying and walking,
    /// returns true when the press should count as a jump
    pub fn tap_jump(&mut self, dt: f32, pressed: bool) -> bool {
        self.since_jump += dt;

        if !pressed {
            return false;
        }

        if self.since_jump >= Self::DOUBLE_TAP {
            self.since_jump = 0.0;
            return true;
        }

        // second tap, so switch modes and don't jump on top of it //

        self.y_state = if self.flying() {
            VerticalState::Falling
        } else {
            VerticalState::Flying
        };

        self.velocity[1] = 0.0;
        self.since_jump = Self::DOUBLE_TAP;
        false
    }

    /// one frame of movement, wish is the move the keys asked for (meters)
    /// jump = space tapped this frame, swim = space held
    pub fn update(
        &mut self,
        world: &World,
        wish: Vec3,
        dt: f32,
        jump: bool,
        swim: bool,
        gait: GaitState,
    ) {
        let chest = self.position - Vec3::new(0.0, Self::CHEST, 0.0);

        self.gait = gait;
        self.medium = if world.water_at(chest) {
            MediumState::Water
        } else {
            MediumState::Air
        };

        if self.flying() {
            self.fly(world, wish, dt);
        } else {
            self.walk(world, wish, dt, jump, swim);
        }
    }

    /// flying: the key move as is, just sped up the longer it's held
    fn fly(&mut self, world: &World, wish: Vec3, dt: f32) {
        self.fly_boost = if wish.length_squared() > 0.0 {
            (self.fly_boost * (1.0 + Self::FLY_RAMP * dt)).min(Self::FLY_BOOST_MAX)
        } else {
            1.0
        };

        let delta = wish * (self.move_speed * self.fly_boost);

        if self.noclip {
            self.position += delta;
        } else {
            self.move_by(world, delta);
        }
    }

    /// walking: the flat part of the key move, plus gravity, jumping and swimming
    fn walk(&mut self, world: &World, wish: Vec3, dt: f32, jump: bool, swim: bool) {
        self.fly_boost = 1.0;

        let vy = if self.medium == MediumState::Water {
            if swim {
                Self::SWIM_SPEED
            } else {
                (self.velocity[1] - Self::GRAVITY * Self::WATER_GRAVITY * dt).max(-Self::SWIM_SPEED)
            }
        } else {
            let vy = if jump && self.y_state == VerticalState::Grounded {
                Self::JUMP_SPEED
            } else {
                self.velocity[1]
            };

            (vy - Self::GRAVITY * dt).max(-Self::TERMINAL_SPEED)
        };

        // space jumps here instead of flying up //

        let mut delta = wish * self.move_speed;
        delta[1] = vy * dt;

        let wanted = self.position[1] + delta[1];
        self.move_by(world, delta);

        // a fall cut short means we landed, a rise cut short means we bonked our head //

        let landed = delta[1] < 0.0 && self.position[1] > wanted + 1e-5;
        let bumped = delta[1] > 0.0 && self.position[1] < wanted - 1e-5;

        self.velocity[1] = if landed || bumped { 0.0 } else { vy };

        self.y_state = if landed {
            VerticalState::Grounded
        } else if self.velocity[1] > 0.0 {
            VerticalState::Rising
        } else {
            VerticalState::Falling
        };
    }

    /// world space hitbox at an eye position
    #[inline(always)]
    pub fn hitbox(eye: Vec3) -> (Vec3, Vec3) {
        (eye + Self::BOX_MIN, eye + Self::BOX_MAX)
    }

    /// move by delta (meters), sliding along walls and stepping up low ledges
    fn move_by(&mut self, world: &World, delta: Vec3) {
        let mut eye = self.position;

        // vertical first, so a step up starts from wherever the box landed //

        let (lo, hi) = Self::hitbox(eye);
        eye[1] += world.sweep(lo, hi, 1, delta[1]).0;

        'axis: for axis in [0, 2] {
            let (lo, hi) = Self::hitbox(eye);
            let (travel, blocked) = world.sweep(lo, hi, axis, delta[axis]);

            if !blocked {
                eye[axis] += travel;
                continue;
            }

            // ok so we hit something, see if it's a low ledge we can just climb //

            for step in 1..=Self::STEP_VOXELS {
                let mut raised = eye;
                raised[1] += step as f32 * VOXEL_SIZE;

                let (lo, hi) = Self::hitbox(raised);

                if world.box_blocked(lo, hi) {
                    break;
                }

                let (further, blocked) = world.sweep(lo, hi, axis, delta[axis]);

                if !blocked {
                    eye = raised;
                    eye[axis] += further;
                    continue 'axis;
                }
            }

            // nope, a real wall, slide along it //
            eye[axis] += travel;
        }

        self.position = eye;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world::sun::Sun;
    use crate::world::tree64::Tree64;
    use crate::world::voxel::VOXELS_PER_METER;

    /// a player dropped onto the ground column at (x, z) meters
    fn standing(world: &World, x: f32, z: f32) -> Player {
        let mut player = Player::new(Vec3::new(x, 200.0, z));
        player.move_by(world, Vec3::new(0.0, -199.0, 0.0));
        player
    }

    fn inside_voxel(world: &World, player: &Player) -> bool {
        let (lo, hi) = Player::hitbox(player.position);
        world.box_blocked(lo, hi)
    }

    #[test]
    fn test_collision() {
        let world = World {
            tree: Tree64::build(1337),
            sun: Sun::default(),
        };

        // falling stops on the ground, the feet rest right on a voxel face //

        let player = standing(&world, 128.0, 128.0);
        let eye = player.position;
        assert!(eye[1] > 10.0 && eye[1] < 190.0, "landed at {}", eye[1]);

        let feet = (eye[1] + Player::BOX_MIN[1]) * VOXELS_PER_METER;
        assert!((feet - feet.round()).abs() < 0.02, "feet at {feet}");
        assert!(!inside_voxel(&world, &player));

        // walking around anywhere never ends up inside a voxel //

        let mut player = player;
        for i in 0..2000 {
            let angle = i as f32 * 0.37;
            let delta = Vec3::new(angle.cos() * 0.4, -0.05, angle.sin() * 0.4);
            player.move_by(&world, delta);
            assert!(!inside_voxel(&world, &player), "inside a voxel at step {i}");
        }

        // a huge move still can't tunnel through the ground //

        let mut player = standing(&world, 100.0, 100.0);
        let before = player.position;
        player.move_by(&world, Vec3::new(0.0, -500.0, 0.0));
        assert_eq!(player.position[1], before[1]);
    }

    #[test]
    fn test_walking() {
        let world = World::new();
        let dt = 1.0 / 60.0;
        let still = Vec3::DEFAULT;
        let step = |player: &mut Player, jump: bool| {
            player.update(&world, still, dt, jump, false, GaitState::Still);
        };

        // a double tap turns flying off, then the player drops and lands //

        let mut player = Player::new(Vec3::new(130.0, 150.0, 120.0));
        assert!(player.tap_jump(dt, true));
        assert!(!player.tap_jump(dt, true));
        assert!(!player.flying());

        for _ in 0..600 {
            step(&mut player, false);
        }

        assert_eq!(player.y_state, VerticalState::Grounded);
        let ground = player.position[1];

        // standing still really is still, no creeping //

        step(&mut player, false);
        assert_eq!(player.position[1], ground);

        // a jump goes up about a meter and comes back down to the same spot //

        let mut peak = ground;
        step(&mut player, true);
        for _ in 0..120 {
            step(&mut player, false);
            peak = peak.max(player.position[1]);
        }

        assert!(
            peak - ground > 0.9 && peak - ground < 1.5,
            "jump peak {}",
            peak - ground
        );
        assert_eq!(player.y_state, VerticalState::Grounded);
        assert!((player.position[1] - ground).abs() < 1e-4);
    }
}
