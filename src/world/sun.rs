use crate::math::vec3::Vec3;
use crate::world::terrain::smoothstep;

/// half width of the sun square in tangent units (~2.3 degrees)
pub const SUN_SIZE: f32 = 0.04;

/// how hard the sun hits a surface facing it when it's high up (linear rgb)
pub const SUN_RGB: Vec3 = Vec3::new(3.1, 2.9, 2.55);

/// how bright the sun square itself looks, only camera / reflection rays ever see it
pub const SUN_DISC_RGB: Vec3 = Vec3::new(18.0, 16.5, 12.5);

/// the sun is part of the world, the dev menu just moves it around
#[derive(Clone, Copy, Debug)]
pub struct Sun {
    /// where it sits, degrees: azimuth turns from +z towards +x, elevation is above the horizon
    pub azimuth: f32,
    pub elevation: f32,

    /// unit vector pointing at the sun
    pub dir: Vec3,
    /// tangent axes, the visible sun is a square in this frame
    pub u: Vec3,
    pub v: Vec3,
    /// light on a surface facing the sun (linear rgb)
    pub rgb: Vec3,
    /// brightness of the visible square (linear rgb)
    pub disc_rgb: Vec3,
}

impl Sun {
    /// mid morning, the sun the engine always started with (degrees)
    pub const DEFAULT_AZIMUTH: f32 = 38.0;
    pub const DEFAULT_ELEVATION: f32 = 55.0;

    /// angles in degrees, elevation gets kept within 5..85 so the tangent frame never breaks
    pub fn new(azimuth: f32, elevation: f32) -> Self {
        let azimuth = azimuth.rem_euclid(360.0);
        let elevation = elevation.clamp(5.0, 85.0);

        let (a, e) = (azimuth.to_radians(), elevation.to_radians());
        let dir = Vec3::new(a.sin() * e.cos(), e.sin(), a.cos() * e.cos());
        let u = Vec3::Y_POS_NORMAL.cross(dir).normalize();

        // a low sun goes dimmer and redder, its light has to cross way more air //

        let high = smoothstep(0.0, 30.0, elevation);
        let tint = Vec3::new(1.0, 0.6 + 0.4 * high, 0.3 + 0.7 * high) * (0.4 + 0.6 * high);

        Self {
            azimuth,
            elevation,
            dir,
            u,
            v: dir.cross(u),
            rgb: SUN_RGB * tint,
            disc_rgb: SUN_DISC_RGB * tint,
        }
    }
}

impl Default for Sun {
    fn default() -> Self {
        Self::new(Self::DEFAULT_AZIMUTH, Self::DEFAULT_ELEVATION)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_sun() {
        // still the old hard coded mid morning direction //
        let sun = Sun::default();
        let old = Vec3::new(0.356_290_6, 0.814_378_6, 0.458_087_9);

        assert!((sun.dir - old).length() < 0.02, "{:?}", sun.dir);
        assert!(sun.u.dot(sun.dir).abs() < 1e-5 && sun.v.dot(sun.dir).abs() < 1e-5);
        assert!((sun.rgb - SUN_RGB).length() < 1e-3);

        // out of range angles get wrapped / clamped instead of breaking the frame //
        let odd = Sun::new(-15.0, 120.0);
        assert_eq!((odd.azimuth, odd.elevation), (345.0, 85.0));
    }
}
