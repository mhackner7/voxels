use crate::math::vec3::Vec3;
use crate::simd::packetvec::Packed_Vec3_x4;
use crate::simd::simdf32::SimdF32;
use crate::simd::simdu32::SimdU32;
use crate::world::sun::*;
use crate::world::voxel::*;

/*
    LIGHTING MODEL

    everything in here is linear rgb, the display pass tone maps it at the end
    the sun is a square light that lives in the world, the sky is a gradient dome,
    surfaces are plain lambert with cosine weighted bounces,
    which is nice because the 1 / pi and the cosine cancel against the sampling pdf
*/

/// how bright the sky is at the horizon / straight up / looking down
pub const SKY_HORIZON: Vec3 = Vec3::new(0.52, 0.72, 1.0);
pub const SKY_ZENITH: Vec3 = Vec3::new(0.20, 0.42, 0.90);
pub const SKY_GROUND: Vec3 = Vec3::new(0.36, 0.42, 0.50);

/// light a bounce hit gets besides the sun, a cheap stand in for a 2nd bounce
pub const BOUNCE_AMBIENT: Vec3 = Vec3::new(0.30, 0.38, 0.50);

/// exposure, applied right before tone mapping
pub const EXPOSURE: f32 = 0.72;

/// how far secondary rays start off the surface (meters, ~1/100 voxel)
pub const SURFACE_OFFSET: f32 = 1e-3;

/// how far you can see through water (meters), anything deeper is just water color
pub const WATER_VISIBILITY: f32 = 6.0;

/// reflection rays off water give up here (meters), past that they just show sky
pub const WATER_REFLECTION_DISTANCE: f32 = 32.0;

#[inline(always)]
pub fn splat(v: Vec3) -> Packed_Vec3_x4 {
    Packed_Vec3_x4::from_single(v)
}

/// sky gradient for every lane's direction, no sun in it
#[inline(always)]
pub fn sky(dirs: Packed_Vec3_x4) -> Packed_Vec3_x4 {
    let up = dirs.ys.max(SimdF32::ZEROES).sqrt();
    let down = (dirs.ys.neg() * SimdF32::from_float(4.0)).min(SimdF32::ONES);

    let horizon = splat(SKY_HORIZON);
    let above = horizon.add_scaled(splat(SKY_ZENITH) - horizon, up);
    let below = horizon.add_scaled(splat(SKY_GROUND) - horizon, down.max(SimdF32::ZEROES));

    Packed_Vec3_x4::select(dirs.ys.is_greater_or_equal(SimdF32::ZEROES), above, below)
}

/// lanes looking right at the sun square
#[inline(always)]
pub fn sun_square(dirs: Packed_Vec3_x4, sun: &Sun) -> SimdU32 {
    let s = dirs.dot(splat(sun.dir));
    let u = dirs.dot(splat(sun.u)).abs();
    let v = dirs.dot(splat(sun.v)).abs();
    let size = s * SimdF32::from_float(SUN_SIZE);

    s.is_greater_than(SimdF32::ZEROES) & u.is_less_than(size) & v.is_less_than(size)
}

/// sky plus the sun square, what a camera or mirror ray sees when it gets out
#[inline(always)]
pub fn sky_with_sun(dirs: Packed_Vec3_x4, sun: &Sun) -> Packed_Vec3_x4 {
    Packed_Vec3_x4::select(sun_square(dirs, sun), splat(sun.disc_rgb), sky(dirs))
}

/// direction to a random spot on the sun square, that's what makes shadow edges soft
#[inline(always)]
pub fn sun_sample(sun: &Sun, u1: SimdF32, u2: SimdF32) -> Packed_Vec3_x4 {
    let spread = SimdF32::from_float(2.0 * SUN_SIZE);
    let half = SimdF32::from_float(0.5);

    splat(sun.dir)
        .add_scaled(splat(sun.u), (u1 - half) * spread)
        .add_scaled(splat(sun.v), (u2 - half) * spread)
        .normalize()
}

/// lowbias32 hash on every lane
#[inline(always)]
pub fn hash(x: SimdU32) -> SimdU32 {
    let x = x.xor(x.shrl::<16>()).mul(SimdU32::from(0x7feb_352d));
    let x = x.xor(x.shrl::<15>()).mul(SimdU32::from(0x846c_a68b));
    x.xor(x.shrl::<16>())
}

/// per lane random numbers, one hash per draw, nothing allocated
pub struct Rng {
    state: SimdU32,
}

impl Rng {
    #[inline(always)]
    pub fn new(pixels: SimdU32, frame: u32) -> Self {
        Self {
            state: hash(pixels.xor(SimdU32::from(frame.wrapping_mul(0x9e37_79b9)))),
        }
    }

    /// uniform [0, 1) on every lane
    #[inline(always)]
    pub fn next(&mut self) -> SimdF32 {
        self.state = hash(self.state + SimdU32::from(0x632b_e5ab));
        self.state.to_f32_unit()
    }
}

/// cosine weighted direction around unit normals
/// branchless orthonormal basis from duff et al. 2017
#[inline(always)]
pub fn cosine_sample(n: Packed_Vec3_x4, u1: SimdF32, u2: SimdF32) -> Packed_Vec3_x4 {
    let sign = n.zs.signs();
    let a = (sign + n.zs).recip().neg();
    let b = n.xs * n.ys * a;

    let tangent = Packed_Vec3_x4 {
        xs: (sign * n.xs * n.xs).mul_then_add(a, SimdF32::ONES),
        ys: sign * b,
        zs: (sign * n.xs).neg(),
    };
    let bitangent = Packed_Vec3_x4 {
        xs: b,
        ys: (n.ys * n.ys).mul_then_add(a, sign),
        zs: n.ys.neg(),
    };

    let r = u1.sqrt();
    let (sin, cos) =
        ((u2 - SimdF32::from_float(0.5)) * SimdF32::from_float(std::f32::consts::TAU)).sin_cos();
    let z = (SimdF32::ONES - u1).max(SimdF32::ZEROES).sqrt();

    tangent
        .scale(r * cos)
        .add_scaled(bitangent, r * sin)
        .add_scaled(n, z)
}

/// albedo of every lane's hit, with a little per voxel brightness jitter for texture
/// the points sit on the hit face, so half a voxel against the normal lands inside the voxel
#[inline(always)]
pub fn albedo(
    materials: [Material; 4],
    points: Packed_Vec3_x4,
    normals: Packed_Vec3_x4,
) -> Packed_Vec3_x4 {
    let inside = points.add_scaled(normals, SimdF32::from_float(-0.5 * VOXEL_SIZE));
    let voxel = inside.scale(SimdF32::from_float(VOXELS_PER_METER));

    let key = |v: SimdF32, prime: u32| {
        v.floor_to_i32()
            .reinterpret_as_u32()
            .mul(SimdU32::from(prime))
    };
    let noise = hash(
        key(voxel.xs, 73_856_093)
            .xor(key(voxel.ys, 19_349_663))
            .xor(key(voxel.zs, 83_492_791)),
    );
    let shade = noise
        .to_f32_unit()
        .mul_then_add(SimdF32::from_float(0.22), SimdF32::from_float(0.89));

    let a = materials.map(|m| Material::ALBEDO[m as usize]);

    Packed_Vec3_x4::new(
        [a[0][0], a[1][0], a[2][0], a[3][0]],
        [a[0][1], a[1][1], a[2][1], a[3][1]],
        [a[0][2], a[1][2], a[2][2], a[3][2]],
    )
    .scale(shade)
}

/// lanes whose material is water
#[inline(always)]
pub fn water_lanes(materials: [Material; 4]) -> SimdU32 {
    SimdU32::from(materials.map(|m| if m == Material::Water { u32::MAX } else { 0 }))
}

/// beer lambert transmission through `distance` meters of water,
/// faded out so nothing at all shows through past WATER_VISIBILITY
#[inline(always)]
pub fn water_transmission(distance: SimdF32) -> Packed_Vec3_x4 {
    let d = distance
        .min(SimdF32::from_float(WATER_VISIBILITY))
        .to_array();
    let fade = d.map(|d| 1.0 - d / WATER_VISIBILITY);
    let [ar, ag, ab] = Material::WATER_ABSORPTION;

    Packed_Vec3_x4::new(
        [0, 1, 2, 3].map(|i| (-ar * d[i]).exp() * fade[i]),
        [0, 1, 2, 3].map(|i| (-ag * d[i]).exp() * fade[i]),
        [0, 1, 2, 3].map(|i| (-ab * d[i]).exp() * fade[i]),
    )
}

/// the color deep water glows with, lit by the sun
#[inline(always)]
pub fn water_scatter(sun: &Sun) -> Packed_Vec3_x4 {
    let [r, g, b] = Material::ALBEDO[Material::Water as usize];
    splat(Vec3::new(r, g, b)).scale(SimdF32::from_float(sun.rgb[1]))
}

/// aces fit (narkowicz) + gamma 2.0, packed as the pixel buffer's 0xAABBGGRR
#[inline(always)]
pub fn display(rgb: Packed_Vec3_x4) -> SimdU32 {
    let aces = |x: SimdF32| {
        let x = x * SimdF32::from_float(EXPOSURE);
        let num = x * x.mul_then_add(SimdF32::from_float(2.51), SimdF32::from_float(0.03));
        let den = x.mul_then_add(
            x.mul_then_add(SimdF32::from_float(2.43), SimdF32::from_float(0.59)),
            SimdF32::from_float(0.14),
        );
        (num / den).clamp_to(SimdF32::ZEROES, SimdF32::ONES)
    };

    let byte = |x: SimdF32| {
        SimdU32::from(
            aces(x)
                .sqrt()
                .mul_then_add(SimdF32::from_float(255.0), SimdF32::from_float(0.5)),
        )
    };

    byte(rgb.xs)
        .or(byte(rgb.ys).shll::<8>())
        .or(byte(rgb.zs).shll::<16>())
        .or(SimdU32::from(0xFF00_0000))
}
