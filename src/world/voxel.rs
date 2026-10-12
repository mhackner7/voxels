/// size of the voxels in meters (0.125m)
/// size of every voxel volume is dependant on this
pub const VOXEL_SIZE: f32 = 1.0 / 8.0;

/// voxels along one meter, converts world space into voxel space
pub const VOXELS_PER_METER: f32 = 1.0 / VOXEL_SIZE;

/// material of cells that are stored as completely solid (no leaf / brick)
pub const SOLID_MATERIAL: Material = Material::Stone;

#[derive(Clone, Copy, Debug, PartialEq, PartialOrd, Ord, Eq)]
#[repr(u8)]
pub enum Material {
    Air,
    Dirt,
    Grass,
    Stone,
    Wood,
    Leaves,
    Snow,
    Water,
    __COUNT,
}

impl Material {
    /// linear rgb diffuse reflectance of every material
    /// Water is never shaded as a surface, its entry is the in water scatter color
    pub const ALBEDO: [[f32; 3]; Material::__COUNT as usize] = [
        [0.0, 0.0, 0.0],     // Air
        [0.14, 0.08, 0.04],  // Dirt
        [0.10, 0.20, 0.045], // Grass
        [0.15, 0.15, 0.16],  // Stone
        [0.18, 0.11, 0.05],  // Wood
        [0.045, 0.15, 0.03], // Leaves
        [0.82, 0.84, 0.88],  // Snow
        [0.02, 0.085, 0.1],  // Water
    ];

    /// how much light water eats per meter, per rgb channel
    /// pretty clear for the first meter, deep teal by WATER_VISIBILITY
    pub const WATER_ABSORPTION: [f32; 3] = [0.9, 0.42, 0.3];
}
