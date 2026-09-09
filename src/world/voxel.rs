use std::default;

use crate::math::vec3::Vec3;

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
    pub const COLORS: [u32; Material::__COUNT as usize] = colors();
}

#[derive(Debug, Copy, Clone)]
pub struct VoxelInfo {
    pub dimensions: Vec3,
    pub color: u32,
    pub material: Material,
    pub opacity: u8,    // 0-10 value -> increments of 10%
    pub durability: u8, // 0-10 value -> increments of 10%
    pub viscosity: u8,  // 0-3 value, 0 = solid, 3 = water
    pub gravitates: bool,
    pub emits_light: bool,
    pub is_electric: bool,
    pub is_interactive: bool,
    pub is_intangible: bool,
}

impl VoxelInfo {
    const DEFAULT: VoxelInfo = VoxelInfo {
        dimensions: Vec3::DEFAULT,
        color: 0xFFFFFF,
        material: Material::Air,
        opacity: 0,
        durability: 0,
        viscosity: 0,
        gravitates: false,
        emits_light: false,
        is_electric: false,
        is_interactive: false,
        is_intangible: false,
    };

    /// size of the voxels in meters
    /// minecraft voxels = 1 meter^3
    /// lay of the land = 0.1 meter^3
    /// size of voxel volumes is dependant on this
    pub const VOXEL_SIZE: f32 = 1.0;
    pub const VOXELS_PER_UNIT: f32 = 1.0 / Self::VOXEL_SIZE;

    #[inline(always)]
    pub const fn is_transparent(self) -> bool {
        self.opacity < 10
    }
    #[inline(always)]
    pub const fn opacity_float(self) -> f32 {
        100.0 / self.opacity as f32 * 10.0
    }

    #[inline(always)]
    pub const fn is_liquid(self) -> bool {
        self.viscosity == 0
    }
}

const fn colors() -> [u32; Material::__COUNT as usize] {
    let mut arr = [0; Material::__COUNT as usize];

    /*
        THIS IS A LITTLE ENDIAN SYSTEM

        0x12345678 IS LAYED OUT IN MEMORY AS

        78   56   34   12

        SO RRGGBBAA MUST BE CHANGED TO

        AA   BB   GG   RR


    */

    arr[Material::Air as usize] = 0x140800;
    arr[Material::Dirt as usize] = 0x00305B;
    arr[Material::Grass as usize] = 0x7A9B9B;
    arr[Material::Stone as usize] = 0x7C818A;
    arr[Material::Wood as usize] = 0x244F7F;
    arr[Material::Leaves as usize] = 0x4A6D65;
    arr[Material::Snow as usize] = 0xEEF3F4;
    arr[Material::Water as usize] = 0x976F2A;

    arr
}

impl From<u8> for Material {
    fn from(value: u8) -> Self {
        unsafe { std::mem::transmute(value) }
    }
}
