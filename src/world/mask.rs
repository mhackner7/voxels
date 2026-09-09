use crate::math::vec3::Vec3;
use crate::simd::packetvec::Packed_Vec3_x4;
use crate::world::brick::Brick;

#[derive(Copy, Clone, Debug, Default)]
pub struct Mask {
    pub bits: u64,
}

impl Mask {
    pub const XYZ_FROM_BIT: [Vec3; 64] = Mask::arr_linear_to_xyz();

    pub const SIMD_COORD_LUT: [Packed_Vec3_x4; 64] = Mask::arr_linear_to_simd_xyz();

    const fn arr_linear_to_xyz() -> [Vec3; 64] {
        let mut arr = [Vec3::DEFAULT; 64];
        let mut i = 0;
        while i < 64 {
            let (x, y, z) = Mask::index_from_linear(i as u8);
            arr[i] = Vec3::new(x as f32, y as f32, z as f32);
            i = i + 1;
        }
        arr
    }

    const fn arr_linear_to_simd_xyz() -> [Packed_Vec3_x4; 64] {
        let mut arr = [Packed_Vec3_x4::ZEROES; 64];
        let mut i = 0;
        while i < 64 {
            let (x, y, z) = Mask::index_from_linear(i as u8);
            unsafe {
                arr[i] = Packed_Vec3_x4 {
                    xs: std::mem::transmute([x as f32; 4]),
                    ys: std::mem::transmute([y as f32; 4]),
                    zs: std::mem::transmute([x as f32; 4]),
                };
            }
            i = i + 1;
        }
        arr
    }

    #[inline(always)]
    pub const fn linear_from_index(x: u8, y: u8, z: u8) -> u8 {
        debug_assert!(
            x < 4,
            "x should be one of 0 | 1 | 2 | 3, its a 0-indexed 4x4x4 volume"
        );
        debug_assert!(
            y < 4,
            "y should be one of 0 | 1 | 2 | 3, its a 0-indexed 4x4x4 volume"
        );
        debug_assert!(
            z < 4,
            "z should be one of 0 | 1 | 2 | 3, its a 0-indexed 4x4x4 volume"
        );

        /*
            Linear Order from xyz
            2 bits per coord, 4x4x4 volume
            z1 z0 y1 y0 x1 x0
        */

        (z << 4) | (y << 2) | x
    }
    #[inline(always)]
    pub const fn index_from_linear(mi: u8) -> (u8, u8, u8) {
        debug_assert!(mi < 64, "invalid linear index");
        
        let x = mi & 0b11u8;
        let y = (mi >> 2) & 0b11u8;
        let z = (mi >> 4) & 0b11u8;

        const BRICK_WIDTH: u8 = Brick::BRICK_WIDTH_VOXELS as u8;

        (x * BRICK_WIDTH, y * BRICK_WIDTH, z * BRICK_WIDTH)
    }
    #[inline(always)]
    pub fn set_bit(mask: Mask, index: u8) -> Mask {
        Mask {
            bits: mask.bits | 0b1u64 << index,
        }
    }

    #[inline(always)]
    /// not passed by &mut self, must reassign
    /// mask = mask.clear_bit()
    pub fn clear_bit(self, index: u32) -> Self {
        Self {
            bits: self.bits & !(0b1u64 << index),
        }
    }

    #[inline(always)]
    pub fn active_count(self) -> u32 {
        self.bits.count_ones()
    }

    #[inline(always)]
    pub fn trailing_zeroes(self) -> u32 {
        self.bits.trailing_zeros()
    }

    #[inline(always)]
    /// get the next index of the vector the mask denotes
    /// intended for use with Multigrid or future bit-masked 4x4x4 volumes
    /// bit_index must be zero-based
    pub fn vector_index_for(self, bit_index: u32) -> usize {
        debug_assert!(bit_index < 64, "invalid bit index, out of range");

        debug_assert!(
            (self.bits & 1u64 << bit_index) != 0,
            "Invalid bit index, tried to access chunk at nonexistent index"
        );

        // all active bits BEFORE given index
        let isolated_bits: u64 = self.bits & !(u64::MAX << bit_index);

        // get the count for the remaining bits
        isolated_bits.count_ones() as usize
    }
}
