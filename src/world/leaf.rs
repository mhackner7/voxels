use crate::world::voxel::Material;

/// occupancy of an 8x8x8 voxel grid, exactly one cache line
/// linear voxel index = x | (y << 3) | (z << 6), word = index >> 6 (one z slice per word)
/// no material data is stored here, the traversal only touches this
#[repr(C, align(64))]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct Leaf {
    /// voxel occupancy bitarray, 1 = any non air voxel (water included)
    pub voxels: [u64; 8],
}

impl Leaf {
    pub const VOXEL_COUNT: usize = 512;
}

/// materials of one leaf, 4 bits per voxel (256 bytes)
/// bricks[i] belongs to leaves[i], same linear voxel index
/// only read once a ray has hit a voxel
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct Brick {
    /// two voxels per byte, even index in the low nibble
    pub grid: [u8; Leaf::VOXEL_COUNT / 2],
}

impl Brick {
    pub const EMPTY: Self = Self {
        grid: [0; Leaf::VOXEL_COUNT / 2],
    };

    #[inline(always)]
    pub fn get(&self, index: usize) -> Material {
        let nibble = (self.grid[index >> 1] >> ((index & 1) << 2)) & 15;
        // SAFETY: only ever written from a Material by set
        unsafe { std::mem::transmute::<u8, Material>(nibble) }
    }

    #[inline(always)]
    pub fn set(&mut self, index: usize, material: Material) {
        let shift = (index & 1) << 2;
        self.grid[index >> 1] =
            (self.grid[index >> 1] & !(15 << shift)) | ((material as u8) << shift);
    }
}
