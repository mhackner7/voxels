use crate::math::vec3::Vec3;
use crate::simd::packetvec::Packed_Vec3_x4;
use crate::world::World;
use crate::world::sun::Sun;
use crate::world::tree64::Tree64;
use std::io::{IsTerminal, Read, Write};
use std::path::{Path, PathBuf};

/*
    WORLD CACHE

    generating the world takes a couple seconds, loading it back takes a fraction of that.
    the file is just every pool of the tree dumped as raw bytes behind a small header,
    they're all plain #[repr(C)] data so there's nothing to convert either way

    the header carries a hash of the generator's own source, so changing the terrain
    (or the tree layout) makes old caches stale on their own, no stale worlds sneaking in.
    only the freshly generated world is ever cached, edits never get written back
*/

/// whether to use the cached world when there is one
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Reuse {
    /// ask in the terminal (just reuse it when there's no terminal to ask in)
    Ask,
    Yes,
    No,
}

const MAGIC: [u8; 4] = *b"VOXW";

/// bump whenever the file layout itself changes
const FORMAT: u64 = 1;

/// fingerprint of everything that decides what a world looks like and how it's laid out,
/// a cache only counts when the exact same code wrote it
pub const SOURCE_HASH: u64 = {
    let mut hash = fnv(0xcbf2_9ce4_8422_2325 ^ FORMAT, include_bytes!("terrain.rs"));
    hash = fnv(hash, include_bytes!("build.rs"));
    hash = fnv(hash, include_bytes!("tree64.rs"));
    hash = fnv(hash, include_bytes!("leaf.rs"));
    fnv(hash, include_bytes!("voxel.rs"))
};

/// fnv-1a, runs at compile time
const fn fnv(mut hash: u64, bytes: &[u8]) -> u64 {
    let mut i = 0;

    while i < bytes.len() {
        hash = (hash ^ bytes[i] as u64).wrapping_mul(0x0100_0000_01b3);
        i += 1;
    }

    hash
}

/// magic + hash + seed + 7 pool lengths + top
const HEADER: usize = 4 + 8 + 8 + 7 * 8 + 4;

impl World {
    /// where the world for a seed gets cached
    pub fn cache_path(seed: u64) -> PathBuf {
        PathBuf::from(format!("cache/world_{seed}.bin"))
    }

    /// the cached world for seed if there is one (and reuse says so), otherwise a freshly
    /// generated one that gets cached for next time
    pub fn load_or_build(seed: u64, reuse: Reuse) -> Self {
        let path = Self::cache_path(seed);
        let start = std::time::Instant::now();

        let cached = if reuse == Reuse::No || !path.exists() {
            None
        } else {
            match Tree64::load(&path, seed) {
                Ok(Some(tree)) => Some(tree),
                Ok(None) => {
                    println!(
                        "[World] the generator changed since that cache was made, building a new one"
                    );
                    None
                }
                Err(e) => {
                    println!("[World] couldn't read {}: {e}", path.display());
                    None
                }
            }
        };

        if let Some(tree) = cached
            && (reuse == Reuse::Yes || ask_reuse(seed, &path))
        {
            println!(
                "[World] loaded cached world (seed {seed}) in {:.0}ms",
                start.elapsed().as_secs_f64() * 1000.0
            );

            return World {
                tree,
                sun: Sun::default(),
            };
        }

        let tree = Tree64::build(seed);

        match tree.save(&path, seed) {
            Ok(()) => println!("[World] cached it in {}", path.display()),
            Err(e) => println!("[World] couldn't cache the world: {e}"),
        }

        World {
            tree,
            sun: Sun::default(),
        }
    }
}

/// asks in the terminal, enter / y means yes. nobody to ask means yes too
fn ask_reuse(seed: u64, path: &Path) -> bool {
    if !std::io::stdin().is_terminal() {
        return true;
    }

    let size = std::fs::metadata(path).map_or(0, |m| m.len()) >> 20;
    print!("reuse cached world (seed {seed}, {size}MB)? [Y/n] ");
    let _ = std::io::stdout().flush();

    let mut answer = String::new();
    let _ = std::io::stdin().read_line(&mut answer);

    !answer.trim().eq_ignore_ascii_case("n")
}

impl Tree64 {
    /// write every pool behind the header, through a temp file so a half written
    /// cache never takes the real one's place
    pub fn save(&self, path: &Path, seed: u64) -> std::io::Result<()> {
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir)?;
        }

        let temp = path.with_extension("tmp");
        let mut file = std::fs::File::create(&temp)?;

        let mut header = Vec::with_capacity(HEADER);
        header.extend_from_slice(&MAGIC);
        header.extend_from_slice(&SOURCE_HASH.to_le_bytes());
        header.extend_from_slice(&seed.to_le_bytes());

        for len in self.pool_lens() {
            header.extend_from_slice(&(len as u64).to_le_bytes());
        }

        header.extend_from_slice(&self.bmax.ys.to_array()[0].to_le_bytes());
        file.write_all(&header)?;

        file.write_all(bytes(&self.branches))?;
        file.write_all(bytes(&self.leaves))?;
        file.write_all(bytes(&self.bricks))?;
        file.write_all(bytes(&self.leaf_blocks))?;
        file.write_all(bytes(&self.leaf_block_materials))?;
        file.write_all(bytes(&self.water_leaves))?;
        file.write_all(bytes(&self.skip))?;
        file.sync_all()?;

        std::fs::rename(temp, path)
    }

    /// read a cache back, Ok(None) = it's from another seed or another version of the code
    pub fn load(path: &Path, seed: u64) -> std::io::Result<Option<Self>> {
        let mut file = std::fs::File::open(path)?;
        let mut header = [0u8; HEADER];
        file.read_exact(&mut header)?;

        let word = |i: usize| u64::from_le_bytes(header[4 + i * 8..12 + i * 8].try_into().unwrap());

        if header[..4] != MAGIC || word(0) != SOURCE_HASH || word(1) != seed {
            return Ok(None);
        }

        let lens: [usize; 7] = std::array::from_fn(|i| word(2 + i) as usize);
        let top = f32::from_le_bytes(header[HEADER - 4..].try_into().unwrap());

        // the file has to be exactly as long as the header says, so a cut off cache //
        // never gets half read (and a garbage length never gets allocated)

        let sizes = Self::pool_sizes();
        let body: u64 = (0..7).map(|i| (lens[i] * sizes[i]) as u64).sum();

        if file.metadata()?.len() != HEADER as u64 + body {
            return Ok(None);
        }

        let mut tree = Self {
            branches: read(&mut file, lens[0])?,
            leaves: read(&mut file, lens[1])?,
            bricks: read(&mut file, lens[2])?,
            leaf_blocks: read(&mut file, lens[3])?,
            leaf_block_materials: read(&mut file, lens[4])?,
            water_leaves: read(&mut file, lens[5])?,
            skip: read(&mut file, lens[6])?,
            bmax: Packed_Vec3_x4::ZEROES,
        };

        tree.bmax = Packed_Vec3_x4::from_single(Vec3::new(Self::WIDTH, top, Self::WIDTH));
        Ok(Some(tree))
    }

    /// pool lengths, in file order
    fn pool_lens(&self) -> [usize; 7] {
        [
            self.branches.len(),
            self.leaves.len(),
            self.bricks.len(),
            self.leaf_blocks.len(),
            self.leaf_block_materials.len(),
            self.water_leaves.len(),
            self.skip.len(),
        ]
    }

    /// bytes per element of every pool, in file order
    fn pool_sizes() -> [usize; 7] {
        [
            size_of::<crate::world::tree64::Branch>(),
            size_of::<crate::world::leaf::Leaf>(),
            size_of::<crate::world::leaf::Brick>(),
            size_of::<[u64; 2]>(),
            size_of::<[u8; 64]>(),
            size_of::<u64>(),
            size_of::<[u8; 2]>(),
        ]
    }
}

/// a pool as raw bytes
fn bytes<T: Copy>(pool: &[T]) -> &[u8] {
    // SAFETY: every pool is #[repr(C)] plain data (ints and arrays of them), no padding
    unsafe { std::slice::from_raw_parts(pool.as_ptr() as *const u8, size_of_val(pool)) }
}

/// a pool of len elements straight from the file into its own memory
fn read<T: Copy>(file: &mut std::fs::File, len: usize) -> std::io::Result<Vec<T>> {
    let mut pool = Vec::<T>::with_capacity(len);

    // SAFETY: read_exact fills all len elements before set_len, and any bit pattern is
    // a valid value for these pools. the materials inside bricks / summaries only ever
    // came from a Material, the source hash guarantees the same code wrote them
    unsafe {
        let raw =
            std::slice::from_raw_parts_mut(pool.as_mut_ptr() as *mut u8, len * size_of::<T>());
        file.read_exact(raw)?;
        pool.set_len(len);
    }

    Ok(pool)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_world_cache() {
        let tree = Tree64::build(1337);
        let dir = std::env::temp_dir().join(format!("voxels_cache_{}", std::process::id()));
        let path = dir.join("world.bin");

        tree.save(&path, 1337).unwrap();

        // everything comes back exactly //

        let loaded = Tree64::load(&path, 1337)
            .unwrap()
            .expect("same seed and code");
        assert_eq!(bytes(&loaded.branches), bytes(&tree.branches));
        assert_eq!(bytes(&loaded.leaves), bytes(&tree.leaves));
        assert_eq!(bytes(&loaded.bricks), bytes(&tree.bricks));
        assert_eq!(loaded.leaf_blocks, tree.leaf_blocks);
        assert_eq!(
            bytes(&loaded.leaf_block_materials),
            bytes(&tree.leaf_block_materials)
        );
        assert_eq!(loaded.water_leaves, tree.water_leaves);
        assert_eq!(loaded.skip, tree.skip);
        assert_eq!(loaded.bmax.ys.to_array(), tree.bmax.ys.to_array());

        for i in 0..2000u32 {
            let [x, y, z] = [i * 7919 % 2048, i * 104_729 % 1024, i * 1_299_709 % 2048];
            assert_eq!(loaded.voxel_at(x, y, z), tree.voxel_at(x, y, z));
        }

        // another seed, other code or a cut off file don't count //

        assert!(Tree64::load(&path, 42).unwrap().is_none());

        let mut file = std::fs::read(&path).unwrap();
        file[4] ^= 1;
        std::fs::write(&path, &file).unwrap();
        assert!(Tree64::load(&path, 1337).unwrap().is_none());

        file[4] ^= 1;
        file.truncate(file.len() - 1);
        std::fs::write(&path, &file).unwrap();
        assert!(Tree64::load(&path, 1337).unwrap().is_none());

        let _ = std::fs::remove_dir_all(dir);
    }
}
