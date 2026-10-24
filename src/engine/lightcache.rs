use std::sync::atomic::{AtomicU32, AtomicU64, Ordering::*};

/*
    FACE LIGHT CACHE

    the light landing on every voxel face that's been on screen, kept in world space,
    so moving the camera never throws it away. every pixel looking at a face just reads
    its light, only one lane a frame (the first one to ask) ever shades it.
    far away faces get lumped into 2x2 / 4x4 / 8x8 voxel patches (the level),
    so a patch still covers a couple pixels and shading it is worth it

    it's one flat open addressing table that never grows, a full probe run kicks out
    whatever face went the longest without being looked at. every frame touches it
    from every thread at once: only the lane that claimed a face writes it that frame,
    everyone else just reads, so plain relaxed atomics are enough (no locks)
*/

/// one face, 32 bytes
#[repr(C, align(32))]
struct Slot {
    /// LightCache::key, 0 = never used
    key: AtomicU64,
    /// mean light landing on the face (linear rgb, f32 bits), no surface color in it
    light: [AtomicU32; 3],
    /// generation << 16 | samples, an old generation counts as empty
    state: AtomicU32,
    /// last frame anything looked at the face
    stamp: AtomicU32,
}

/// what a lane gets told to do about its face
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Claim {
    /// shade it, it lives in slot and has samples so far (0 = brand new)
    Shade { slot: u32, samples: u32 },
    /// it already has light (or will by the time this tile reads it)
    Lit,
    /// brand new and another tile is shading it right now, so there's nothing to read yet
    Missing,
}

pub struct LightCache {
    slots: Vec<Slot>,
    /// slots.len() - 1
    mask: usize,
    /// 64 - log2(slots.len()), the hash keeps its top bits
    shift: u32,
    /// clear() bumps it, slots of older ones read as empty (16 bits)
    generation: u32,

    /// faces that have ever moved into an empty slot
    pub used: AtomicU32,
    /// face samples shaded since the last take_lit()
    pub lit: AtomicU32,
}

impl LightCache {
    /// how many slots (2^21 faces, 64MB)
    pub const SLOTS: usize = 1 << 21;

    /// slots looked at past where a key lands before giving up and evicting
    const PROBES: usize = 8;

    /// faces under this many samples get shaded every frame no matter the budget
    pub const YOUNG: u32 = 8;

    /// a face's mean stops counting up here, after that new samples just fade in
    /// (that's how edits and moving shadows get picked up eventually)
    pub const SAMPLES: u32 = 128;

    /// converged faces get shaded this many times less often than the budget says
    pub const SETTLED: u32 = 16;

    pub fn new(slots: usize) -> Self {
        assert!(slots.is_power_of_two());

        Self {
            slots: (0..slots)
                .map(|_| Slot {
                    key: AtomicU64::new(0),
                    light: [const { AtomicU32::new(0) }; 3],
                    state: AtomicU32::new(0),
                    stamp: AtomicU32::new(0),
                })
                .collect(),
            mask: slots - 1,
            shift: 64 - slots.trailing_zeros(),
            generation: 1,
            used: AtomicU32::new(0),
            lit: AtomicU32::new(0),
        }
    }

    /// key of a face patch: voxel coords (0..2047), face (axis * 2 + positive) and
    /// level, the patch is 2^level voxels wide so the voxel coords lose level bits
    #[inline(always)]
    pub fn key(voxel: [u32; 3], face: u8, level: u32) -> u64 {
        let [x, y, z] = voxel.map(|c| (c >> level) as u64);
        (x | (y << 11) | (z << 22) | ((face as u64) << 33) | ((level as u64) << 36)) + 1
    }

    /// slot where a key's probe run starts, fibonacci hashing
    #[inline(always)]
    fn home(&self, key: u64) -> usize {
        (key.wrapping_mul(0x9E37_79B9_7F4A_7C15) >> self.shift) as usize
    }

    /// samples a slot holds for the current generation
    #[inline(always)]
    fn samples(&self, slot: &Slot) -> u32 {
        let state = slot.state.load(Relaxed);

        if state >> 16 == self.generation {
            state & 0xFFFF
        } else {
            0
        }
    }

    /// mean light of a face and how many samples that is, None = no light yet
    #[inline(always)]
    pub fn lookup(&self, key: u64) -> Option<([f32; 3], u32)> {
        let home = self.home(key);

        for probe in 0..Self::PROBES {
            let slot = &self.slots[(home + probe) & self.mask];
            let found = slot.key.load(Acquire);

            if found == key {
                let samples = self.samples(slot);

                if samples == 0 {
                    return None;
                }

                return Some((
                    slot.light
                        .each_ref()
                        .map(|c| f32::from_bits(c.load(Relaxed))),
                    samples,
                ));
            }

            if found == 0 {
                return None;
            }
        }

        None
    }

    /*
        a lane looking at a face asks whether it should shade it this frame.
        the first lane of the frame to ask decides, everyone after just reads:
        young faces always get shaded, the rest when their turn comes around
        (every `every` frames, SETTLED times less once they've converged)
    */
    #[inline(always)]
    pub fn claim(&self, key: u64, frame: u32, every: u32) -> Claim {
        let home = self.home(key);
        let mut stalest = (0u32, usize::MAX, 0u64);

        for probe in 0..Self::PROBES {
            let index = (home + probe) & self.mask;
            let slot = &self.slots[index];
            let found = slot.key.load(Acquire);

            if found == key {
                return self.claim_slot(index, key, frame, every);
            }

            if found == 0 {
                // empty slot, it's ours unless someone beat us to it //

                match slot.key.compare_exchange(0, key, AcqRel, Acquire) {
                    Ok(_) => {
                        self.used.fetch_add(1, Relaxed);
                        return self.reset(index, frame);
                    }
                    Err(other) if other == key => {
                        return self.claim_slot(index, key, frame, every);
                    }
                    Err(_) => continue,
                }
            }

            let age = frame.wrapping_sub(slot.stamp.load(Relaxed));

            if age > stalest.0 {
                stalest = (age, index, found);
            }
        }

        // a full run, kick out whatever went unseen the longest //
        // (never something that got looked at this frame, it's on screen)

        let (age, index, old) = stalest;

        if age == 0 || index == usize::MAX {
            return Claim::Missing;
        }

        // zero the samples first, so anyone still reading the old face sees no light //
        self.slots[index].state.store(0, Relaxed);

        if self.slots[index]
            .key
            .compare_exchange(old, key, AcqRel, Relaxed)
            .is_ok()
        {
            self.reset(index, frame)
        } else {
            Claim::Missing
        }
    }

    /// a slot that just got a new face
    #[inline(always)]
    fn reset(&self, index: usize, frame: u32) -> Claim {
        let slot = &self.slots[index];
        slot.state.store(self.generation << 16, Relaxed);
        slot.stamp.store(frame, Relaxed);

        Claim::Shade {
            slot: index as u32,
            samples: 0,
        }
    }

    #[inline(always)]
    fn claim_slot(&self, index: usize, key: u64, frame: u32, every: u32) -> Claim {
        let slot = &self.slots[index];
        let samples = self.samples(slot);

        // somebody already asked this frame, they had the say //

        if slot.stamp.swap(frame, Relaxed) == frame {
            return if samples == 0 {
                Claim::Missing
            } else {
                Claim::Lit
            };
        }

        let period = if samples < Self::YOUNG {
            1
        } else if samples < Self::SAMPLES {
            every
        } else {
            every * Self::SETTLED
        };

        // spread the turns out over frames so it's not every face at once //
        let turn = ((key ^ (key >> 29)) as u32).wrapping_add(frame);

        if turn.is_multiple_of(period) {
            Claim::Shade {
                slot: index as u32,
                samples,
            }
        } else {
            Claim::Lit
        }
    }

    /// one more light sample into a claimed slot (only the claimer ever calls this)
    #[inline(always)]
    pub fn add(&self, slot: u32, light: [f32; 3]) {
        let slot = &self.slots[slot as usize];
        let samples = (self.samples(slot) + 1).min(Self::SAMPLES);
        let weight = 1.0 / samples as f32;

        for (c, &new) in slot.light.iter().zip(&light) {
            let old = if samples == 1 {
                0.0
            } else {
                f32::from_bits(c.load(Relaxed))
            };

            c.store((old + (new - old) * weight).to_bits(), Relaxed);
        }

        slot.state.store((self.generation << 16) | samples, Relaxed);
        self.lit.fetch_add(1, Relaxed);
    }

    /// forget every face (the sun moved, the lighting changed), costs nothing
    pub fn clear(&mut self) {
        self.generation = (self.generation + 1) & 0xFFFF;

        // wrapped all the way around, so really old slots could look current //

        if self.generation == 0 {
            for slot in &self.slots {
                slot.state.store(0, Relaxed);
            }

            self.generation = 1;
        }
    }

    /*
        something got built or broken, so the faces around lo..hi (voxels, inclusive)
        go back to being young: they keep their old light as a single sample and get
        shaded every frame till the new light takes over. every level gets it,
        a far patch overlapping the box is just as stale
    */
    pub fn invalidate(&self, lo: [i32; 3], hi: [i32; 3]) {
        for level in 0..4 {
            let [x0, y0, z0] = lo.map(|c| (c.clamp(0, 2047) as u32) >> level);
            let [x1, y1, z1] = hi.map(|c| (c.clamp(0, 2047) as u32) >> level);

            for z in z0..=z1 {
                for y in y0..=y1 {
                    for x in x0..=x1 {
                        for face in 0..6 {
                            let key = Self::key([x << level, y << level, z << level], face, level);
                            self.forget(key);
                        }
                    }
                }
            }
        }
    }

    /// drop a face down to one sample, if it's in here at all
    #[inline(always)]
    fn forget(&self, key: u64) {
        let home = self.home(key);

        for probe in 0..Self::PROBES {
            let slot = &self.slots[(home + probe) & self.mask];
            let found = slot.key.load(Relaxed);

            if found == key {
                if self.samples(slot) > 1 {
                    slot.state.store((self.generation << 16) | 1, Relaxed);
                }
                return;
            }

            if found == 0 {
                return;
            }
        }
    }

    /// face samples shaded since the last call, for the perf overlay
    pub fn take_lit(&self) -> u32 {
        self.lit.swap(0, Relaxed)
    }

    pub fn slots(&self) -> usize {
        self.slots.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_light_cache() {
        let mut cache = LightCache::new(1 << 10);
        let key = LightCache::key([100, 200, 300], 3, 0);

        // a new face gets shaded once per frame, the second ask just waits on it //

        assert_eq!(cache.lookup(key), None);

        let Claim::Shade { slot, samples: 0 } = cache.claim(key, 1, 1) else {
            panic!("a new face gets shaded");
        };
        assert_eq!(cache.claim(key, 1, 1), Claim::Missing);
        assert_eq!(
            cache.claim(key, 2, 1),
            Claim::Shade { slot, samples: 0 },
            "young faces shade every frame"
        );

        cache.add(slot, [1.0, 2.0, 3.0]);
        cache.add(slot, [3.0, 2.0, 1.0]);
        assert_eq!(cache.lookup(key), Some(([2.0, 2.0, 2.0], 2)));
        assert_eq!(cache.claim(key, 2, 1), Claim::Lit);

        // a patch key covers all the voxels in it //

        assert_eq!(
            LightCache::key([96, 200, 300], 3, 3),
            LightCache::key([103, 207, 303], 3, 3)
        );
        assert_ne!(LightCache::key([96, 200, 300], 3, 3), key);

        // converged faces only come around every so often //

        for _ in 0..LightCache::SAMPLES {
            cache.add(slot, [2.0; 3]);
        }

        let shaded = (10..10 + 64)
            .filter(|&frame| matches!(cache.claim(key, frame, 4), Claim::Shade { .. }))
            .count();
        assert_eq!(shaded, 64 / (4 * LightCache::SETTLED) as usize);

        // invalidating around it makes it young again, light kept as one sample //

        cache.invalidate([90, 190, 290], [110, 210, 310]);
        assert_eq!(cache.lookup(key).map(|l| l.1), Some(1));

        let far = LightCache::key([500, 500, 500], 0, 0);
        let Claim::Shade { slot: far_slot, .. } = cache.claim(far, 100, 1) else {
            panic!();
        };
        cache.add(far_slot, [1.0; 3]);
        cache.add(far_slot, [1.0; 3]);
        cache.invalidate([90, 190, 290], [110, 210, 310]);
        assert_eq!(cache.lookup(far).map(|l| l.1), Some(2));

        // clear forgets everything //

        cache.clear();
        assert_eq!(cache.lookup(key), None);
        assert_eq!(cache.lookup(far), None);

        // a full table evicts the stalest face, never one seen this frame //

        let mut small = LightCache::new(8);

        for i in 0..8 {
            let k = LightCache::key([i, 0, 0], 0, 0);
            assert!(matches!(
                small.claim(k, i, 1),
                Claim::Shade { samples: 0, .. }
            ));
        }

        let newcomer = LightCache::key([50, 0, 0], 0, 0);
        assert!(matches!(
            small.claim(newcomer, 20, 1),
            Claim::Shade { samples: 0, .. }
        ));
        assert_eq!(small.lookup(LightCache::key([0, 0, 0], 0, 0)), None);

        small.clear();
        let other = LightCache::key([60, 0, 0], 0, 0);
        assert!(matches!(small.claim(other, 20, 1), Claim::Shade { .. }));
    }
}
