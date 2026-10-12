use super::packetvec::Packed_Vec3_x4;
use super::simdf32::SimdF32;
use super::simdu32::SimdU32;
use crate::math::vec3::Vec3;

#[repr(C)]
#[derive(Copy, Clone)]
#[allow(non_camel_case_types)]
pub struct Packet {
    pub origins: Packed_Vec3_x4,
    pub directions: Packed_Vec3_x4,
    pub inverse_dirs: Packed_Vec3_x4,
}

impl Packet {
    /// built from packed origins and directions (directions must be normalized)
    /// zero direction components get a tiny stand in so the inverse never hits infinity
    #[inline(always)]
    pub fn new(origins: Packed_Vec3_x4, directions: Packed_Vec3_x4) -> Self {
        let guard = |d: SimdF32| SimdF32::select(d.is_eq_to(SimdF32::ZEROES), SimdF32::TINY, d);

        let inverse_dirs = Packed_Vec3_x4 {
            xs: guard(directions.xs).recip(),
            ys: guard(directions.ys).recip(),
            zs: guard(directions.zs).recip(),
        };

        Self {
            origins,
            directions,
            inverse_dirs,
        }
    }

    /// built from 4 origin vectors and 4 direction vectors
    #[cfg(test)]
    pub fn from_vecs(origins: [Vec3; 4], directions: [Vec3; 4]) -> Self {
        let lane = |v: [Vec3; 4], axis: usize| [v[0][axis], v[1][axis], v[2][axis], v[3][axis]];

        Self::new(
            Packed_Vec3_x4::new(lane(origins, 0), lane(origins, 1), lane(origins, 2)),
            Packed_Vec3_x4::new(
                lane(directions, 0),
                lane(directions, 1),
                lane(directions, 2),
            ),
        )
    }

    /// point along every ray at distance t
    #[inline(always)]
    pub fn at(&self, t: SimdF32) -> Packed_Vec3_x4 {
        self.origins.add_scaled(self.directions, t)
    }
}

/// helper struct to build ray tracing packets without constant duplication
/// every frame, camera origin and pixel deltas are constants
/// this groups them together to help make packet forming easier
#[derive(Clone, Copy)]
pub struct PacketBuilder {
    /// camera origin broadcast into 3 simd lanes
    origin: Packed_Vec3_x4,
    /// combined math to get the 2x2 pixel packet as directions
    builder: Packed_Vec3_x4,
}

impl PacketBuilder {
    pub fn new(cam_origin: Vec3, pixel_delta_x: Vec3, pixel_delta_y: Vec3) -> PacketBuilder {
        const XDELTA_MASK: SimdU32 = SimdU32::comptime([0, 1, 0, 1]);
        const YDELTA_MASK: SimdU32 = SimdU32::comptime([0, 0, 1, 1]);

        let origin = Packed_Vec3_x4::from_single(cam_origin);

        let xdelta = Packed_Vec3_x4::from_single(pixel_delta_x);
        let ydelta = Packed_Vec3_x4::from_single(pixel_delta_y);

        //    pixel packet layed out as
        //    A  B  C  D in memory
        //
        //    or conceptually:
        //    A B
        //    C D
        //
        //
        // pixel XA  no changes
        //       XB  + Xdelta.x
        //       XC  - Ydelta.x
        //       XD  + Xdelta.x - Ydelta.x
        //
        // pixel YA  no changes
        //       YB  + Xdelta.y
        //       YC  - Ydelta.y
        //       YD  + Xdelta.y - Ydelta.y
        //
        //
        // pixel ZA  no changes
        //       ZB  + Xdelta.z
        //       ZC  - Ydelta.z
        //       ZD  + Xdelta.z - Ydelta.z
        //
        // subtract origin from all
        //
        // precompute them as
        //
        // A -> -origin
        // B -> -origin + Xdelta
        // C -> -origin - Ydelta
        // D -> -origin + Xdelta - Ydelta

        let xdelta_masked = Packed_Vec3_x4 {
            xs: SimdF32::select(XDELTA_MASK, xdelta.xs, SimdF32::ZEROES),
            ys: SimdF32::select(XDELTA_MASK, xdelta.ys, SimdF32::ZEROES),
            zs: SimdF32::select(XDELTA_MASK, xdelta.zs, SimdF32::ZEROES),
        };

        let ydelta_masked = Packed_Vec3_x4 {
            xs: SimdF32::select(YDELTA_MASK, ydelta.xs, SimdF32::ZEROES),
            ys: SimdF32::select(YDELTA_MASK, ydelta.ys, SimdF32::ZEROES),
            zs: SimdF32::select(YDELTA_MASK, ydelta.zs, SimdF32::ZEROES),
        };

        let builder = Packed_Vec3_x4 {
            xs: xdelta_masked.xs - ydelta_masked.xs - origin.xs,
            ys: xdelta_masked.ys - ydelta_masked.ys - origin.ys,
            zs: xdelta_masked.zs - ydelta_masked.zs - origin.zs,
        };

        Self { origin, builder }
    }

    /// yields the 2x2 ray packet from the top left pixel
    #[inline(always)]
    pub fn make(&self, pixel00: Vec3) -> Packet {
        let stretched = Packed_Vec3_x4::from_single(pixel00);

        let directions = Packed_Vec3_x4 {
            xs: stretched.xs + self.builder.xs,
            ys: stretched.ys + self.builder.ys,
            zs: stretched.zs + self.builder.zs,
        };

        Packet::new(self.origin, directions.normalize())
    }
}
