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
    /// built from 4 origin vectors and 4 direction vectors
    pub fn from_vecs(origins: [Vec3; 4], directions: [Vec3; 4]) -> Self {
        let origins_x: [f32; 4] = [origins[0].x, origins[1].x, origins[2].x, origins[3].x];
        let origins_y: [f32; 4] = [origins[0].y, origins[1].y, origins[2].y, origins[3].y];
        let origins_z: [f32; 4] = [origins[0].z, origins[1].z, origins[2].z, origins[3].z];

        let origins = Packed_Vec3_x4::new(origins_x, origins_y, origins_z);

        let directions_x: [f32; 4] = [
            directions[0].x,
            directions[1].x,
            directions[2].x,
            directions[3].x,
        ];

        let directions_y: [f32; 4] = [
            directions[0].y,
            directions[1].y,
            directions[2].y,
            directions[3].y,
        ];

        let directions_z: [f32; 4] = [
            directions[0].z,
            directions[1].z,
            directions[2].z,
            directions[3].z,
        ];

        let directions = Packed_Vec3_x4::new(directions_x, directions_y, directions_z);

        let mask_x = directions.xs.is_eq_to(SimdF32::ZEROES);
        let dxs = SimdF32::select(mask_x, SimdF32::TINY, directions.xs);

        let mask_y = directions.ys.is_eq_to(SimdF32::ZEROES);
        let dys = SimdF32::select(mask_y, SimdF32::TINY, directions.ys);

        let mask_z = directions.zs.is_eq_to(SimdF32::ZEROES);
        let dzs = SimdF32::select(mask_z, SimdF32::TINY, directions.zs);

        let inverse_dirs = Packed_Vec3_x4 {
            xs: dxs.recip(),
            ys: dys.recip(),
            zs: dzs.recip(),
        };

        Self {
            origins,
            directions,
            inverse_dirs,
        }
    }

    /// built from the pixel origin and direction as well as offsets
    pub fn from_pixel(
        pixel00_center: Vec3,
        camera_origin: Vec3,
        pixel_delta_x: Vec3,
        pixel_delta_y: Vec3,
    ) -> Self {
        let origins = Packed_Vec3_x4::from_single(camera_origin);

        //    pixels corresponding to tile in packet
        //    A B
        //    C D
        let a = pixel00_center;
        let b = a + pixel_delta_x;
        let c = a - pixel_delta_y;
        let d = b - pixel_delta_y;

        let xs = SimdF32::from([a.x, b.x, c.x, d.x]) - origins.xs;
        let ys = SimdF32::from([a.y, b.y, c.y, d.y]) - origins.ys;
        let zs = SimdF32::from([a.z, b.z, c.z, d.z]) - origins.zs;

        let length_squared = xs.mul_then_add(xs, ys.mul_then_add(ys, SimdF32::mul(zs, zs)));

        let inv_length = length_squared.inv_sqrt();

        let directions = Packed_Vec3_x4 {
            xs: xs * inv_length,
            ys: ys * inv_length,
            zs: zs * inv_length,
        };

        let mask_x = directions.xs.is_eq_to(SimdF32::ZEROES);
        let dxs = SimdF32::select(mask_x, SimdF32::TINY, directions.xs);

        let mask_y = directions.ys.is_eq_to(SimdF32::ZEROES);
        let dys = SimdF32::select(mask_y, SimdF32::TINY, directions.ys);

        let mask_z = directions.zs.is_eq_to(SimdF32::ZEROES);
        let dzs = SimdF32::select(mask_z, SimdF32::TINY, directions.zs);

        let inverse_dirs = Packed_Vec3_x4 {
            xs: dxs.recip(),
            ys: dys.recip(),
            zs: dzs.recip(),
        };

        Self {
            origins,
            directions,
            inverse_dirs,
        }
    }

    #[inline(always)]
    pub fn at(&self, t: SimdF32) -> Packed_Vec3_x4 {
        self.directions
            .mul_then_add(Packed_Vec3_x4::stretched(t), self.origins)
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
    pub fn make(&self, pixel00: Vec3) -> Packet {
        let stretched = Packed_Vec3_x4::from_single(pixel00);

        let xs = stretched.xs + self.builder.xs;
        let ys = stretched.ys + self.builder.ys;
        let zs = stretched.zs + self.builder.zs;

        let directions = Packed_Vec3_x4 { xs, ys, zs };

        let length_squared = xs.mul_then_add(xs, ys.mul_then_add(ys, SimdF32::mul(zs, zs)));

        let inv_length = length_squared.inv_sqrt();

        let directions = Packed_Vec3_x4 {
            xs: xs * inv_length,
            ys: ys * inv_length,
            zs: zs * inv_length,
        };

        let mask_x = directions.xs.is_eq_to(SimdF32::ZEROES);
        let dxs = SimdF32::select(mask_x, SimdF32::TINY, directions.xs);

        let mask_y = directions.ys.is_eq_to(SimdF32::ZEROES);
        let dys = SimdF32::select(mask_y, SimdF32::TINY, directions.ys);

        let mask_z = directions.zs.is_eq_to(SimdF32::ZEROES);
        let dzs = SimdF32::select(mask_z, SimdF32::TINY, directions.zs);

        let inverse_dirs = Packed_Vec3_x4 {
            xs: dxs.recip(),
            ys: dys.recip(),
            zs: dzs.recip(),
        };

        Packet {
            origins: self.origin,
            directions,
            inverse_dirs,
        }
    }
}
