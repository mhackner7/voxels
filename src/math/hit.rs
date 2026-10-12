use crate::simd::simdu32::SimdU32;
use crate::simd::{packetvec::Packed_Vec3_x4, simdf32::SimdF32};

#[inline(always)]
/// packed simd aabb testing
/// ro - packed ray origin
/// inv_dir - inverse ray direction
/// bmin - bounding box starting coord
/// bmax - bounding box ending coordy
pub fn simd_aabb(
    ro: Packed_Vec3_x4,
    inv_dir: Packed_Vec3_x4,
    bmin: Packed_Vec3_x4,
    bmax: Packed_Vec3_x4,
) -> (SimdF32, SimdU32) {
    let xs1 = (bmin.xs - ro.xs) * inv_dir.xs;
    let xs2 = (bmax.xs - ro.xs) * inv_dir.xs;

    let ys1 = (bmin.ys - ro.ys) * inv_dir.ys;
    let ys2 = (bmax.ys - ro.ys) * inv_dir.ys;

    let zs1 = (bmin.zs - ro.zs) * inv_dir.zs;
    let zs2 = (bmax.zs - ro.zs) * inv_dir.zs;

    let mut tmin = xs1.min(xs2);
    let mut tmax = xs1.max(xs2);

    tmin = tmin.max(SimdF32::min(ys1, ys2));
    tmax = tmax.min(SimdF32::max(ys1, ys2));

    tmin = tmin.max(SimdF32::min(zs1, zs2));
    tmax = tmax.min(SimdF32::max(zs1, zs2));

    tmin = tmin.max(SimdF32::ZEROES);

    let tmask = tmax.is_greater_or_equal(tmin);

    let result = SimdF32::select(tmask, tmin, SimdF32::INFY);

    let mask = result.is_less_than(SimdF32::INFY);

    (result, mask)
}
