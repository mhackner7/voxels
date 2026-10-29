use super::rendering::{face_of, lanes, packed};
use super::shading::*;
use super::*;
use crate::simd::packetray::Packet;
use crate::simd::packetvec::Packed_Vec3_x4;
use crate::simd::simdf32::SimdF32;
use crate::simd::simdu32::SimdU32;
use crate::world::tree64::PacketHit;
use crate::world::voxel::Material;
use lightcache::LightCache;

// turning hits into light: view (sky, water, fog), reflect, face_light + bounce

/// what a packet of camera rays sees, the surface light gets read from the face cache later
#[derive(Clone, Copy)]
pub(super) struct View {
    /// light that reached the eye without landing on a lit surface (sky, water, fog)
    pub(super) radiance: Packed_Vec3_x4,
    /// how much the surface reflects, already dimmed by any water in front of it
    pub(super) base: Packed_Vec3_x4,
    pub(super) points: Packed_Vec3_x4,
    pub(super) normals: Packed_Vec3_x4,
    /// distance from the surface to the eye (meters), sizes the light patch
    pub(super) distances: SimdF32,
    pub(super) lit: SimdU32,

    /// water lanes, how much of the reflection shows (fresnel), and the reflection ray
    pub(super) water: SimdU32,
    pub(super) fresnel: SimdF32,
    pub(super) mirror_origins: Packed_Vec3_x4,
    pub(super) mirror_dirs: Packed_Vec3_x4,
}

impl View {
    pub(super) const EMPTY: Self = Self {
        radiance: Packed_Vec3_x4::ZEROES,
        base: Packed_Vec3_x4::ZEROES,
        points: Packed_Vec3_x4::ZEROES,
        normals: Packed_Vec3_x4::ZEROES,
        distances: SimdF32::ZEROES,
        lit: SimdU32::ALL_INACTIVE,
        water: SimdU32::ALL_INACTIVE,
        fresnel: SimdF32::ZEROES,
        mirror_origins: Packed_Vec3_x4::ZEROES,
        mirror_dirs: Packed_Vec3_x4::ZEROES,
    };
}

impl Engine {
    // sky, water (reflection + refraction mixed by fresnel) and fog under water
    #[inline(always)]
    pub(super) fn view(
        &self,
        ray: Packet,
        surface: PacketHit,
        underwater: bool,
        cone: f32,
    ) -> View {
        let tree = &self.world.tree;
        let sun = &self.world.sun;
        let offset = SimdF32::from_float(SURFACE_OFFSET);

        let mut surface = surface;
        let mut ray = ray;
        let mut throughput = Packed_Vec3_x4::ONES;
        let mut traveled = surface.distances;

        let mut radiance = Packed_Vec3_x4::select(
            surface.hit,
            Packed_Vec3_x4::ZEROES,
            sky_with_sun(ray.directions, sun),
        );

        // under water everything's seen through the water //

        if underwater {
            throughput = water_transmission(surface.distances);
            radiance = (Packed_Vec3_x4::ONES - throughput) * water_scatter(sun);
        }

        /* WATER */

        let water = water_lanes(surface.materials) & surface.hit;
        let mut mirror_origins = Packed_Vec3_x4::ZEROES;
        let mut mirror_dirs = Packed_Vec3_x4::ZEROES;
        let mut mirror_share = SimdF32::ZEROES;

        if water.has_active() {
            // water from above is always its flat top, a lumped block's side would mirror like a wall
            let n = Packed_Vec3_x4::select(water, splat(Vec3::new(0.0, 1.0, 0.0)), surface.normals);
            let d = ray.directions;
            let p = ray.at(surface.distances);

            let cos_i = d.dot(n).neg().clamp_to(SimdF32::ZEROES, SimdF32::ONES);
            let k = SimdF32::ONES - cos_i;
            let k2 = k * k;
            let fresnel =
                (k2 * k2 * k).mul_then_add(SimdF32::from_float(0.98), SimdF32::from_float(0.02));

            // reflection gets traced later, far water doesn't need one per pixel //

            let mirrored = d.add_scaled(n, d.dot(n) * SimdF32::from_float(-2.0));
            mirror_origins = p.add_scaled(n, offset);
            mirror_dirs = mirrored;
            mirror_share = fresnel;

            // refraction into water (eta = 1 / 1.33), the bottom shows up to WATER_VISIBILITY //

            let eta = SimdF32::from_float(1.0 / 1.33);
            let sin2 = (SimdF32::ONES - cos_i * cos_i) * eta * eta;
            let cos_t = (SimdF32::ONES - sin2).max(SimdF32::ZEROES).sqrt();
            let bent = d
                .scale(eta)
                .add_scaled(n, eta.mul_then_add(cos_i, cos_t.neg()))
                .normalize();

            let under_ray = Packet::new(
                Packed_Vec3_x4::select(water, p.add_scaled(n, offset.neg()), ray.origins),
                Packed_Vec3_x4::select(water, bent, ray.directions),
            );
            let under = tree.trace::<false, true>(
                under_ray,
                water,
                SimdF32::from_float(WATER_VISIBILITY),
                cone,
            );

            // whatever light the water eats gets swapped for its own scattered color //

            let transmission = water_transmission(under.distances);
            let scatter = water_scatter(sun);

            let refracted = SimdF32::ONES - fresnel;

            radiance = Packed_Vec3_x4::select(
                water,
                ((Packed_Vec3_x4::ONES - transmission) * scatter).scale(refracted),
                radiance,
            );
            throughput = Packed_Vec3_x4::select(water, transmission.scale(refracted), throughput);

            ray = under_ray;
            traveled = SimdF32::select(water, traveled + under.distances, traveled);
            surface.hit = (surface.hit & water.not()) | under.hit;
            surface.distances = SimdF32::select(water, under.distances, surface.distances);
            surface.normals = Packed_Vec3_x4::select(water, under.normals, surface.normals);

            let water_array = water.to_array();
            let under_hits = under.hit.to_array();

            for (lane, material) in surface.materials.iter_mut().enumerate() {
                if water_array[lane] != 0 {
                    *material = if under_hits[lane] != 0 {
                        under.materials[lane]
                    } else {
                        Material::Air
                    };
                }
            }
        }

        let p = ray.at(surface.distances);

        View {
            radiance,
            base: albedo(surface.materials, p, surface.normals) * throughput,
            points: p,
            normals: surface.normals,
            distances: traveled,
            lit: surface.hit,
            water,
            fresnel: mirror_share,
            mirror_origins,
            mirror_dirs,
        }
    }

    /// what 4 water lanes see reflected, faces show their cached light, uncached ones just get the sun
    #[inline(always)]
    pub(super) fn reflect(&self, origins: Packed_Vec3_x4, dirs: Packed_Vec3_x4) -> Packed_Vec3_x4 {
        let sun = &self.world.sun;
        let ray = Packet::new(origins, dirs);
        let mirror = self.world.tree.trace::<false, true>(
            ray,
            SimdU32::ALL_ACTIVE,
            SimdF32::from_float(WATER_REFLECTION_DISTANCE),
            Self::CAMERA_CONE,
        );

        let n = mirror.normals;
        let p = ray.at(mirror.distances);
        let (cached, found) = self.cached_light(p, n, mirror.hit, 1);
        let guessed = splat(sun.rgb).scale(n.dot(splat(sun.dir)).max(SimdF32::ZEROES))
            + splat(BOUNCE_AMBIENT);

        Packed_Vec3_x4::select(
            mirror.hit,
            albedo(mirror.materials, p, n) * Packed_Vec3_x4::select(found, cached, guessed),
            sky_with_sun(dirs, sun),
        )
    }

    /// light landing on 4 face spots: sun with a shadow ray plus one cosine bounce, no surface color
    #[inline(always)]
    pub(super) fn face_light(
        &self,
        start: Packed_Vec3_x4,
        n: Packed_Vec3_x4,
        rng: &mut Rng,
    ) -> Packed_Vec3_x4 {
        let sun = &self.world.sun;

        let to_sun = sun_sample(sun, rng.next(), rng.next());
        let n_dot_l = n.dot(to_sun).max(SimdF32::ZEROES);
        let facing = n_dot_l.is_greater_than(SimdF32::ZEROES);

        let shadow = self.world.tree.trace::<true, true>(
            Packet::new(start, to_sun),
            facing,
            SimdF32::INFY,
            Self::SHADOW_CONE,
        );
        let sunlit = facing & shadow.hit.not();

        let direct = Packed_Vec3_x4::select(
            sunlit,
            splat(sun.rgb).scale(n_dot_l),
            Packed_Vec3_x4::ZEROES,
        );

        direct + self.bounce(start, n, rng)
    }

    /// one cosine bounce, a hit on a cached face just borrows its light so later bounces are free
    #[inline(always)]
    pub(super) fn bounce(
        &self,
        start: Packed_Vec3_x4,
        n: Packed_Vec3_x4,
        rng: &mut Rng,
    ) -> Packed_Vec3_x4 {
        let tree = &self.world.tree;
        let sun = &self.world.sun;

        let bounce_dirs = cosine_sample(n, rng.next(), rng.next());

        if !self.bounce_lighting {
            return sky(bounce_dirs);
        }

        let bounce_ray = Packet::new(start, bounce_dirs);
        let bounce = tree.trace::<false, true>(
            bounce_ray,
            SimdU32::ALL_ACTIVE,
            SimdF32::from_float(Self::BOUNCE_MAX_DISTANCE),
            Self::BOUNCE_CONE,
        );

        // escaped bounces pick up the sky dome, the sun was already counted as direct light //

        let indirect = sky(bounce_dirs);

        if bounce.hit.has_none_active() {
            return indirect;
        }

        let n2 = bounce.normals;
        let p2 = bounce_ray.at(bounce.distances);
        let (cached, found) = self.cached_light(p2, n2, bounce.hit, Self::BORROW_SAMPLES);

        let to_sun = sun_sample(sun, rng.next(), rng.next());
        let n2_dot_l = n2.dot(to_sun).max(SimdF32::ZEROES);
        let facing2 = bounce.hit & found.not() & n2_dot_l.is_greater_than(SimdF32::ZEROES);

        let shadow2 = tree.trace::<true, true>(
            Packet::new(
                p2.add_scaled(n2, SimdF32::from_float(SURFACE_OFFSET)),
                to_sun,
            ),
            facing2,
            SimdF32::from_float(Self::BOUNCE_SHADOW_DISTANCE),
            Self::BOUNCE_CONE,
        );
        let sunlit2 = facing2 & shadow2.hit.not();

        let guessed = Packed_Vec3_x4::select(
            sunlit2,
            splat(sun.rgb).scale(n2_dot_l),
            Packed_Vec3_x4::ZEROES,
        ) + splat(BOUNCE_AMBIENT);

        Packed_Vec3_x4::select(
            bounce.hit,
            albedo(bounce.materials, p2, n2) * Packed_Vec3_x4::select(found, cached, guessed),
            indirect,
        )
    }

    /// cached light for every hit lane's face, and the lanes with at least min_samples
    #[inline(always)]
    pub(super) fn cached_light(
        &self,
        points: Packed_Vec3_x4,
        normals: Packed_Vec3_x4,
        hit: SimdU32,
        min_samples: u32,
    ) -> (Packed_Vec3_x4, SimdU32) {
        let hits = hit.to_array();
        let points = lanes(points);
        let normals = lanes(normals);
        let mut found = [0u32; 4];

        let light = [0, 1, 2, 3].map(|lane| {
            if hits[lane] == 0 {
                return [0.0; 3];
            }

            let (voxel, face) = face_of(points[lane], normals[lane]);

            for level in 0..4 {
                if let Some((light, samples)) =
                    self.lights.lookup(LightCache::key(voxel, face, level))
                    && samples >= min_samples
                {
                    found[lane] = u32::MAX;
                    return light;
                }
            }

            [0.0; 3]
        });

        (packed(light), SimdU32::from(found))
    }
}
