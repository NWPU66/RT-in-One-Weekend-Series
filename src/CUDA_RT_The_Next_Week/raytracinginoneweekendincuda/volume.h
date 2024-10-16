/**
 * @file volume.h
 * @author your name (you@domain.com)
 * @brief
 * @version 0.1
 * @date 2024-10-16
 *
 * @copyright Copyright (c) 2024
 *
 */

#ifndef VOLUMEH
#define VOLUMEH

// c
#include <cfloat>
#include <cmath>

// c++
#include <curand_kernel.h>
#include <curand_uniform.h>
#include <utility>

// 3rd party

// users
#include "aabb.h"
#include "hitable.h"
#include "material.h"
#include "ray.h"
#include "texture.h"
#include "vec3.h"

#define SURFACE_DISTANCE_THRESHOLD 1e-3

class constant_medium : public hitable {
public:
    __device__ constant_medium(hitable* b, double d, Texture* a, curandState* rand_state)
        : boundary(b), neg_inv_density(-1.0 / d), phase_function(new isotropic(a)),
          local_rand_state(rand_state)
    {
    }
    __device__ ~constant_medium()
    {
        delete boundary;
        delete phase_function;
    }

    __device__ bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const override;
    __device__ bool bounding_box(float t0, float t1, aabb& output_box) const override
    {
        return boundary->bounding_box(t0, t1, output_box);
    }

    __device__ float sdf(const vec3& p, float time) const override;

public:
    curandState* local_rand_state;
    hitable*     boundary;
    material*    phase_function;
    double       neg_inv_density;
};

__device__ bool constant_medium::hit(const ray& r, float t_min, float t_max, hit_record& rec) const
{
    hit_record rec1, rec2;
    if (!boundary->hit(r, -FLT_MAX, FLT_MAX, rec1)) { return false; }         // 第一交点
    if (!boundary->hit(r, rec1.t + 0.0001, FLT_MAX, rec2)) { return false; }  // 第二交点

    if (rec1.t < t_min) { rec1.t = t_min; }
    if (rec2.t > t_max) { rec2.t = t_max; }  // clamp
    if (rec1.t >= rec2.t) { return false; }
    if (rec1.t < 0) { rec1.t = 0; }  // clamp to itself

    const auto ray_length               = r.direction().length();
    const auto distance_inside_boundary = (rec2.t - rec1.t) * ray_length;
    const auto hit_distance = neg_inv_density * log(curand_uniform_double(local_rand_state));

    if (hit_distance > distance_inside_boundary) { return false; }

    rec.t       = rec1.t + hit_distance / ray_length;
    rec.p       = r.point_at_parameter(rec.t);
    rec.normal  = vec3(1, 0, 0);  // arbitrary
    rec.mat_ptr = phase_function;
    return true;
}

__device__ float constant_medium::sdf(const vec3& p, float time) const
{
    return boundary->sdf(p, time);
}

class SSS_volume : public hitable {
public:
    __device__ SSS_volume(hitable* dielectric, hitable* volume)
        : dielectric(dielectric), volume(volume)
    {
    }
    __device__ ~SSS_volume()
    {
        delete dielectric;
        delete volume;
    }

    __device__ bool  hit(const ray& r, float t_min, float t_max, hit_record& rec) const override;
    __device__ bool  bounding_box(float t0, float t1, aabb& output_box) const override;
    __device__ float sdf(const vec3& p, float time) const override;

public:
    hitable *dielectric, *volume;
};

__device__ bool SSS_volume::hit(const ray& r, float t_min, float t_max, hit_record& rec) const
{
    /**
     * @brief 先做一次hit，若击中，只有两种可能，从外部击中，或从内部击中
     * 外部击中对应dielectric，内部击中对应volume
     */
    hit_record dielectric_rec;
    bool       dielectric_hit = dielectric->hit(r, t_min, t_max, dielectric_rec);
    if (!dielectric_hit) { return false; }

    // 射线点乘真法向，计算内外
    if (dot(dielectric_rec.p - r.origin(), dielectric_rec.normal) < 0)
    {
        rec = dielectric_rec;
        return dielectric_hit;  // 外部击中
    }

    // 内部击中，但是不知道是否穿透
    hit_record volume_rec;
    bool       volume_hit = volume->hit(r, t_min, t_max, volume_rec);
    if (volume_hit)
    {
        // 内部击中
        rec = volume_rec;
        return volume_hit;
    }
    else
    {
        // 穿透
        rec = dielectric_rec;
        return dielectric_hit;  // 外部击中
    }
}

__device__ bool SSS_volume::bounding_box(float t0, float t1, aabb& output_box) const
{
    return dielectric->bounding_box(t0, t1, output_box);
}

__device__ float SSS_volume::sdf(const vec3& p, float time) const
{
    return dielectric->sdf(p, time);
}

#endif
