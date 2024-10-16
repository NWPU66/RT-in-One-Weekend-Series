#ifndef BOXH
#define BOXH

#include <cmath>
#include <cstdlib>
#include <curand_kernel.h>

#include "aabb.h"
#include "hitable.h"
#include "hitable_list.h"
#include "ray.h"
#include "rectangle.h"
#include "texture.h"
#include "vec3.h"

class box : public hitable {
public:
    __device__ box(const vec3& p0, const vec3& p1, material* ptr);
    __device__ ~box() { delete sides; }

    __device__ bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const override;

    __device__ bool bounding_box(float t0, float t1, aabb& output_box) const override
    {
        output_box = aabb(box_min, box_max);
        return true;
    }

    __device__ float sdf(const vec3& p, float time) const override;

    __device__ bool is_inside(const vec3& p) const
    {
        return (p.x() >= box_min.x() && p.x() <= box_max.x() && p.y() >= box_min.y() &&
                p.y() <= box_max.y() && p.z() >= box_min.z() && p.z() <= box_max.z());
    }

    __device__ vec3 center() const { return (box_min + box_max) / 2; }

public:
    vec3          box_min;
    vec3          box_max;
    hitable_list* sides;
};

__device__ box::box(const vec3& p0, const vec3& p1, material* ptr) : box_min(p0), box_max(p1)
{
    auto** list = (hitable**)malloc(6 * sizeof(hitable*));

    int i     = 0;
    list[i++] = new xy_rect(p0.x(), p1.x(), p0.y(), p1.y(), p1.z(), ptr);
    list[i++] = new flip_face(new xy_rect(p0.x(), p1.x(), p0.y(), p1.y(), p0.z(), ptr));

    list[i++] = new xz_rect(p0.x(), p1.x(), p0.z(), p1.z(), p1.y(), ptr);
    list[i++] = new flip_face(new xz_rect(p0.x(), p1.x(), p0.z(), p1.z(), p0.y(), ptr));

    list[i++] = new yz_rect(p0.y(), p1.y(), p0.z(), p1.z(), p1.x(), ptr);
    list[i++] = new flip_face(new yz_rect(p0.y(), p1.y(), p0.z(), p1.z(), p0.x(), ptr));

    sides = new hitable_list(list, i);
}

__device__ bool box::hit(const ray& r, float t_min, float t_max, hit_record& rec) const
{
    return sides->hit(r, t_min, t_max, rec);
}

__device__ float box::sdf(const vec3& p, float time) const
{
    vec3 box_local_p = (p - center()).abs();
    vec3 half_size   = (box_max - box_min) / 2.0f;

    if (box_local_p < half_size)  // inside the box
    {
        return (box_local_p - half_size).abs().min_value();
    }

    // outside the box
    return max(box_local_p - half_size, vec3(0)).length();
}

#endif
