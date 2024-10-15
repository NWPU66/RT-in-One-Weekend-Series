#ifndef BOXH
#define BOXH

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

class box_factory : public hitable_factory {};

#endif
