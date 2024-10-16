#ifndef HITABLELISTH
#define HITABLELISTH

#include "aabb.h"
#include "hitable.h"
#include <cfloat>
#include <cstdlib>

class hitable_list : public hitable {
public:
    __device__ hitable_list() {}
    __device__ hitable_list(hitable** l, int n)
    {
        list      = l;
        list_size = n;
    }
    __device__ virtual bool hit(const ray& r, float tmin, float tmax, hit_record& rec) const;
    __device__ virtual bool bounding_box(float t0, float t1, aabb& output_box) const;

    __device__ float sdf(const vec3& p, float time) const override;

    hitable** list;
    int       list_size;
};

__device__ bool hitable_list::hit(const ray& r, float t_min, float t_max, hit_record& rec) const
{
    hit_record temp_rec;
    bool       hit_anything   = false;
    float      closest_so_far = t_max;
    for (int i = 0; i < list_size; i++)
    {
        if (list[i]->hit(r, t_min, closest_so_far, temp_rec))
        {
            hit_anything   = true;
            closest_so_far = temp_rec.t;
            rec            = temp_rec;
        }
    }
    return hit_anything;
}

__device__ bool hitable_list::bounding_box(float t0, float t1, aabb& output_box) const
{
    if (list_size == 0) { return false; }

    aabb temp_box;
    bool first_box = true;

    for (int i = 0; i < list_size; i++)
    {
        if (!list[i]->bounding_box(t0, t1, temp_box)) { return false; }
        output_box = first_box ? temp_box : surrounding_box(temp_box, output_box);
        first_box  = false;
    }

    return true;
}

__device__ float hitable_list::sdf(const vec3& p, float time) const
{
    float min_sdf = FLT_MAX;
    for (int i = 0; i < list_size; i++)
    {
        float sdf = list[i]->sdf(p,time);
        if (abs(sdf) < abs(min_sdf)) { min_sdf = sdf; }
    }
    return min_sdf;
}

#endif
