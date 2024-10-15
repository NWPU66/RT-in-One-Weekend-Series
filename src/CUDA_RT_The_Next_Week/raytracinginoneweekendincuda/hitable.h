#ifndef HITABLEH
#define HITABLEH

#include <cfloat>
#include <cmath>

#include <utility>

#include "aabb.h"
#include "ray.h"
#include "vec3.h"

#define PI 3.14159265358979323846

class material;

__device__ inline vec3 rotate_around_y(vec3 p, float angle, const vec3& center = {0})
{
    const float theta = angle * PI / 180.0;
    p -= center;

    float rotated_x = cos(theta) * p.x() + sin(theta) * p.z();
    float rotated_z = -sin(theta) * p.x() + cos(theta) * p.z();

    return vec3(rotated_x, p.y(), rotated_z) + center;
}

struct hit_record
{
    float     t;
    double    u, v;
    vec3      p;
    vec3      normal;
    material* mat_ptr;
};

class hitable {
public:
    __device__ virtual bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const = 0;
    __device__ virtual bool bounding_box(float t0, float t1, aabb& output_box) const           = 0;
};

class flip_face : public hitable {
public:
    __device__ flip_face(hitable* p) : ptr(p) {}
    __device__ ~flip_face() { delete ptr; }

    __device__ bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const override
    {
        if (!ptr->hit(r, t_min, t_max, rec)) { return false; }
        rec.normal = -rec.normal;
        return true;
    }

    __device__ bool bounding_box(float t0, float t1, aabb& output_box) const override
    {
        return ptr->bounding_box(t0, t1, output_box);
    }

public:
    hitable* ptr;
};

class translate : public hitable {
public:
    __device__ translate(hitable* p, const vec3& displacement) : ptr(p), offset(displacement) {}
    __device__ ~translate() { delete ptr; }

    __device__ bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const override;

    __device__ bool bounding_box(float t0, float t1, aabb& output_box) const override;

public:
    hitable* ptr;
    vec3     offset;
};

__device__ bool translate::hit(const ray& r, float t_min, float t_max, hit_record& rec) const
{
    ray moved_r(r.origin() - offset, r.direction(), r.time());
    if (!ptr->hit(moved_r, t_min, t_max, rec)) { return false; }
    rec.p += offset;
    return true;
}

__device__ bool translate::bounding_box(float t0, float t1, aabb& output_box) const
{
    if (!ptr->bounding_box(t0, t1, output_box)) { return false; }

    output_box = aabb(output_box.min() + offset, output_box.max() + offset);
    return true;
}

class rotate_y : public hitable {
public:
    __device__ rotate_y(hitable* p, float angle);
    __device__ ~rotate_y() { delete ptr; }

    __device__ bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const override;

    __device__ bool bounding_box(float t0, float t1, aabb& output_box) const override
    {
        output_box = bbox;
        return hasbox;
    }

public:
    hitable* ptr;
    float    angle;
    bool     hasbox;
    aabb     bbox;
};

__device__ rotate_y::rotate_y(hitable* p, float angle) : ptr(p), angle(angle)
{
    hasbox = ptr->bounding_box(0, 1, bbox);

    vec3 _min(-FLT_MAX), _max(FLT_MAX);
    for (int i = 0; i < 2; i++)
    {
        for (int j = 0; j < 2; j++)
        {
            for (int k = 0; k < 2; k++)
            {
                vec3 bbox_p{
                    i * bbox.max().x() + (1 - i) * bbox.min().x(),
                    j * bbox.max().y() + (1 - j) * bbox.min().y(),
                    k * bbox.max().z() + (1 - k) * bbox.min().z(),
                };

                vec3 rotated_p = rotate_around_y(bbox_p, angle, bbox.centrer());

                _min = min(_min, rotated_p);
                _max = max(_max, rotated_p);
            }
        }
    }
    bbox = aabb(_min, _max);
}

__device__ bool rotate_y::hit(const ray& r, float t_min, float t_max, hit_record& rec) const
{
    vec3 rotated_ori = rotate_around_y(r.origin(), -angle, bbox.centrer());
    vec3 rotated_dir = rotate_around_y(r.direction(), -angle);
    ray  rotated_r(rotated_ori, rotated_dir);

    if (!ptr->hit(rotated_r, t_min, t_max, rec)) { return false; }

    // correct the pos and the normal
    rec.p      = rotate_around_y(rec.p, angle, bbox.centrer());
    rec.normal = rotate_around_y(rec.normal, angle);

    return true;
}

class hitable_factory {
public:
    __host__ virtual hitable* create() const = 0;
    __host__ virtual ~hitable_factory() {}

private:
    __host__ virtual hitable* malloc() const {}
};

#endif
