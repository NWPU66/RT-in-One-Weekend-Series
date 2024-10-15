#ifndef SPHEREH
#define SPHEREH

#include <cmath>

#include "aabb.h"
#include "hitable.h"

#define PI 3.14159265358979323846

__device__ void get_sphere_uv(const vec3& p, double& u, double& v)
{
    auto phi   = atan2(p.z(), p.x());
    auto theta = asin(p.y());
    u          = 1 - (phi + PI) / (2 * PI);
    v          = (theta + PI / 2) / PI;
}

class sphere : public hitable {
public:
    __device__ sphere() {}
    __device__ sphere(vec3 cen, float r, material* m) : center(cen), radius(r), mat_ptr(m) {};
    __device__ ~sphere() { delete mat_ptr; }
    __device__ virtual bool hit(const ray& r, float tmin, float tmax, hit_record& rec) const;
    __device__ virtual bool bounding_box(float t0, float t1, aabb& output_box) const;

    vec3      center;
    float     radius;
    material* mat_ptr;
};

__device__ bool sphere::hit(const ray& r, float t_min, float t_max, hit_record& rec) const
{
    vec3  oc           = r.origin() - center;
    float a            = dot(r.direction(), r.direction());
    float b            = dot(oc, r.direction());
    float c            = dot(oc, oc) - radius * radius;
    float discriminant = b * b - a * c;
    if (discriminant > 0)
    {
        float temp = (-b - sqrt(discriminant)) / a;
        if (temp < t_max && temp > t_min)
        {
            rec.t       = temp;
            rec.p       = r.point_at_parameter(rec.t);
            rec.normal  = (rec.p - center) / radius;
            rec.mat_ptr = mat_ptr;
            get_sphere_uv(rec.normal, rec.u, rec.v);
            return true;
        }
        temp = (-b + sqrt(discriminant)) / a;
        if (temp < t_max && temp > t_min)
        {
            rec.t       = temp;
            rec.p       = r.point_at_parameter(rec.t);
            rec.normal  = (rec.p - center) / radius;
            rec.mat_ptr = mat_ptr;
            get_sphere_uv(rec.normal, rec.u, rec.v);
            return true;
        }
    }
    return false;
}

__device__ bool sphere::bounding_box(float t0, float t1, aabb& output_box) const
{
    output_box = aabb(center - vec3(abs(radius)), center + vec3(abs(radius)));
    return true;
}

#endif
