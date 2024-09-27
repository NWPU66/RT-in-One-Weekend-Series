#ifndef AABBH
#define AABBH

#include <cmath>
#include <utility>

#include "ray.h"
#include "vec3.h"

class aabb {
public:
    __device__ aabb() {}
    __device__ aabb(const vec3& a, const vec3& b) : _min(a), _max(b) {}

    __device__ vec3 min() const { return _min; }
    __device__ vec3 max() const { return _max; }

    __device__ bool hit(const ray& r, float tmin, float tmax) const
    {
        // Andrew Kensler's hit method
        for (int i = 0; i < 3; i++)
        {
            auto invD = 1.0f / r.direction()[i];
            auto t0   = (min()[i] - r.origin()[i]) * invD;
            auto t1   = (max()[i] - r.origin()[i]) * invD;

            if (invD < 0.0f)
            {
                auto temp = t0;
                t0        = t1;
                t1        = temp;
            }

            tmin = t0 > tmin ? t0 : tmin;
            tmax = t1 < tmax ? t1 : tmax;

            if (tmax <= tmin) { return false; }
        }
        return true;
    }

    vec3 _min;
    vec3 _max;
};

__device__ aabb surrounding_box(aabb box0, aabb box1)
{
    vec3 small(fmin(box0.min().x(), box1.min().x()), fmin(box0.min().y(), box1.min().y()),
               fmin(box0.min().z(), box1.min().z()));
    vec3 big(fmax(box0.max().x(), box1.max().x()), fmax(box0.max().y(), box1.max().y()),
             fmax(box0.max().z(), box1.max().z()));
    return aabb(small, big);
}

#endif