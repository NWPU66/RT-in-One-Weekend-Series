#pragma once
#include "marco.h"

#include "ray.h"

class camera {
public:
    camera() : origin(0), lower_left_corner(-2, -1, -1), horizontal(4, 0, 0), vertical(0, 2, 0) {}

#ifdef HIGH_PRECISION
    ray get_ray(double u, double v)
    {
        return ray(origin, lower_left_corner + u * horizontal + v * vertical - origin);
    }
#else  // low precision
    ray get_ray(float u, float v)
    {
        return ray(origin, lower_left_corner + u * horizontal + v * vertical - origin);
    }
#endif

public:
    vec3 origin;
    vec3 lower_left_corner;
    vec3 horizontal;
    vec3 vertical;
};
