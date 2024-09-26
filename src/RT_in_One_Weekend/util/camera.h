#pragma once
#include <glm/geometric.hpp>
#include <glm/trigonometric.hpp>

#include "marco.h"

#include "color.h"
#include "ray.h"

class camera {
public:
    camera(vec3   lookfrom,
           vec3   lookat,
           vec3   vup,
           double vfov,
           double aspect,
           double aperture,
           double focus_dist)
    {
        origin      = lookfrom;
        lens_radius = aperture / 2;

        auto theta       = glm::radians(vfov);
        auto half_height = glm::tan(theta / 2);  // 相机距离视平面 = 焦距
        auto half_width  = aspect * half_height;

        w = glm::normalize(lookfrom - lookat);
        u = glm::normalize(glm::cross(vup, w));
        v = glm::cross(w, u);

        lower_left_corner = origin - vec3(focus_dist) * w - vec3(half_height * focus_dist) * v -
                            vec3(half_width * focus_dist) * u;

        horizontal = vec3(2 * half_width * focus_dist) * u;
        vertical   = vec3(2 * half_height * focus_dist) * v;
    }

#ifdef HIGH_PRECISION
    ray get_ray(double u, double v)
    {
        vec3 rd     = vec3(lens_radius) * random_in_unit_disk();
        vec3 offset = u * vec3(rd.x) + v * vec3(rd.y);
        return ray(origin + offset,
                   lower_left_corner + u * horizontal + v * vertical - origin - offset);
    }
#else  // low precision
    ray get_ray(float u, float v)
    {
        vec3 rd     = vec3(lens_radius) * random_in_unit_disk();
        vec3 offset = u * vec3(rd.x) + v * vec3(rd.y);
        return ray(origin + offset,
                   lower_left_corner + u * horizontal + v * vertical - origin - offset);
    }
#endif

public:
    vec3   origin;
    vec3   lower_left_corner;
    vec3   horizontal;
    vec3   vertical;
    vec3   u, v, w;
    double lens_radius;
};
