#pragma once

#include "glm/glm.hpp"
#include <glm/exponential.hpp>
#include <glm/geometric.hpp>

#include "hittable.h"
#include "marco.h"

class sphere : public hittable {
public:
    sphere() = default;
    sphere(vec3 cen, double r) : center(cen), radius(r) {};

    virtual bool hit(const ray& r, double tmin, double tmax, hit_record& rec) const;

public:
    vec3   center;
    double radius;
};

bool sphere::hit(const ray& r, double t_min, double t_max, hit_record& rec) const
{
    vec3 oc           = r.origin() - center;
    auto a            = dot(r.direction(), r.direction());
    auto b            = 2.0 * dot(oc, r.direction());
    auto c            = dot(oc, oc) - radius * radius;
    auto discriminant = b * b - 4 * a * c;

    if (discriminant > 0)
    {
        double root = glm::sqrt(discriminant);
        double t    = (-b - root) / (2 * a);
        if (t < t_max && t > t_min)
        {
            rec.t               = t;
            rec.p               = r.at(t);
            vec3 outward_normal = glm::normalize(rec.p - center);
            rec.set_face_normal(r, outward_normal);
            return true;
        }
        t = (-b + root) / (2 * a);
        if (t < t_max && t > t_min)
        {
            rec.t               = t;
            rec.p               = r.at(t);
            vec3 outward_normal = glm::normalize(rec.p - center);
            rec.set_face_normal(r, outward_normal);
            return true;
        }
    }
    return false;
}
