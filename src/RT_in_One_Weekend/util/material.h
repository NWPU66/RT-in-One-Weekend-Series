#pragma once
#include <glm/common.hpp>
#include <glm/geometric.hpp>

#include "marco.h"

#include "color.h"
#include "ray.h"

class material {
public:
    virtual bool scatter(const ray& r_in,
                         vec3&      hit_position,
                         vec3&      normal,
                         bool       front_face,
                         vec3&      attenuation,
                         ray&       scattered) const = 0;
};

class lambertian : public material {
public:
    lambertian(const vec3& a) : albedo(a) {}

    virtual bool scatter(const ray& r_in,
                         vec3&      hit_position,
                         vec3&      normal,
                         bool       front_face,
                         vec3&      attenuation,
                         ray&       scattered) const
    {
        vec3 scatter_direction = normal + random_unit_vector();
        scattered              = ray(hit_position, scatter_direction);
        attenuation            = albedo;
        return true;
    }

public:
    vec3 albedo;
};

class metal : public material {
public:
    metal(const vec3& a, double f) : albedo(a), fuzz(glm::clamp(f, 0.0, 1.0)) {}

    virtual bool scatter(const ray& r_in,
                         vec3&      hit_position,
                         vec3&      normal,
                         bool       front_face,
                         vec3&      attenuation,
                         ray&       scattered) const
    {
        vec3 reflected = reflect(glm::normalize(r_in.direction()), normal);
        scattered      = ray(hit_position, reflected + vec3(fuzz) * random_in_unit_sphere());
        attenuation    = albedo;
        return (dot(scattered.direction(), normal) > 0);
    }

public:
    vec3   albedo;
    double fuzz;
};

class dielectric : public material {
public:
    dielectric(double ri) : ref_idx(ri) {}

    virtual bool scatter(const ray& r_in,
                         vec3&      hit_position,
                         vec3&      normal,
                         bool       front_face,
                         vec3&      attenuation,
                         ray&       scattered) const
    {
        attenuation           = vec3(1);
        double etai_over_etat = front_face ? 1 / ref_idx : ref_idx;

        vec3   unit_direction = glm::normalize(r_in.direction());
        double cos_theta      = fmin(dot(-unit_direction, normal), 1.0);
        double sin_theta      = sqrt(1.0 - cos_theta * cos_theta);
        double reflect_prob   = schlick(cos_theta, etai_over_etat);
        if (etai_over_etat * sin_theta > 1.0 || random_double() < reflect_prob)
        {
            // Must Reflect
            vec3 reflected = reflect(unit_direction, normal);
            scattered      = ray(hit_position, reflected);
        }
        else
        {
            // Can Refract
#ifdef HIGH_PRECISION
            vec3 refracted = glm::refract(unit_direction, normal, etai_over_etat);
#else  // low precision
            vec3 refracted = glm::refract(unit_direction, normal, (float)etai_over_etat);
#endif
            // vec3 refracted = refract(unit_direction, normal, etai_over_etat);
            scattered = ray(hit_position, refracted);
        }
        return true;
    }

public:
    double ref_idx;
};
