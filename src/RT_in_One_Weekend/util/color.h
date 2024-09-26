#pragma once
#include <functional>
#include <glm/geometric.hpp>
#include <iostream>
#include <ostream>
#include <random>

#include "glm/glm.hpp"

#include "marco.h"

#define PI 3.14159265358979323846

void write_color(std::ostream& out, const vec3& color, int samples_per_pixel)
{
    constexpr double inverse_gamma = 1 / 2.2;

    auto scale = 1.0 / samples_per_pixel;
    auto r     = glm::pow(scale * color.x, inverse_gamma);
    auto g     = glm::pow(scale * color.y, inverse_gamma);
    auto b     = glm::pow(scale * color.z, inverse_gamma);

    out << static_cast<int>(256 * glm::clamp(r, 0.0, 0.999)) << ' '
        << static_cast<int>(256 * glm::clamp(g, 0.0, 0.999)) << ' '
        << static_cast<int>(256 * glm::clamp(b, 0.0, 0.999)) << '\n';
}

inline double random_double()
{
    static std::uniform_real_distribution<double> distribution(0.0, 1.0);
    static std::mt19937                           generator;
    static std::function<double()> rand_generator = std::bind(distribution, generator);
    return rand_generator();
}

inline vec3 random_vec3()
{
    return vec3(random_double(), random_double(), random_double());
}

inline vec3 random_in_unit_sphere()
{
    while (true)
    {
        auto p = random_vec3() * vec3(2) - vec3(1);
        if (glm::dot(p, p) < 1.0) { return p; }
    }
}

inline vec3 random_unit_vector()
{
    auto a = random_double() * 2 * PI;
    auto z = random_double() * 2 - 1;
    auto r = glm::sqrt(1 - glm::pow(z, 2));
    return vec3(r * glm::cos(a), r * glm::sin(a), z);
}

inline vec3 random_in_hemisphere(const vec3& normal)
{
    vec3 in_unit_sphere = random_in_unit_sphere();
    if (glm::dot(in_unit_sphere, normal) > 0.0)  // In the same hemisphere as the normal
    {
        return in_unit_sphere;
    }
    else { return -in_unit_sphere; }
}

inline vec3 random_in_unit_disk()
{
    while (true)
    {
        auto p = vec3(random_double() * 2 - 1, random_double() * 2 - 1, 0);
        if (glm::dot(p, p) < 1) { return p; }
    }
}

inline vec3 reflect(const vec3& v, const vec3& n)
{
    return v - 2 * dot(v, n) * n;
}

inline vec3 refract(const vec3& uv, const vec3& n, double etai_over_etat)
{
    auto cos_theta      = dot(-uv, n);
    vec3 r_out_parallel = vec3(etai_over_etat) * (uv + cos_theta * n);
    vec3 r_out_perp     = -vec3(sqrt(1.0 - glm::dot(r_out_parallel, r_out_parallel))) * n;
    return r_out_parallel + r_out_perp;
}

inline double schlick(double cosine, double ref_idx)
{
    auto r0 = (1 - ref_idx) / (1 + ref_idx);
    r0 *= r0;
    return r0 + (1 - r0) * pow((1 - cosine), 5);
}
