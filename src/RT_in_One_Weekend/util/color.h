#pragma once
#include <glm/common.hpp>
#include <ostream>

#include "glm/glm.hpp"

#include "marco.h"

void write_color(std::ostream& out, const vec3& color, int samples_per_pixel)
{
    auto scale = 1.0 / samples_per_pixel;
    auto r     = scale * color.x;
    auto g     = scale * color.y;
    auto b     = scale * color.z;

    out << static_cast<int>(256 * glm::clamp(r, 0.0, 0.999)) << ' '
        << static_cast<int>(256 * glm::clamp(g, 0.0, 0.999)) << ' '
        << static_cast<int>(256 * glm::clamp(b, 0.0, 0.999)) << '\n';
}
