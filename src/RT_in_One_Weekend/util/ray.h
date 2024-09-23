#pragma once
#include "glm/glm.hpp"

#include "marco.h"

class ray {
public:
    ray() = default;
    ray(const vec3& origin, const vec3& direction) : orig(origin), dir(direction) {}

    vec3 origin() const { return orig; }
    vec3 direction() const { return dir; }

#ifdef HIGH_PRECISION
    vec3 at(double t) const { return orig + t * dir; }
#else  // low precision
    vec3 at(float t) const { return orig + t * dir; }
#endif

public:
    vec3 orig;
    vec3 dir;
};