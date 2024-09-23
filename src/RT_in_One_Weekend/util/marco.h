#pragma once
#include "glm/glm.hpp"

#ifdef HIGH_PRECISION
using vec4 = glm::dvec4;
using vec3 = glm::dvec3;
using vec2 = glm::dvec2;
using mat4 = glm::dmat4;
#else  // low precision
using vec4 = glm::vec4;
using vec3 = glm::vec3;
using vec2 = glm::vec2;
using mat4 = glm::mat4;
#endif