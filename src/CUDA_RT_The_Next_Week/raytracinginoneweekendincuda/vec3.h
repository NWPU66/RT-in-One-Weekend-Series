#ifndef VEC3H
#define VEC3H

#include <cmath>
#include <iostream>
#include <math.h>

class vec3 {
public:
    __host__ __device__ vec3() {}
    __host__ __device__ vec3(float e0)
    {
        e[0] = e0;
        e[1] = e0;
        e[2] = e0;
    }
    __host__ __device__ vec3(float e0, float e1, float e2)
    {
        e[0] = e0;
        e[1] = e1;
        e[2] = e2;
    }
    __host__ __device__ inline float x() const { return e[0]; }
    __host__ __device__ inline float y() const { return e[1]; }
    __host__ __device__ inline float z() const { return e[2]; }
    __host__ __device__ inline float r() const { return e[0]; }
    __host__ __device__ inline float g() const { return e[1]; }
    __host__ __device__ inline float b() const { return e[2]; }

    __host__ __device__ inline const vec3& operator+() const { return *this; }
    __host__ __device__ inline vec3        operator-() const { return vec3(-e[0], -e[1], -e[2]); }
    __host__ __device__ inline float       operator[](int i) const { return e[i]; }
    __host__ __device__ inline float&      operator[](int i) { return e[i]; };

    __host__ __device__ inline vec3& operator+=(const vec3& v2);
    __host__ __device__ inline vec3& operator-=(const vec3& v2);
    __host__ __device__ inline vec3& operator*=(const vec3& v2);
    __host__ __device__ inline vec3& operator/=(const vec3& v2);
    __host__ __device__ inline vec3& operator*=(const float t);
    __host__ __device__ inline vec3& operator/=(const float t);

    __host__ __device__ inline float length() const
    {
        return sqrt(e[0] * e[0] + e[1] * e[1] + e[2] * e[2]);
    }
    __host__ __device__ inline float squared_length() const
    {
        return e[0] * e[0] + e[1] * e[1] + e[2] * e[2];
    }
    __host__ __device__ inline vec3 floor() const
    {
        return {::floor(e[0]), ::floor(e[1]), ::floor(e[2])};
    }
    __host__ __device__ inline vec3 abs() const { return {::abs(e[0]), ::abs(e[1]), ::abs(e[2])}; }
    __host__ __device__ inline void make_unit_vector();
    __host__ __device__ inline vec3 gamma_correction() const;

    __host__ __device__ inline float min_value() const { return fmin(e[0], fmin(e[1], e[2])); }
    __host__ __device__ inline float max_value() const { return fmax(e[0], fmax(e[1], e[2])); }

    float             e[3];
    static const vec3 worldUp;
};

const vec3 vec3::worldUp = vec3(0, 1, 0);

inline std::istream& operator>>(std::istream& is, vec3& t)
{
    is >> t.e[0] >> t.e[1] >> t.e[2];
    return is;
}

inline std::ostream& operator<<(std::ostream& os, const vec3& t)
{
    os << (int)t.e[0] << " " << (int)t.e[1] << " " << (int)t.e[2];
    return os;
}

__host__ __device__ inline void vec3::make_unit_vector()
{
    float k = 1.0 / sqrt(e[0] * e[0] + e[1] * e[1] + e[2] * e[2]);
    e[0] *= k;
    e[1] *= k;
    e[2] *= k;
}

__host__ __device__ inline vec3 operator+(const vec3& v1, const vec3& v2)
{
    return vec3(v1.e[0] + v2.e[0], v1.e[1] + v2.e[1], v1.e[2] + v2.e[2]);
}

__host__ __device__ inline vec3 operator+(const vec3& v1, float v2)
{
    return vec3(v1.e[0] + v2, v1.e[1] + v2, v1.e[2] + v2);
}

__host__ __device__ inline vec3 operator-(const vec3& v1, const vec3& v2)
{
    return vec3(v1.e[0] - v2.e[0], v1.e[1] - v2.e[1], v1.e[2] - v2.e[2]);
}

__host__ __device__ inline vec3 operator*(const vec3& v1, const vec3& v2)
{
    return vec3(v1.e[0] * v2.e[0], v1.e[1] * v2.e[1], v1.e[2] * v2.e[2]);
}

__host__ __device__ inline vec3 operator/(const vec3& v1, const vec3& v2)
{
    return vec3(v1.e[0] / v2.e[0], v1.e[1] / v2.e[1], v1.e[2] / v2.e[2]);
}

__host__ __device__ inline vec3 operator*(float t, const vec3& v)
{
    return vec3(t * v.e[0], t * v.e[1], t * v.e[2]);
}

__host__ __device__ inline vec3 operator/(vec3 v, float t)
{
    return vec3(v.e[0] / t, v.e[1] / t, v.e[2] / t);
}

__host__ __device__ inline vec3 operator*(const vec3& v, float t)
{
    return vec3(t * v.e[0], t * v.e[1], t * v.e[2]);
}

__host__ __device__ inline float dot(const vec3& v1, const vec3& v2)
{
    return v1.e[0] * v2.e[0] + v1.e[1] * v2.e[1] + v1.e[2] * v2.e[2];
}

__host__ __device__ inline vec3 cross(const vec3& v1, const vec3& v2)
{
    return vec3((v1.e[1] * v2.e[2] - v1.e[2] * v2.e[1]), (-(v1.e[0] * v2.e[2] - v1.e[2] * v2.e[0])),
                (v1.e[0] * v2.e[1] - v1.e[1] * v2.e[0]));
}

__host__ __device__ inline vec3& vec3::operator+=(const vec3& v)
{
    e[0] += v.e[0];
    e[1] += v.e[1];
    e[2] += v.e[2];
    return *this;
}

__host__ __device__ inline vec3& vec3::operator*=(const vec3& v)
{
    e[0] *= v.e[0];
    e[1] *= v.e[1];
    e[2] *= v.e[2];
    return *this;
}

__host__ __device__ inline vec3& vec3::operator/=(const vec3& v)
{
    e[0] /= v.e[0];
    e[1] /= v.e[1];
    e[2] /= v.e[2];
    return *this;
}

__host__ __device__ inline vec3& vec3::operator-=(const vec3& v)
{
    e[0] -= v.e[0];
    e[1] -= v.e[1];
    e[2] -= v.e[2];
    return *this;
}

__host__ __device__ inline vec3& vec3::operator*=(const float t)
{
    e[0] *= t;
    e[1] *= t;
    e[2] *= t;
    return *this;
}

__host__ __device__ inline vec3& vec3::operator/=(const float t)
{
    float k = 1.0 / t;

    e[0] *= k;
    e[1] *= k;
    e[2] *= k;
    return *this;
}

__host__ __device__ inline bool operator<(const vec3& v1, const vec3& v2)
{
    return v1.e[0] < v2.e[0] && v1.e[1] < v2.e[1] && v1.e[2] < v2.e[2];
}

__host__ __device__ inline bool operator<=(const vec3& v1, const vec3& v2)
{
    return v1.e[0] <= v2.e[0] && v1.e[1] <= v2.e[1] && v1.e[2] <= v2.e[2];
}

__host__ __device__ inline bool operator>(const vec3& v1, const vec3& v2)
{
    return v1.e[0] > v2.e[0] && v1.e[1] > v2.e[1] && v1.e[2] > v2.e[2];
}

__host__ __device__ inline bool operator>=(const vec3& v1, const vec3& v2)
{
    return v1.e[0] >= v2.e[0] && v1.e[1] >= v2.e[1] && v1.e[2] >= v2.e[2];
}

__host__ __device__ inline vec3 unit_vector(vec3 v)
{
    return v / v.length();
}

__host__ __device__ inline vec3 vec3::gamma_correction() const
{
    const float inverse_gamma = 1.0f / 2.2f;
    return vec3(powf(e[0], inverse_gamma), powf(e[1], inverse_gamma), powf(e[2], inverse_gamma));
}

__host__ __device__ inline vec3 max(const vec3& a, const vec3& b)
{
    return vec3(fmax(a.e[0], b.e[0]), fmax(a.e[1], b.e[1]), fmax(a.e[2], b.e[2]));
}

__host__ __device__ inline vec3 min(const vec3& a, const vec3& b)
{
    return vec3(fmin(a.e[0], b.e[0]), fmin(a.e[1], b.e[1]), fmin(a.e[2], b.e[2]));
}

#endif
