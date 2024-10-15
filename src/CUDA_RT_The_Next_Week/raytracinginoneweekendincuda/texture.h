#ifndef TEXTUREH
#define TEXTUREH

#include <curand_kernel.h>
#include <curand_uniform.h>

#include "perlin.h"
#include "vec3.h"

class Texture {
public:
    __device__ virtual vec3 value(double u, double v, const vec3& p) const = 0;
};

class const_texture : public Texture {
public:
    __device__ const_texture() {}
    __device__ const_texture(const vec3& c) : color(c) {}

    __device__ virtual vec3 value(double u, double v, const vec3& p) const override
    {
        return color;
    }

public:
    vec3 color;
};

class checker_texture : public Texture {
public:
    __device__ checker_texture() {}
    __device__ checker_texture(Texture* e, Texture* o) : even(e), odd(o) {}

    __device__ virtual vec3 value(double u, double v, const vec3& p) const override
    {
        auto sines = sin(10 * p.x()) * sin(10 * p.y()) * sin(10 * p.z());
        if (sines < 0) { return odd->value(u, v, p); }
        else { return even->value(u, v, p); }
    }

    __device__ ~checker_texture()
    {
        delete even;
        delete odd;
    }

public:
    Texture *even, *odd;
};

class noise_texture : public Texture {
public:
    __device__ noise_texture(float scale, curandState* rand_state)
        : scale(scale), noise(new perlin(rand_state))
    {
    }
    __device__ ~noise_texture() { delete noise; }

    __device__ vec3 value(double u, double v, const vec3& p) const override
    {
        auto c = scale * p.z() + 10 * noise->turb(p);
        return vec3(1, 1, 1) * 0.5 * (1 + sin(c));
    }

private:
    float   scale;
    perlin* noise;
};

class image_texture : public Texture {
public:
    __device__ image_texture() {}
    __device__ image_texture(unsigned char* pixels, int A, int B) : data(pixels), nx(A), ny(B) {}
    __device__ ~image_texture() { delete data; }

    __device__ vec3 value(double u, double v, const vec3& p) const override
    {
        if (data == nullptr) { return vec3(0, 1, 1); }

        auto i = static_cast<int>((u)*nx);
        auto j = static_cast<int>((1 - v) * ny - 0.001);

        // clamp
        if (i < 0) { i = 0; }
        if (j < 0) { j = 0; }
        if (i > nx - 1) { i = nx - 1; }
        if (j > ny - 1) { j = ny - 1; }

        auto r = data[3 * i + 3 * j * nx + 0] / 255.0;
        auto g = data[3 * i + 3 * j * nx + 1] / 255.0;
        auto b = data[3 * i + 3 * j * nx + 2] / 255.0;

        return vec3(r, g, b);
    }

public:
    unsigned char* data;
    int            nx, ny;
};

#endif
