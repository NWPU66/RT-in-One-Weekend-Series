#ifndef PERLINH
#define PERLINH

// c
#include <cstdlib>

// c++
#include <algorithm>
#include <functional>
#include <iostream>

// 3rdparty
#include <curand_kernel.h>
#include <curand_normal.h>
#include <curand_uniform.h>
#include <tuple>

// users
#include "vec3.h"

#define RND (curand_uniform(&local_rand_state))
#define dRND (curand_uniform_double(&local_rand_state))

__device__ inline double trilinear_interp(double c[2][2][2], float u, float v, float w)
{
    u = u * u * (3 - 2 * u);
    v = v * v * (3 - 2 * v);
    w = w * w * (3 - 2 * w);

    auto accum = 0.0;
    for (int i = 0; i < 2; i++)
    {
        for (int j = 0; j < 2; j++)
        {
            for (int k = 0; k < 2; k++)
            {
                accum += (i * u + (1 - i) * (1 - u)) * (j * v + (1 - j) * (1 - v)) *
                         (k * w + (1 - k) * (1 - w)) * c[i][j][k];
            }
        }
    }
    return accum;
}

__device__ inline double trilinear_interp(vec3 c[2][2][2], float u, float v, float w)
{
    auto uu = u * u * (3 - 2 * u);
    auto vv = v * v * (3 - 2 * v);
    auto ww = w * w * (3 - 2 * w);

    auto accum = 0.0;
    for (int i = 0; i < 2; i++)
    {
        for (int j = 0; j < 2; j++)
        {
            for (int k = 0; k < 2; k++)
            {
                vec3 weight_v(u - i, v - j, w - k);
                accum += (i * uu + (1 - i) * (1 - uu)) * (j * vv + (1 - j) * (1 - vv)) *
                         (k * ww + (1 - k) * (1 - ww)) * dot(c[i][j][k], weight_v);
            }
        }
    }
    return accum;
}

class perlin {
public:
    __device__ perlin(curandState* rand_state) : rand_state(rand_state)
    {
        if (rand_state == nullptr) { return; }

        ranvec = new vec3[point_count];
        for (int i = 0; i < point_count; i++)
        {
            ranvec[i] = {curand_uniform(rand_state) * 2 - 1, curand_uniform(rand_state) * 2 - 1,
                         curand_uniform(rand_state) * 2 - 1};
        }

        perm_x = perlin_generate_perm();
        perm_y = perlin_generate_perm();
        perm_z = perlin_generate_perm();
    }
    __device__ ~perlin() { delete[] ranvec, perm_x, perm_y, perm_z; }

    __device__ double noise(const vec3& p) const
    {
        auto u = p.x() - floor(p.x());
        auto v = p.y() - floor(p.y());
        auto w = p.z() - floor(p.z());

        // hermite cube插值
        u = u * u * (3 - 2 * u);
        v = v * v * (3 - 2 * v);
        w = w * w * (3 - 2 * w);

        int i = floor(p.x());
        int j = floor(p.y());
        int k = floor(p.z());

        vec3 c[2][2][2];
        for (int di = 0; di < 2; di++)
        {
            for (int dj = 0; dj < 2; dj++)
            {
                for (int dk = 0; dk < 2; dk++)
                {
                    c[di][dj][dk] = ranvec[perm_x[(i + di) & 255] ^ perm_y[(j + dj) & 255] ^
                                           perm_z[(k + dk) & 255]];
                }
            }
        }

        return trilinear_interp(c, u, v, w);
    }

    __device__ double turb(vec3 p, int depth = 5) const
    {
        auto accum  = 0.0;
        auto weight = 1.0;

        for (int i = 0; i < depth; i++)
        {
            accum += weight * noise(p);
            weight *= 0.5;
            p *= 2;
        }

        return fabs(accum);
    }

private:
    curandState*     rand_state;
    static const int point_count = 256;
    vec3*            ranvec;
    int*             perm_x;
    int*             perm_y;
    int*             perm_z;

    __device__ int* perlin_generate_perm() const
    {
        auto p = new int[point_count];
        for (int i = 0; i < perlin::point_count; i++)
        {
            p[i] = i;
        }
        permute(p, point_count);
        return p;
    }

    __device__ void permute(int* p, int n) const
    {
        if (rand_state == nullptr) { return; }

        for (int i = n - 1; i > 0; i--)
        {
            int target = (int)(curand_uniform(rand_state) * (i + 1));

            int tmp   = p[i];
            p[i]      = p[target];
            p[target] = tmp;
        }
    }
};

#endif