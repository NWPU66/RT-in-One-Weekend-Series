#ifndef MATERIALH
#define MATERIALH

#include <curand_kernel.h>

#include "hitable.h"
#include "ray.h"
#include "texture.h"
#include "vec3.h"

__device__ float schlick(float cosine, float ref_idx)
{
    float r0 = (1.0f - ref_idx) / (1.0f + ref_idx);
    r0       = r0 * r0;
    return r0 + (1.0f - r0) * pow((1.0f - cosine), 5.0f);
}

__device__ bool refract(const vec3& v, const vec3& n, float ni_over_nt, vec3& refracted)
{
    vec3  uv           = unit_vector(v);
    float dt           = dot(uv, n);
    float discriminant = 1.0f - ni_over_nt * ni_over_nt * (1 - dt * dt);
    if (discriminant > 0)
    {
        refracted = ni_over_nt * (uv - n * dt) - n * sqrt(discriminant);
        return true;
    }
    else
        return false;
}

#define RANDVEC3                                                                                   \
    vec3(curand_uniform(local_rand_state), curand_uniform(local_rand_state),                       \
         curand_uniform(local_rand_state))

__device__ vec3 random_in_unit_sphere(curandState* local_rand_state)
{
    vec3 p;
    do
    {
        p = 2.0f * RANDVEC3 - vec3(1, 1, 1);
    } while (p.squared_length() >= 1.0f);
    return p;
}

__device__ vec3 reflect(const vec3& v, const vec3& n)
{
    return v - 2.0f * dot(v, n) * n;
}

class material {
public:
    __device__ virtual bool scatter(const ray&        r_in,
                                    const hit_record& rec,
                                    vec3&             attenuation,
                                    ray&              scattered,
                                    curandState*      local_rand_state) const = 0;

    __device__ virtual vec3 emitted(double u, double v, const vec3& p) const { return vec3(0); }

    // __device__ virtual ~material() {}
    /**FIXME - ~material() = default 问题
     被显式申明为default的函数，在NVCC10版本后，会自动忽略__device__
     */
};

class lambertian : public material {
public:
    __device__              lambertian(Texture* a) : albedo(a) {}
    __device__              lambertian(const vec3& a) : albedo(new const_texture(a)) {}
    __device__ virtual bool scatter(const ray&        r_in,
                                    const hit_record& rec,
                                    vec3&             attenuation,
                                    ray&              scattered,
                                    curandState*      local_rand_state) const
    {
        vec3 normal =
            (dot(rec.normal, r_in.direction()) < 0) ? rec.normal : -rec.normal;  // double face

        vec3 target = rec.p + normal + random_in_unit_sphere(local_rand_state);
        scattered   = ray(rec.p, target - rec.p, r_in.time());
        attenuation = albedo->value(rec.u, rec.v, rec.p);
        return true;
    }

    __device__ virtual ~lambertian() { delete albedo; }

    Texture* albedo;
};

class metal : public material {
public:
    __device__ metal(const vec3& a, float f) : albedo(a)
    {
        if (f < 1)
            fuzz = f;
        else
            fuzz = 1;
    }
    __device__ virtual bool scatter(const ray&        r_in,
                                    const hit_record& rec,
                                    vec3&             attenuation,
                                    ray&              scattered,
                                    curandState*      local_rand_state) const
    {
        vec3 reflected = reflect(unit_vector(r_in.direction()), rec.normal);
        scattered =
            ray(rec.p, reflected + fuzz * random_in_unit_sphere(local_rand_state), r_in.time());
        attenuation = albedo;
        return (dot(scattered.direction(), rec.normal) > 0.0f);  // double face protection
    }

    __device__ virtual ~metal() {}

    vec3  albedo;
    float fuzz;
};

class dielectric : public material {
public:
    __device__              dielectric(float ri) : ref_idx(ri) {}
    __device__ virtual bool scatter(const ray&        r_in,
                                    const hit_record& rec,
                                    vec3&             attenuation,
                                    ray&              scattered,
                                    curandState*      local_rand_state) const
    {
        vec3  outward_normal;
        vec3  reflected = reflect(r_in.direction(), rec.normal);
        float ni_over_nt;
        attenuation = vec3(1.0, 1.0, 1.0);
        vec3  refracted;
        float reflect_prob;
        float cosine;
        if (dot(r_in.direction(), rec.normal) > 0.0f)
        {
            outward_normal = -rec.normal;
            ni_over_nt     = ref_idx;
            cosine         = dot(r_in.direction(), rec.normal) / r_in.direction().length();
            cosine         = sqrt(1.0f - ref_idx * ref_idx * (1 - cosine * cosine));
        }
        else
        {
            outward_normal = rec.normal;
            ni_over_nt     = 1.0f / ref_idx;
            cosine         = -dot(r_in.direction(), rec.normal) / r_in.direction().length();
        }
        if (refract(r_in.direction(), outward_normal, ni_over_nt, refracted))
            reflect_prob = schlick(cosine, ref_idx);
        else
            reflect_prob = 1.0f;
        if (curand_uniform(local_rand_state) < reflect_prob)
            scattered = ray(rec.p, reflected, r_in.time());
        else
            scattered = ray(rec.p, refracted, r_in.time());
        return true;
    }

    __device__ virtual ~dielectric() {}

    float ref_idx;
};

class diffuse_light : public material {
public:
    __device__ diffuse_light(Texture* a) : emit(a) {}
    __device__ ~diffuse_light() { delete emit; }

    __device__ virtual bool
    scatter(const ray&, const hit_record&, vec3&, ray&, curandState*) const override
    {
        return false;
    }

    __device__ virtual vec3 emitted(double u, double v, const vec3& p) const override
    {
        return emit->value(u, v, p);
    }

public:
    Texture* emit;
};

class isotropic : public material {
public:
    __device__ isotropic(Texture* a) : albedo(a) {}

    __device__ bool scatter(const ray&        r_in,
                            const hit_record& rec,
                            vec3&             attenuation,
                            ray&              scattered,
                            curandState*      local_rand_state) const override
    {
        scattered   = ray(rec.p, random_in_unit_sphere(local_rand_state), r_in.time());
        attenuation = albedo->value(rec.u, rec.v, rec.p);
        return true;
    }

    __device__ ~isotropic() { delete albedo; }

public:
    Texture* albedo;
};

#endif
