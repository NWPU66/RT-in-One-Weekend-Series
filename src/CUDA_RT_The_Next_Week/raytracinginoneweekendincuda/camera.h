#ifndef CAMERAH
#define CAMERAH

#include "ray.h"
#include "util.h"
#include "vec3.h"

#include <cstdio>
#include <curand_kernel.h>
#include <curand_uniform.h>

#ifndef M_PI
#    define M_PI 3.14159265358979323846
#endif

__device__ vec3 random_in_unit_disk(curandState* local_rand_state)
{
    vec3 p;
    do
    {
        p = 2.0f * vec3(curand_uniform(local_rand_state), curand_uniform(local_rand_state), 0) -
            vec3(1, 1, 0);
    } while (dot(p, p) >= 1.0f);
    return p;
}

class camera {
public:
    __device__ camera(vec3  lookfrom,
                      vec3  lookat,
                      vec3  vup,
                      float vfov,
                      float aspect,
                      float aperture,
                      float focus_dist,
                      float t0 = 0.0,
                      float t1 = 0.0)
    {
        // vfov is top to bottom in degrees
        lens_radius       = aperture / 2.0f;
        time0             = t0;
        time1             = t1;
        float theta       = vfov * ((float)M_PI) / 180.0f;
        float half_height = tan(theta / 2.0f);
        float half_width  = aspect * half_height;
        origin            = lookfrom;
        w                 = unit_vector(lookfrom - lookat);
        u                 = unit_vector(cross(vup, w));
        v                 = cross(w, u);
        lower_left_corner =
            origin - half_width * focus_dist * u - half_height * focus_dist * v - focus_dist * w;
        horizontal = 2.0f * half_width * focus_dist * u;
        vertical   = 2.0f * half_height * focus_dist * v;
    }

    __device__ ray get_ray(float s, float t, curandState* local_rand_state)
    {
        vec3 rd     = lens_radius * random_in_unit_disk(local_rand_state);
        vec3 offset = u * rd.x() + v * rd.y();
        return ray(origin + offset,
                   lower_left_corner + s * horizontal + t * vertical - origin - offset,
                   mapping<float>(curand_uniform(local_rand_state), time0, time1));
    }

    __device__ vec3 CameraLookDir() { return w; }

    __device__ vec3 CameraOrigin() { return origin; }

    vec3  origin;
    vec3  lower_left_corner;
    vec3  horizontal;
    vec3  vertical;
    vec3  u, v, w;
    float lens_radius;
    float time0, time1;
};

#ifdef USE_GLM

class MovingCamera : public camera {
public:
    __device__ MovingCamera(vec3  lookfrom0,
                            vec3  lookfrom1,
                            vec3  lookat0,
                            vec3  lookat1,
                            vec3  vup,
                            float vfov,
                            float aspect,
                            float aperture,
                            float focus_dist,
                            float maxZDepth,
                            float t0 = 0.0,
                            float t1 = 0.0)
        : camera(lookfrom0, lookat0, vup, vfov, aspect, aperture, focus_dist, t0, t1),
          lookfrom0(lookfrom0), lookfrom1(lookfrom1), lookat0(lookat0), lookat1(lookat1),
          lookDir0(unit_vector(lookat0 - lookfrom0)), lookDir1(unit_vector(lookat1 - lookfrom1)),
          vfov(vfov), aspect(aspect), vup(vup), focus_dist(focus_dist), maxZDepth(maxZDepth)
    {
        printf("create camera 1\n");
        UpdateUVW(0);
        printf("create camera 2\n");
    }

    __host__ void SetCameraTime(float time) { UpdateUVW(time); }

    __host__ __device__ mat4 GetMVP_T() { return matrix4(MVP_T); }
    __host__ __device__ mat4 GetPrevMVP_T() { return matrix4(prevMVP_T); }

private:
    vec3 lookfrom0;
    vec3 lookfrom1;
    vec3 lookat0;
    vec3 lookat1;
    vec3 lookDir0;
    vec3 lookDir1;

    vec3      origin;
    vec3      lower_left_corner;
    vec3      horizontal;
    vec3      vertical;
    vec3      u, v, w;
    glm::mat4 MVP_T;
    glm::mat4 prevMVP_T;

    float lens_radius;
    float time0, time1;
    float vfov;
    float aspect;
    vec3  vup;
    float focus_dist;
    float maxZDepth;

    __host__ __device__ void UpdateUVW(float time)
    {
        vec3 lookfrom = (1 - time) * lookfrom0 + time * lookfrom1;
        vec3 lookat   = lookfrom + (1 - time) * lookDir0 + time * lookDir1;

        float theta       = vfov * ((float)M_PI) / 180.0f;
        float half_height = tan(theta / 2.0f);
        float half_width  = aspect * half_height; 
        origin            = lookfrom;
        w                 = unit_vector(lookfrom - lookat);
        u                 = unit_vector(cross(vup, w));
        v                 = cross(w, u);
        lower_left_corner =
            origin - half_width * focus_dist * u - half_height * focus_dist * v - focus_dist * w;
        horizontal = 2.0f * half_width * focus_dist * u;
        vertical   = 2.0f * half_height * focus_dist * v;

        // MVP
        printf("1\n");
        prevMVP_T = MVP_T;
        printf("2\n");
        MVP_T = glm::transpose(glm::perspective(theta, aspect, 0.1f, maxZDepth) *
                               glm::lookAt(vector3(origin), vector3(lookat), vector3(vup)) *
                               glm::mat4(1));
        printf("3\n");
    }
};

#endif

#endif
