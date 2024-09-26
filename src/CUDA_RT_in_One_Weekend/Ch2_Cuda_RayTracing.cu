#include <cfloat>
#include <cstddef>
#include <cstdlib>

#include <chrono>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>

// cuda
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <curand_kernel.h>

#include "camera.cuh"
#include "hitable.cuh"
#include "hitable_list.cuh"
#include "ray.cuh"
#include "sphere.cuh"
#include "vec3.cuh"

#ifdef HIGH_PRECISION
using real = double;
// using vec3 = glm::dvec3;
#else  // low precision
using real = float;
// using vec3 = glm::vec3;
#endif

using timeType      = std::chrono::milliseconds;
const real REAL_MAX = std::numeric_limits<real>::max();

const std::string OUTPUT_FILE = "output.ppm";
const int         nx = 1200, ny = 600, ns = 100;
const int         tx = 8, ty = 8;
// 每个线程块上有8x8个线程，块上的线程数量是32的倍数
// 并且只有当一整个快上的线程都完成计算后，才会进行下一块的计算

#define checkCudaErrors(val) check_cuda((val), #val, __FILE__, __LINE__)
void check_cuda(cudaError_t result, char const* const func, const char* const file, const int line)
{
    if (result)
    {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " << file << ":"
                  << line << " '" << func << "' \n";
        // Make sure we call CUDA Device Reset before exiting
        cudaDeviceReset();
        exit(99);
    }
}

#define RANDVEC3                                                                                   \
    vec3(curand_uniform(local_rand_state), curand_uniform(local_rand_state),                       \
         curand_uniform(local_rand_state))
__device__ vec3 random_in_unit_sphere(curandState* local_rand_state)
{
    vec3 p;
    while (true)
    {
        p = RANDVEC3 * 2 - 1;
        if (p.squared_length() < 1.0f) { return p; }
    }
}

__device__ vec3 color(const ray& r, hitable** world, curandState* local_rand_state)
{
    ray        cur_ray = r;
    hit_record rec;
    real       cur_attenuation = 1.0f;
    for (int i = 0; i < 50; i++)
    {
        if ((*world)->hit(cur_ray, 0.0001, FLT_MAX, rec))
        {
            cur_attenuation *= 0.5f;
            cur_ray = ray(rec.p, rec.normal + random_in_unit_sphere(local_rand_state));
        }
        else
        {
            vec3 unit_direction = unit_vector(cur_ray.direction());
            real t              = 0.5f * (unit_direction.y() + 1.0f);
            vec3 c              = (1.0f - t) * vec3(1.0, 1.0, 1.0) + t * vec3(0.5, 0.7, 1.0);
            return cur_attenuation * c;
        }
    }

    return vec3(0);
}

__global__ void render_init(int max_x, int max_y, curandState* rand_state)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if ((i >= max_x) || (j >= max_y)) { return; }

    int pixel_index = j * max_x + i;

    // Each thread gets same seed, a different sequence number, no offset
    curand_init(3777, pixel_index, 0, &rand_state[pixel_index]);
}

__global__ void render(vec3*        fb,
                       int          max_x,
                       int          max_y,
                       int          ns,
                       camera**     cam,
                       hitable**    world,
                       curandState* rand_state)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if ((i >= max_x) || (j >= max_y)) { return; }

    int         pixel_index      = j * max_x + i;
    curandState local_rand_state = rand_state[pixel_index];

    vec3 col(0);
    for (int s = 0; s < ns; s++)
    {
        // FIXME - for循环的迭代变量写成i，会覆盖for循环之外的局部变量i
        real u = (i + curand_uniform(&local_rand_state)) / max_x;
        real v = (j + curand_uniform(&local_rand_state)) / max_y;
        ray  r = (*cam)->get_ray(u, v);
        col += color(r, world, &local_rand_state);
    }
    fb[pixel_index] = (col / real(ns)).gamma_correction();
}

__global__ void create_world(hitable** d_list, hitable** d_world, camera** d_camera)
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        *(d_list)     = new sphere(vec3(0, 0, -1), 0.5);
        *(d_list + 1) = new sphere(vec3(0, -100.5, -1), 100);
        *d_world      = new hitable_list(d_list, 2);
        *d_camera     = new camera();
    }
}

__global__ void free_world(hitable** d_list, hitable** d_world, camera** d_camera)
{
    delete *(d_list);
    delete *(d_list + 1);
    delete *d_world;
    delete *d_camera;
}

int main(int argc, char** argv)
{
    // time
    auto start = std::chrono::system_clock ::now();

    int    num_pixels = nx * ny;
    size_t fb_size    = num_pixels * sizeof(vec3);

    // allocate FB on host
    vec3* fb = nullptr;
    checkCudaErrors(cudaMallocManaged((void**)&fb, fb_size));

    // curand
    curandState* d_rand_state;
    checkCudaErrors(cudaMalloc((void**)&d_rand_state, num_pixels * sizeof(curandState)));

    // allocate scene data
    hitable **d_list, **d_world;
    camera**  d_camera;
    checkCudaErrors(cudaMalloc((void**)&d_list, 2 * sizeof(hitable*)));
    checkCudaErrors(cudaMalloc((void**)&d_world, sizeof(hitable*)));
    checkCudaErrors(cudaMalloc((void**)&d_camera, sizeof(camera*)));
    create_world<<<1, 1>>>(d_list, d_world, d_camera);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    // Render our buffer
    dim3 blocks(nx / tx + 1, ny / ty + 1);
    dim3 threads(tx, ty);
    render_init<<<blocks, threads>>>(nx, ny, d_rand_state);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    render<<<blocks, threads>>>(fb, nx, ny, ns, d_camera, d_world, d_rand_state);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    // TODO - 可选的，在GPU写回数据到CPU时，把real转换成8bit整数以节省带宽

    // Output FB as Image
    std::ofstream out(OUTPUT_FILE);
    if (!out.is_open())
    {
        std::cerr << "Failed to open file: " << OUTPUT_FILE << std::endl;
        return EXIT_FAILURE;
    }
    out << "P3\n" << nx << ' ' << ny << "\n255\n";

    for (int j = ny - 1; j >= 0; j--)
    {
        for (int i = 0; i < nx; i++)
        {
            size_t pixel_index = j * nx + i;
            vec3   col         = fb[pixel_index] * 255.99;
            out << (int)col.x() << " " << (int)col.y() << " " << (int)col.z() << "\n";
            // std::cout << col.x() << " " << col.y() << " " << col.z() << "\n";
        }
    }

    // 释放cuda上的资源
    free_world<<<1, 1>>>(d_list, d_world, d_camera);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaFree(d_list));
    checkCudaErrors(cudaFree(d_world));
    checkCudaErrors(cudaFree(fb));
    cudaDeviceReset();

    // time
    auto end      = std::chrono::system_clock ::now();
    auto duration = std::chrono::duration_cast<timeType>(end - start);
    std::cout << "Time taken: " << duration.count() / 1000.0f << " s\n";

    return EXIT_SUCCESS;
}
