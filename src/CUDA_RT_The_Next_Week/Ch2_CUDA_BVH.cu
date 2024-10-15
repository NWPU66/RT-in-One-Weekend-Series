// c
#include <cfloat>

// cpp
#include <chrono>
#include <fstream>
#include <iostream>

// cuda
#include "cuda_runtime.h"
#include "curand_kernel.h"
#include "device_launch_parameters.h"

// user
#include "raytracinginoneweekendincuda/bvh.h"
#include "raytracinginoneweekendincuda/camera.h"
#include "raytracinginoneweekendincuda/hitable.h"
#include "raytracinginoneweekendincuda/hitable_list.h"
#include "raytracinginoneweekendincuda/material.h"
#include "raytracinginoneweekendincuda/moving_sphere.h"
#include "raytracinginoneweekendincuda/ray.h"
#include "raytracinginoneweekendincuda/sphere.h"
#include "raytracinginoneweekendincuda/util.h"
#include "raytracinginoneweekendincuda/vec3.h"

#define checkCudaErrors(val) check_cuda((val), #val, __FILE__, __LINE__)
void check_cuda(cudaError_t result, char const* const func, const char* const file, int const line)
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

// Matching the C++ code would recurse enough into color() calls that
// it was blowing up the stack, so we have to turn this into a
// limited-depth loop instead.  Later code in the book limits to a max
// depth of 50, so we adapt this a few chapters early on the GPU.
__device__ vec3 color(const ray& r, hitable** world, curandState* local_rand_state)
{
    ray  cur_ray         = r;
    vec3 cur_attenuation = vec3(1.0, 1.0, 1.0);
    for (int i = 0; i < 50; i++)
    {
        hit_record rec;
        if ((*world)->hit(cur_ray, 0.001f, FLT_MAX, rec))
        {
            ray  scattered;
            vec3 attenuation;
            if (rec.mat_ptr->scatter(cur_ray, rec, attenuation, scattered, local_rand_state))
            {
                cur_attenuation *= attenuation;
                cur_ray = scattered;
            }
            else { return vec3(0.0, 0.0, 0.0); }
        }
        else
        {
            vec3  unit_direction = unit_vector(cur_ray.direction());
            float t              = 0.5f * (unit_direction.y() + 1.0f);
            vec3  c              = (1.0f - t) * vec3(1.0, 1.0, 1.0) + t * vec3(0.5, 0.7, 1.0);
            return cur_attenuation * c;
        }
    }
    return vec3(0.0, 0.0, 0.0);  // exceeded recursion
}

__global__ void rand_init(curandState* rand_state)
{
    if (threadIdx.x == 0 && blockIdx.x == 0) { curand_init(1984, 0, 0, rand_state); }
}

__global__ void render_init(int max_x, int max_y, curandState* rand_state)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if ((i >= max_x) || (j >= max_y)) return;
    int pixel_index = j * max_x + i;
    // Original: Each thread gets same seed, a different sequence number, no offset
    // curand_init(1984, pixel_index, 0, &rand_state[pixel_index]);
    // BUGFIX, see Issue#2: Each thread gets different seed, same sequence for
    // performance improvement of about 2x!
    curand_init(1984 + pixel_index, 0, 0, &rand_state[pixel_index]);
}

__global__ void render(vec3*        fb,
                       int          max_x,
                       int          max_y,
                       int          ns,
                       camera**     cam,
                       hitable**    world,
                       hitable** bvh,
                       curandState* rand_state)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if ((i >= max_x) || (j >= max_y)) return;
    int         pixel_index      = j * max_x + i;
    curandState local_rand_state = rand_state[pixel_index];
    vec3        col(0, 0, 0);
    for (int s = 0; s < ns; s++)
    {
        float u = float(i + curand_uniform(&local_rand_state)) / float(max_x);
        float v = float(j + curand_uniform(&local_rand_state)) / float(max_y);
        ray   r = (*cam)->get_ray(u, v, &local_rand_state);
        // col += color(r, world, &local_rand_state);
        col += color(r, bvh, &local_rand_state);
    }
    rand_state[pixel_index] = local_rand_state;
    col /= float(ns);
    fb[pixel_index] = col.gamma_correction();
}

#define RND (curand_uniform(&local_rand_state))

__global__ void create_world(hitable**    d_list,
                             hitable**    d_world,
                             hitable**    d_bvh,
                             camera**     d_camera,
                             int          nx,
                             int          ny,
                             int          n_objects,
                             curandState* rand_state)
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        curandState local_rand_state = *rand_state;
        d_list[0] = new sphere(vec3(0, -1000.0, -1), 1000, new lambertian(vec3(0.5, 0.5, 0.5)));
        int i     = 1;
        for (int a = -11; a < 11; a++)
        {
            for (int b = -11; b < 11; b++)
            {
                float choose_mat = RND;
                vec3  center(a + RND, 0.2, b + RND);
                if (choose_mat < 0.8f)
                {
                    d_list[i++] = new sphere(center, 0.2,
                                             new lambertian(vec3(RND * RND, RND * RND, RND * RND)));
                    // new moving_sphere(center, center + vec3(0, RND * 0.5, 0), 0.0, 1.0, 0.2,
                    //                   new lambertian(vec3(RND * RND, RND * RND, RND * RND)));
                }
                else if (choose_mat < 0.95f)
                {
                    d_list[i++] =
                        new sphere(center, 0.2,
                                   new metal(vec3(0.5f * (1.0f + RND), 0.5f * (1.0f + RND),
                                                  0.5f * (1.0f + RND)),
                                             0.5f * RND));
                }
                else { d_list[i++] = new sphere(center, 0.2, new dielectric(1.5)); }
            }
        }
        d_list[i++] = new sphere(vec3(0, 1, 0), 1.0, new dielectric(1.5));
        d_list[i++] = new sphere(vec3(0, 1, 0), -0.95, new dielectric(1.5));
        d_list[i++] = new sphere(vec3(-4, 1, 0), 1.0, new lambertian(vec3(0.4, 0.2, 0.1)));
        d_list[i++] = new sphere(vec3(4, 1, 0), 1.0, new metal(vec3(0.7, 0.6, 0.5), 0.0));
        *rand_state = local_rand_state;
        *d_world    = new hitable_list(d_list, n_objects);

        vec3  lookfrom(13, 2, 3);
        vec3  lookat(0, 0, 0);
        float dist_to_focus = 10.0;
        float aperture      = 0.001;
        float time0 = 0.0, time1 = 1.0;
        *d_camera = new camera(lookfrom, lookat, vec3(0, 1, 0), 30.0, float(nx) / float(ny),
                               aperture, dist_to_focus, time0, time1);

        // create the bvh tree
        *d_bvh = new bvh_node(d_list, 0, n_objects, time0, time1, rand_state);

        // hit_record rec;
        // (**d_bvh).hit(ray(vec3(0), vec3(0, -0.5, -1), 0), 0.0001, FLT_MAX, rec);
    }
}

__global__ void free_world(hitable** d_list, hitable** d_world, camera** d_camera, hitable** d_bvh)
{
    for (int i = 0; i < 22 * 22 + 1 + 4; i++)  // NOTE -
    {
        // mat_ptr 已经在析构函数上释放了
        // FIXME - 并非所有的hitable都是sphere类
        delete d_list[i];
    }
    delete *d_world;
    delete *d_bvh;
    delete *d_camera;
}

int main()
{
    const int         nx          = 1200;
    const int         ny          = 800;
    const int         ns          = 16;
    const int         tx          = 8;
    const int         ty          = 8;
    const std::string OUTPUT_FILE = "output.ppm";

    std::cout << "Rendering a " << nx << "x" << ny << " image with " << ns << " samples per pixel ";
    std::cout << "in " << tx << "x" << ty << " blocks.\n";

    constexpr int    num_pixels = nx * ny;
    constexpr size_t fb_size    = num_pixels * sizeof(vec3);

    // allocate FB
    vec3* fb;
    checkCudaErrors(cudaMallocManaged((void**)&fb, fb_size));

    // allocate random state
    curandState* d_rand_state;
    checkCudaErrors(cudaMalloc((void**)&d_rand_state, num_pixels * sizeof(curandState)));
    curandState* d_rand_state2;
    checkCudaErrors(cudaMalloc((void**)&d_rand_state2, 1 * sizeof(curandState)));

    // we need that 2nd random state to be initialized for the world creation
    rand_init<<<1, 1>>>(d_rand_state2);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    // NOTE - num of objects
    const int     small = 22 * 22, big = 4, ground = 1;
    constexpr int total = small + big + ground;

    // make our world of hitables & the camera
    hitable** d_list;
    checkCudaErrors(cudaMalloc((void**)&d_list, total * sizeof(hitable*)));
    hitable** d_world;
    checkCudaErrors(cudaMalloc((void**)&d_world, sizeof(hitable*)));
    camera** d_camera;
    checkCudaErrors(cudaMalloc((void**)&d_camera, sizeof(camera*)));
    hitable** d_bvh;
    checkCudaErrors(cudaMalloc((void**)&d_bvh, sizeof(hitable*)));
    create_world<<<1, 1>>>(d_list, d_world, d_bvh, d_camera, nx, ny, total, d_rand_state2);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    // render with bvh
    auto start = std::chrono::system_clock::now();
    // Render our buffer
    const dim3 blocks(nx / tx + 1, ny / ty + 1);
    const dim3 threads(tx, ty);
    render_init<<<blocks, threads>>>(nx, ny, d_rand_state);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    render<<<blocks, threads>>>(fb, nx, ny, ns, d_camera, d_world, d_bvh, d_rand_state);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    auto stop     = std::chrono::system_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
    std::cout << "With BVH, Time:  " << duration.count() / 1000.0f << " s\n";

    // render without bvh
    start = std::chrono::system_clock::now();
    render<<<blocks, threads>>>(fb, nx, ny, ns, d_camera, d_world, d_world, d_rand_state);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    stop     = std::chrono::system_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
    std::cout << "Without BVH, Time:  " << duration.count() / 1000.0f << " s\n";

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
        }
    }
    out.close();

    // clean up
    checkCudaErrors(cudaDeviceSynchronize());
    free_world<<<1, 1>>>(d_list, d_world, d_camera, d_bvh);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaFree(d_camera));
    checkCudaErrors(cudaFree(d_world));
    checkCudaErrors(cudaFree(d_bvh));
    checkCudaErrors(cudaFree(d_list));
    checkCudaErrors(cudaFree(d_rand_state));
    checkCudaErrors(cudaFree(d_rand_state2));
    checkCudaErrors(cudaFree(fb));

    cudaDeviceReset();

    return EXIT_SUCCESS;
}
