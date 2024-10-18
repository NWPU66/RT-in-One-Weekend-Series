// c
#include <cfloat>

// cpp
#include <chrono>
#include <cstdlib>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>

// cuda
#include "cuda_runtime.h"
#include "curand_kernel.h"

// user
#include "raytracinginoneweekendincuda/box.h"
#include "raytracinginoneweekendincuda/bvh.h"
#include "raytracinginoneweekendincuda/camera.h"
#include "raytracinginoneweekendincuda/hitable.h"
#include "raytracinginoneweekendincuda/hitable_list.h"
#include "raytracinginoneweekendincuda/material.h"
#include "raytracinginoneweekendincuda/moving_sphere.h"
#include "raytracinginoneweekendincuda/perlin.h"
#include "raytracinginoneweekendincuda/ray.h"
#include "raytracinginoneweekendincuda/rectangle.h"
#include "raytracinginoneweekendincuda/sphere.h"
#include "raytracinginoneweekendincuda/texture.h"
#include "raytracinginoneweekendincuda/util.h"
#include "raytracinginoneweekendincuda/vec3.h"
#include "raytracinginoneweekendincuda/volume.h"

// 3rdparty
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

const std::string IMAGE_FILE  = "E:/Study/CodeProj/RT-in-One-Weekend-Series/asset/world.jpg";
const std::string OUTPUT_FILE = "output.ppm";

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

__device__ vec3 ray_color(const ray&   r,
                          hitable**    world,
                          curandState* local_rand_state,
                          const vec3&  background = {0})
{
    ray  cur_ray         = r;
    vec3 cur_attenuation = vec3(1);
    for (int i = 0; i < 50; i++)
    {
        hit_record rec;
        if ((*world)->hit(cur_ray, 0.001f, FLT_MAX, rec))
        {
            ray  scattered;
            vec3 attenuation;

            vec3 emitted = rec.mat_ptr->emitted(rec.u, rec.v, rec.p);
            if (dot(cur_ray.direction(), rec.normal) > 0) { emitted = vec3{0}; }

            if (rec.mat_ptr->scatter(cur_ray, rec, attenuation, scattered, local_rand_state))
            {
                cur_attenuation *= attenuation;
                cur_ray = scattered;
            }
            else
            {
                // 自发光材质，没有scatter
                return cur_attenuation * emitted;
            }
        }
        else
        {
            // sky light
            vec3  unit_direction = unit_vector(cur_ray.direction());
            float t              = 0.5f * (unit_direction.y() + 1.0f);
            vec3  sky_light      = (1.0f - t) * vec3(1.0, 1.0, 1.0) + t * vec3(0.5, 0.7, 1.0);

            // return cur_attenuation * (background + sky_light * 0.05);
            return cur_attenuation * background;
        }
    }
    return vec3(0);  // exceeded recursion
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
                       curandState* rand_state)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if ((i >= max_x) || (j >= max_y)) return;
    int         pixel_index      = j * max_x + i;
    curandState local_rand_state = rand_state[pixel_index];
    vec3        col(0);
    for (int s = 0; s < ns; s++)
    {
        float u = float(i + curand_uniform(&local_rand_state)) / float(max_x);
        float v = float(j + curand_uniform(&local_rand_state)) / float(max_y);
        ray   r = (*cam)->get_ray(u, v, &local_rand_state);
        // col += color(r, world, &local_rand_state);
        col += ray_color(r, world, &local_rand_state);
    }
    rand_state[pixel_index] = local_rand_state;
    col /= float(ns);
    fb[pixel_index] = col.gamma_correction();
}

#define RND (curand_uniform(&local_rand_state))

__global__ void create_world(hitable**      d_list,
                             hitable**      d_world,
                             camera**       d_camera,
                             int            nx,
                             int            ny,
                             int            n_objects,
                             unsigned char* texture_device_data,
                             int            texture_width,
                             int            texture_height,
                             curandState*   rand_state)
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        curandState local_rand_state = *rand_state;

        // texture
        auto red_mat   = new lambertian(new const_texture(vec3(0.65, 0.05, 0.05)));
        auto white_mat = new lambertian(new const_texture(vec3(0.73, 0.73, 0.73)));
        auto green_mat = new lambertian(new const_texture(vec3(0.12, 0.45, 0.15)));
        auto light_mat = new diffuse_light(new const_texture(vec3(15.0)));

        // main objects
        int i       = 0;
        d_list[i++] = new yz_rect(0, 555, 0, 555, 555, green_mat);
        d_list[i++] = new yz_rect(0, 555, 0, 555, 0, red_mat);
        d_list[i++] = new xz_rect(0, 555, 0, 555, 0, white_mat);
        d_list[i++] = new xy_rect(0, 555, 0, 555, 555, white_mat);
        d_list[i++] = new xz_rect(0, 555, 0, 555, 555, white_mat);
        // 2 box
        auto* dieletric = new sphere(vec3(210, 100, 150), 80, new dielectric(1.5));
        auto* volume =
            new constant_medium(dieletric, 0.02, new const_texture(vec3(0, 0, 0.8)), rand_state);
        d_list[i++] = new SSS_volume(dieletric, volume);
        d_list[i++] =
            new rotate_y(new box(vec3(265, 0, 295), vec3(430, 330, 460), white_mat), -45.0f);

        // light
        d_list[i++] = new flip_face(new xz_rect(213, 343, 227, 332, 554, light_mat));

        // create the world
        *d_world = new hitable_list(d_list, n_objects);

        // camera
        const auto aspect_ratio = double(nx) / ny;
        vec3       lookfrom(278, 278, -800);
        vec3       lookat(278, 278, 0);
        vec3       vup(0, 1, 0);
        auto       dist_to_focus = 10.0;
        auto       aperture      = 0.0;
        auto       vfov          = 40.0;
        float      time0 = 0.0, time1 = 1.0;
        *d_camera = new camera(lookfrom, lookat, vup, vfov, aspect_ratio, aperture, dist_to_focus,
                               time0, time1);
    }
}

__global__ void free_world(hitable** d_list, int num_objects, hitable** d_world, camera** d_camera)
{
    for (int i = 0; i < num_objects; i++)
    {
        // mat_ptr 已经在析构函数上释放了
        // FIXME - 并非所有的hitable都是sphere类
        delete d_list[i];
        // FIXME - delete好像不会检查指针的有效性，直接全删了，
        // 我只有6个对象，delete d_list[20]全删了
    }
    delete *d_world;
    delete *d_camera;
}

int main()
{
    const int nx = 800;
    const int ny = 800;
    const int ns = 128;
    const int tx = 8;
    const int ty = 8;

    std::cout << "Rendering a " << nx << "x" << ny << " image with " << ns << " samples per pixel ";
    std::cout << "in " << tx << "x" << ty << " blocks.\n";

    constexpr int    num_pixels = nx * ny;
    constexpr size_t fb_size    = num_pixels * sizeof(vec3);

    // allocate FB
    vec3 *fb, *fb_host;
    fb_host = new vec3[num_pixels];
    checkCudaErrors(cudaMallocManaged((void**)&fb, fb_size));

    // import image to cuda
    int            width, height, channels;
    unsigned char* texture_host_data = stbi_load(IMAGE_FILE.c_str(), &width, &height, &channels, 0);
    int            texture_size      = width * height * channels;
    unsigned char* texture_device_data;
    checkCudaErrors(cudaMallocManaged((void**)&texture_device_data, texture_size));
    checkCudaErrors(
        cudaMemcpy(texture_device_data, texture_host_data, texture_size, cudaMemcpyHostToDevice));

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
    const int num_objects = 8;

    // make our world of hitables & the camera
    hitable** d_list;
    checkCudaErrors(cudaMalloc((void**)&d_list, num_objects * sizeof(hitable*)));
    hitable** d_world;
    checkCudaErrors(cudaMalloc((void**)&d_world, sizeof(hitable*)));
    camera** d_camera;
    checkCudaErrors(cudaMalloc((void**)&d_camera, sizeof(camera*)));
    create_world<<<1, 1>>>(d_list, d_world, d_camera, nx, ny, num_objects, texture_device_data,
                           width, height, d_rand_state2);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    auto start = std::chrono::system_clock::now();
    // Render our buffer
    const dim3 blocks(nx / tx + 1, ny / ty + 1);
    const dim3 threads(tx, ty);
    render_init<<<blocks, threads>>>(nx, ny, d_rand_state);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    render<<<blocks, threads>>>(fb, nx, ny, ns, d_camera, d_world, d_rand_state);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    auto stop     = std::chrono::system_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
    std::cout << "Time:  " << duration.count() / 1000.0f << " s\n";

    // transfer FB to host
    checkCudaErrors(cudaMemcpy(fb_host, fb, fb_size, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    //  Output FB as Image
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
            vec3   col         = fb_host[pixel_index] * 255.999;
            out << col << "\n";
        }
    }
    out.close();

    // clean up
    checkCudaErrors(cudaDeviceSynchronize());
    free_world<<<1, 1>>>(d_list, num_objects, d_world, d_camera);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaFree(d_camera));
    checkCudaErrors(cudaFree(d_world));
    checkCudaErrors(cudaFree(d_list));
    checkCudaErrors(cudaFree(d_rand_state));
    checkCudaErrors(cudaFree(d_rand_state2));
    checkCudaErrors(cudaFree(fb));
    delete[] texture_host_data, fb_host;
    checkCudaErrors(cudaFree(texture_device_data));

    cudaDeviceReset();

    return EXIT_SUCCESS;
}
