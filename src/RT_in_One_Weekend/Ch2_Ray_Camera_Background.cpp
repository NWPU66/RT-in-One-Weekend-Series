#include <array>
#include <cmath>
#include <cstdlib>

#include <chrono>
#include <cstring>
#include <fstream>
#include <functional>
#include <glm/geometric.hpp>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <string>
#include <thread>

#include "util/util.h"

const std::string OUTPUT_FILE = "output.ppm";

vec3 ray_color(const ray& r, const hittable_list& world, int depth)
{
    hit_record rec;

    if (depth <= 0) { return vec3(0); }

    if (world.hit(r, 0.001, std::numeric_limits<double>::max(), rec))
    {
        ray  scattered;
        vec3 attenuation;

        if (rec.mat_ptr->scatter(r, rec.p, rec.normal, rec.front_face, attenuation, scattered))
        {
            return attenuation * ray_color(scattered, world, depth - 1);
        }
        return vec3(0);
    }

    vec3 unit_direction = glm::normalize(r.direction());
    auto t              = 0.5 * (unit_direction.y + 1.0);
    return (1.0 - t) * vec3(1.0, 1.0, 1.0) + t * vec3(0.5, 0.7, 1.0);
}

int main(int argc, char** argv)
{
    const int image_width       = 720;
    const int image_height      = 360;
    const int samples_per_pixel = 64;
    const int max_depth         = 16;

    // image
    vec3* image = new vec3[image_height * image_width];

    std::ofstream out(OUTPUT_FILE);
    if (!out.is_open())
    {
        std::cerr << "Failed to open file: " << OUTPUT_FILE << std::endl;
        return EXIT_FAILURE;
    }
    out << "P3\n" << image_width << ' ' << image_height << "\n255\n";

    const auto aspect_ratio = double(image_width) / image_height;
    vec3       lookfrom(3, 3, 2);
    vec3       lookat(0, 0, -1);
    vec3       vup(0, 1, 0);
    auto       dist_to_focus = glm::length(lookfrom - lookat);
    auto       aperture      = 2.0;
    camera     cam(lookfrom, lookat, vup, 20, aspect_ratio, aperture, dist_to_focus);

    hittable_list world;
    world.add(std::make_shared<sphere>(vec3(0, 0, -1), 0.5,
                                       std::make_shared<lambertian>(vec3(1.0, 0.2, 0.5))));
    world.add(std::make_shared<sphere>(vec3(0, -100.5, -1), 100,
                                       std::make_shared<lambertian>(vec3(0.8, 0.8, 0.0))));
    world.add(std::make_shared<sphere>(vec3(1, 0, -1), 0.5,
                                       std::make_shared<metal>(vec3(0.8, 0.6, 0.2), 0.3)));
    world.add(std::make_shared<sphere>(vec3(-1, 0, -1), 0.5, std::make_shared<dielectric>(1.5)));
    world.add(std::make_shared<sphere>(vec3(-1, 0, -1), -0.45, std::make_shared<dielectric>(1.5)));

    // debug
    // ray_color(ray(vec3(0), vec3(-1, -0.53, -1)), world, max_depth);

    for (const int test : {1})
    {
        // time
        auto start = std::chrono::system_clock::now();

        // muti thread
        const int num_threads    = test;
        const int row_per_thread = ceil(image_height / (float)num_threads);

        std::function<void(int)> thread_fn = [&](int thread_id) {
            for (int j = image_height - 1 - thread_id * row_per_thread;
                 j > image_height - 1 - (thread_id + 1) * row_per_thread; --j)
            {
                std::cout << "\rScanlines remaining[" << thread_id
                          << "]: " << j - (image_height - 1 - (thread_id + 1) * row_per_thread)
                          << ' ' << std::flush;
                for (int i = 0; i < image_width; ++i)
                {
                    vec3 color(0);
                    for (int s = 0; s < samples_per_pixel; ++s)
                    {
                        auto u = (i + random_double()) / image_width;
                        auto v = (j + random_double()) / image_height;

                        color += ray_color(cam.get_ray(u, v), world, max_depth);
                    }
                    // write_color(out, color, samples_per_pixel);
                    image[j * image_width + i] = color;
                }
            }
        };

        // create threads
        std::thread th[num_threads];
        for (int i = 0; i < num_threads; ++i)
        {
            th[i] = std::thread(thread_fn, i);
        }
        for (int i = 0; i < num_threads; ++i)
        {
            th[i].join();  // join threads
        }

        // time
        auto end      = std::chrono::system_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::seconds>(end - start);
        std::cout << "[" << test << "]Time taken: " << duration.count() << " seconds\n";
    }

    // write image to ppm
    for (int j = image_height - 1; j >= 0; --j)
    {
        for (int i = 0; i < image_width; ++i)
        {
            write_color(out, image[j * image_width + i], samples_per_pixel);
        }
    }
    std::cout << "\nDone.\n";

    return EXIT_SUCCESS;
}
