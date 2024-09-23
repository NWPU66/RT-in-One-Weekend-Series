#include <cstdlib>

#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <string>

#include "util/util.h"

const std::string OUTPUT_FILE = "output.ppm";

vec3 ray_color(const ray& r, const hittable_list& world)
{
    hit_record rec;
    if (world.hit(r, 0.01, std::numeric_limits<double>::max(), rec))
    {
        return 0.5 * (rec.normal + vec3(1));
    }

    vec3 unit_direction = glm::normalize(r.direction());
    auto t              = 0.5 * (unit_direction.y + 1.0);
    return (1.0 - t) * vec3(1.0, 1.0, 1.0) + t * vec3(0.5, 0.7, 1.0);
}

inline double random_double()
{
    static std::uniform_real_distribution<double> distribution(0.0, 1.0);
    static std::mt19937                           generator;
    static std::function<double()> rand_generator = std::bind(distribution, generator);
    return rand_generator();
}

int main(int argc, char** argv)
{
    const int image_width       = 200;
    const int image_height      = 100;
    const int samples_per_pixel = 100;

    std::ofstream out(OUTPUT_FILE);
    if (!out.is_open())
    {
        std::cerr << "Failed to open file: " << OUTPUT_FILE << std::endl;
        return EXIT_FAILURE;
    }
    out << "P3\n" << image_width << ' ' << image_height << "\n255\n";

    hittable_list world;
    world.add(std::make_shared<sphere>(vec3(0, 0, -1), 0.5));
    world.add(std::make_shared<sphere>(vec3(0, -100.5, -1), 100));
    camera cam;

    for (int j = image_height - 1; j >= 0; --j)
    {
        std::cout << "\rScanlines remaining: " << j << ' ' << std::flush;
        for (int i = 0; i < image_width; ++i)
        {
            vec3 color(0);
            for (int s = 0; s < samples_per_pixel; ++s)
            {
                auto u = (i + random_double()) / image_width;
                auto v = (j + random_double()) / image_height;

                color += ray_color(cam.get_ray(u, v), world);
            }
            write_color(out, color, samples_per_pixel);
        }
    }
    std::cout << "\nDone.\n";

    return EXIT_SUCCESS;
}

// #include <cstdlib>
// #include <stdexcept>

// #include <fstream>
// #include <iostream>
// #include <string>

// #include <glm/geometric.hpp>

// #define HIGH_PRECISION
// #include "marco.h"
// #include "ray.h"

// const std::string OUTPUT_FILE = "output.ppm";

// double hit_sphere(const vec3& center, double radius, const ray& r)
// {
//     vec3 oc           = r.origin() - center;
//     auto a            = dot(r.direction(), r.direction());
//     auto b            = 2.0 * dot(oc, r.direction());
//     auto c            = dot(oc, oc) - radius * radius;
//     auto discriminant = b * b - 4 * a * c;

//     if (discriminant < 0) { return -1.0; }
//     else
//     {
//         // 返回交点中的较小值，有可能是负的
//         return (-b - sqrt(discriminant)) / (2.0 * a);
//     }
// }

// vec3 ray_color(const ray& r)
// {
//     const vec3   sphere_center = vec3(0, 0, -1);
//     const double sphere_radius = 0.5;

//     auto t = hit_sphere(sphere_center, sphere_radius, r);
//     if (t > 0.0)
//     {
//         vec3 N = glm::normalize(r.at(t) - sphere_center);
//         return 0.5 * (N + vec3(1));
//     }

//     vec3 unit_direction = glm::normalize(r.direction());
//     t                   = 0.5 * (unit_direction.y + 1.0);
//     return (1.0 - t) * vec3(1.0, 1.0, 1.0) + t * vec3(0.5, 0.7, 1.0);
// }

// int main(int argc, char** argv)
// {
//     try
//     {
//         const int image_width  = 200;
//         const int image_height = 100;

//         const vec3 lower_left_corner(-2.0, -1.0, -1.0);
//         const vec3 horizontal(4.0, 0.0, 0.0);
//         const vec3 vertical(0.0, 2.0, 0.0);
//         const vec3 origin(0.0, 0.0, 0.0);

//         std::ofstream out(OUTPUT_FILE);
//         out.is_open();
//         out << "P3\n" << image_width << " " << image_height << "\n255\n";
//         for (int j = image_height - 1; j >= 0; --j)
//         {
//             for (int i = 0; i < image_width; ++i)
//             {
//                 std::cout << "\rScanlines remaining: " << j << ' ' << std::flush;

//                 auto u = double(i) / image_width;
//                 auto v = double(j) / image_height;
//                 ray  r(origin, lower_left_corner + u * horizontal + v * vertical);
//                 vec3 color = ray_color(r);

//                 out << (int)(255.999 * color.x) << " " << (int)(255.999 * color.y) << " "
//                     << (int)(255.999 * color.z) << "\n";
//             }
//         }

//         std::cout << "\nDone.\n";
//     }
//     catch (std::runtime_error& e)
//     {
//         std::cerr << e.what() << std::endl;
//         return EXIT_FAILURE;
//     }

//     return EXIT_SUCCESS;
// }