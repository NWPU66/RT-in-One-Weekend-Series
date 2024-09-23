#include <cstdlib>

#include <fstream>
#include <iostream>
#include <string>

#include "./util/util.h"

const std::string OUTPUT_FILE = "./image.ppm";

int main(int argc, char** argv)
{
    const int image_width  = 400;
    const int image_height = 200;

    std::ofstream file(OUTPUT_FILE);
    if (file.is_open())
    {
        file << "P3\n" << image_width << " " << image_height << "\n255\n";

        for (int j = image_height - 1; j >= 0; j--)
        {
            for (int i = 0; i < image_width; i++)
            {
                glm::vec3 color(double(i) / image_width, double(j) / image_height, 0.0);
                int       ir = int(255.99 * color.x);
                int       ig = int(255.99 * color.y);
                int       ib = int(255.99 * color.z);
                file << ir << " " << ig << " " << ib << std::endl;
            }
        }

        file.close();
        std::cout << "Successfully wrote to file: " << OUTPUT_FILE << std::endl;
    }
    else
    {
        std::cerr << "Failed to open file: " << OUTPUT_FILE << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}