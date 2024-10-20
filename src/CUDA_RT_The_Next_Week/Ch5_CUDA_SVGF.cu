/**
 * @file Ch5_CUDA_SVGF.cu
 * @author Zijie Yuan (ZijieYuan@gmail.com)
 * @brief 时空方差引导滤波算法实现
 * @version 0.1
 * @date 2024-10-16
 *
 * @copyright Copyright (c) 2024
 *
 */

// c
#ifdef DONOTUSE
#    include <__clang_cuda_runtime_wrapper.h>
#endif
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>

// c++
#include <chrono>
#include <functional>
#include <initializer_list>
#include <iostream>
#include <map>
#include <mutex>
#include <new>
#include <queue>
#include <set>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

// cuda
#include "cuda_runtime.h"
#include "curand_kernel.h"
#include <curand_uniform.h>

// user
#define USE_GLM
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
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "glm/glm.hpp"
#include "stb_image.h"
#include "stb_image_write.h"

// ANCHOR - Marco and Variable----------------------------------------------------------------------

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

/**
 * @brief 全局变量
 *
 */
// frame buffer
const int     imageWidth  = 800;
const int     imageHeight = 800;
constexpr int numPixels   = imageWidth * imageHeight;
// camera and rendering
constexpr float aspectRatio = (float)imageWidth / (float)imageHeight;
const vec3      lookfrom(278, 278, -800);
const vec3      lookat(278, 278, 0);
const int       SPP       = 128;
const int       maxDepth  = 50;
const float     maxZDepth = 1500.0f;
// cuda concurrency
const dim3   threads(8, 8);
const dim3   blocks(imageWidth / threads.x + 1, imageHeight / threads.y + 1);
const size_t curandSeed = 3777;
std::mutex   CurandMutex;
// SVGF core
const int gBufferHistorySize = 2;
// image save
const std::string savePath = "out.png";

// ANCHOR - Curand------------------------------------------------------------------------------

__global__ void initCurandState(curandState* _pCurandState)
{
    if (threadIdx.x == 0 && blockIdx.x == 0) { curand_init(curandSeed, 0, 0, _pCurandState); }
}

__global__ void
initCurandStatePerPixel(int imageWidth, int imageHeight, curandState* _pCurandStatePerPixel)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if ((i >= imageWidth) || (j >= imageHeight)) { return; }
    int pixel_index = j * imageWidth + i;
    curand_init(curandSeed + pixel_index, 0, 0, _pCurandStatePerPixel + pixel_index);
}

/**
 * @brief Timer
 *
 */
class Timer {
public:
    __host__ Timer(std::string _message = "") : message(_message)
    {
        startTime = std::chrono::system_clock::now();
    }
    __host__ ~Timer()
    {
        auto stop     = std::chrono::system_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - startTime);
        std::cout << "Timer(): " << message << " Time:  " << duration.count() / 1000.0f << " s\n";
    }

private:
    std::chrono::time_point<std::chrono::system_clock> startTime;
    std::string                                        message;
};

/**
 * @brief Curand
 *
 */
class Curand {
public:
    __host__ static Curand* GetInstance()
    {
        if (singleton == nullptr)
        {
            std::lock_guard<std::mutex> lock(CurandMutex);
            if (singleton == nullptr) { singleton = new Curand(); }
        }
        return singleton;
    }

    __host__ void DestoryInstance()
    {
        if (singleton != nullptr)
        {
            delete singleton;
            singleton = nullptr;
        }
    }

    __host__ curandState* GetCurandState() const { return pCurandState; }
    __host__ curandState* GetCurandStatePerPixel() const { return pCurandStatePerPixel; }
    __host__ curandState* GetCurandStatePerPixel(int i, int j) const
    {
        int pixel_index = j * imageWidth + i;
        return pCurandStatePerPixel + pixel_index;
    }

private:
    static Curand* singleton;
    curandState*   pCurandState;
    curandState*   pCurandStatePerPixel;
    int            numPixels;

    Curand(const Curand&)            = delete;
    Curand(Curand&&)                 = delete;
    Curand& operator=(const Curand&) = delete;
    Curand& operator=(Curand&&)      = delete;

    __host__ Curand() : numPixels(::numPixels)
    {
        std::cout << "Curand(): Curand initlization" << std::endl;
        auto totalMen = numPixels * sizeof(curandState) + 1 * sizeof(curandState);
        std::cout << "Curand(): Total memory allocated: " << totalMen << "bytes. ";
        std::cout << "A Single curandState Cost: " << sizeof(curandState) << "bytes. "
                  << "Number of curandState: " << numPixels << std::endl;

        checkCudaErrors(cudaMalloc((void**)&pCurandState, numPixels * sizeof(curandState)));
        checkCudaErrors(cudaMalloc((void**)&pCurandStatePerPixel, 1 * sizeof(curandState)));

        initCurandState<<<1, 1>>>(pCurandState);
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize());

        initCurandStatePerPixel<<<blocks, threads>>>(imageWidth, imageHeight, pCurandStatePerPixel);
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize());
    }

    __host__ ~Curand()
    {
        checkCudaErrors(cudaFree(pCurandState));
        checkCudaErrors(cudaFree(pCurandStatePerPixel));
        checkCudaErrors(cudaDeviceSynchronize());
        pCurandState         = nullptr;
        pCurandStatePerPixel = nullptr;
    }
};

Curand* Curand::singleton = nullptr;

#define RNDSTATE (Curand::GetInstance()->GetCurandState())
#define RNDPPSTATE (Curand::GetInstance()->GetCurandState())
#define RNDf (curand_uniform(RNDSTATE))
#define RND3f (vec3(RNDf, RNDf, RNDf))

// ANCHOR - ImageBuffer---------------------------------------------------------------------------

/**
 * @brief ImageBuffer
 *
 */
template <typename DataType = float> class ImageBuffer {
public:
    /**
     * @brief create a new image buffer with empty data
     *
     * @param width
     * @param height
     * @param channels
     */
    __host__ ImageBuffer(int _width = ::imageWidth, int _height = ::imageHeight, int _channels = 3)
        : imageWidth(_width), imageHeight(_height), imageChannels(_channels),
          numPixels(_width * _height), byteSize(_width * _height * _channels * sizeof(DataType))
    {
        std::cout << "ImageBuffer(): ImageBuffer initlization Empty Buffer. " << std::endl;
        std::cout << "ImageBUffer(): Buffer size: " << byteSize
                  << "bytes. With Width: " << imageWidth << " Height: " << imageHeight
                  << " Channels: " << imageChannels << std::endl;
        pUnifiedMenData = AllocateUnifiedMenData(byteSize);
    }

    /**
     * @brief create a new image buffer with data from file
     *
     * @param filePath
     */
    __host__ ImageBuffer(const std::string& filePath)
    {
        std::cout << "ImageBuffer(): ImageBuffer initlization from File. " << std::endl;

        // load image from file
        int            width, height, channels;
        unsigned char* imageData = stbi_load(filePath.c_str(), &width, &height, &channels, 0);
        if (imageData == nullptr)
        {
            std::cerr << "load image failed" << std::endl;
            stbi_image_free(imageData);
            return;
        }

        // calculate image info
        imageWidth    = width;
        imageHeight   = height;
        imageChannels = channels;
        numPixels     = width * height;
        byteSize      = width * height * channels * sizeof(DataType);

        // transfer data type to DataType
        const size_t sourceByteSize = width * height * channels * sizeof(unsigned char);
        DataType*    pConvertedData =
            DataTypeConvertion<unsigned char, DataType>(imageData, sourceByteSize, 1 / 255, true);

        // transfer data to GPU
        pUnifiedMenData = AllocateUnifiedMenData(byteSize);
        checkCudaErrors(
            cudaMemcpy(pUnifiedMenData, pConvertedData, byteSize, cudaMemcpyHostToDevice));

        free(pConvertedData);  // free converted data
    }

    __host__ ~ImageBuffer() { ReleaseUnifiedMenData(pUnifiedMenData); }

    __host__ void Save(const std::string& filePath)
    {
        if (pUnifiedMenData == nullptr)
        {
            std::cerr << "ImageBuffer::Save() error: no data to save" << std::endl;
            return;
        }

        std::cout << "ImageBuffer::Save() Save image to " << filePath << std::endl;

        unsigned char* dataToSave =
            DataTypeConvertion<DataType, unsigned char>(pUnifiedMenData, byteSize, 255.99f, false);
        stbi_flip_vertically_on_write(true);
        stbi_write_png(filePath.c_str(), imageWidth, imageHeight, imageChannels, dataToSave, 0);
        stbi_image_free(dataToSave);
    }

    __host__ __device__ DataType* GetUnifiedMenData() const { return pUnifiedMenData; }

    __host__ bool hasData() const { return (pUnifiedMenData != nullptr); }

    /**
     * @brief DataTypeConvertion被设计用来转换数据，全程发生在__host__端
     *
     */
    template <typename SourceType, typename TargetType>
    __host__ TargetType* DataTypeConvertion(SourceType* pSource,
                                            size_t      sourceByteSize,
                                            float       multiplier    = 1.0f,
                                            bool        releaseSource = false)
    {
        if (pSource == nullptr)
        {
            std::cerr << "Error: Invalid pointer when converting data type" << std::endl;
            return nullptr;
        }
        if (static_cast<bool>(std::is_same<SourceType, TargetType>::value))
        {
            return reinterpret_cast<TargetType*>(pSource);  // 类型相同, 直接返回
        }

        const size_t numElements    = sourceByteSize / sizeof(SourceType);
        const size_t targetByteSize = numElements * sizeof(TargetType);
        TargetType*  pTarget        = (TargetType*)malloc(targetByteSize);
        for (int i = 0; i < numElements; i++)
        {
            pTarget[i] = static_cast<DataType>(pSource[i] * multiplier);
        }

        if (releaseSource) { free(pSource); }
        return pTarget;
    }

private:
    DataType* pUnifiedMenData;
    int       imageWidth;
    int       imageHeight;
    int       imageChannels;
    int       numPixels;
    size_t    byteSize;

    __host__ void ReleaseUnifiedMenData(DataType* dst)
    {
        if (dst != nullptr)
        {
            checkCudaErrors(cudaFree(dst));
            checkCudaErrors(cudaGetLastError());
            checkCudaErrors(cudaDeviceSynchronize());
            dst = nullptr;
        }
    }

    template <typename TargetDataType = DataType>
    __host__ TargetDataType* AllocateUnifiedMenData(size_t byteSize)
    {
        TargetDataType* ptr = nullptr;
        checkCudaErrors(cudaMallocManaged((void**)&ptr, byteSize));
        // cudaMallocManaged is a unified memory allocator that allows you to
        // allocate memory that is accessible from both the host and the device.

        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize());
        return ptr;
    }
};

// ANCHOR - GBufferPool---------------------------------------------------------------------------

/**
 * @brief GBufferPool
 * @note 从队首取出一个buffer指针，直接修改指针上的内存区，然后放回队尾
 */
class GBufferPool {
public:
    enum struct GBufferType : int {
        FullRendering = 3,
        Albedo        = 3,
        Irrandiance   = 3,
        Position      = 3,
        Depth         = 1,
        Normal        = 3,
        UV            = 3,
        Motion        = 3,
    };

    using GBufferType2Ptr = std::map<GBufferType, ImageBuffer<float>*>;

    __host__ GBufferPool(std::set<GBufferType> _gBufferTypes) : gBufferTypes(_gBufferTypes)
    {
        std::cout << "GBufferPool(): GBufferPool initlization." << std::endl;
        for (auto type : gBufferTypes)
        {
            auto* q = new std::queue<ImageBuffer<float>*>();
            for (int i = 0; i < gBufferHistorySize; i++)
            {
                q->emplace(new ImageBuffer<float>(imageWidth, imageHeight, static_cast<int>(type)));
            }
            gBufferMap.emplace(type, q);
        }
    }

    __host__ ~GBufferPool()
    {
        for (auto [type, ptr] : gBufferMap)
        {
            delete ptr;
        }
    }

    __host__ GBufferType2Ptr PopFrontBufferPtr()
    {
        auto map = GetFrontBufferPtr();
        for (auto [type, ptr] : gBufferMap)
        {
            ptr->pop();
        }
        return map;
    }

    __host__ GBufferType2Ptr GetFrontBufferPtr()
    {
        GBufferType2Ptr map;
        for (auto [type, ptr] : gBufferMap)
        {
            map.emplace(type, ptr->front());
        }
        return map;
    }

    __host__ GBufferType2Ptr GetBackBufferPtr()
    {
        GBufferType2Ptr map;
        for (auto [type, ptr] : gBufferMap)
        {
            map.emplace(type, ptr->back());
        }
        return map;
    }

    __host__ void UpdateGBufferPool(GBufferType2Ptr bufferMap)
    {
        for (auto [type, ptr] : bufferMap)
        {
            auto it = gBufferMap.find(type);
            if (it != gBufferMap.end()) { it->second->push(ptr); }
            else { std::cerr << "Error: GBufferType not found in GBufferPool" << std::endl; }
        }
    }

private:
    std::set<GBufferType>                                   gBufferTypes;
    std::map<GBufferType, std::queue<ImageBuffer<float>*>*> gBufferMap;
    // Irradiance = source / texture albedo
};

using GBufferType = GBufferPool::GBufferType;

// ANCHOR - Scene---------------------------------------------------------------------------

__device__ hitable* Transform(hitable* geometry, const vec3& offset = {0}, float rotateYAngle = 0)
{
    return new translate(new rotate_y(geometry, rotateYAngle), offset);
}

__global__ static void CreateWorld(class camera** const camera,
                                   hitable** const      geometries,
                                   hitable_list** const geometryList,
                                   material** const     materials,
                                   bvh_node** const     bvh,
                                   curandState* const   randState)
{
    if (threadIdx.x != 0 && blockIdx.x != 0) { return; }

    // create camera
    printf("create camera\n");
    vec3  lookfrom(278, 278, -800);
    vec3  lookat(278, 278, 0);
    vec3  vup(0, 1, 0);
    auto  dist_to_focus = 10.0;
    auto  aperture      = 0.0;
    auto  vfov          = 40.0;
    auto  aspectRatio   = (float)imageWidth / (float)imageHeight;
    float time0 = 0.0, time1 = 1.0;
    camera[0] = new class MovingCamera(lookfrom, lookfrom, lookat, lookat, vup, vfov, aspectRatio,
                                       aperture, dist_to_focus, time0, time1);
    // TODO - 设置相机的运动

    // create materials
    printf("create material\n");
    auto* red_mat    = new lambertian(new const_texture(vec3(0.65, 0.05, 0.05)));
    auto* white_mat  = new lambertian(new const_texture(vec3(0.73, 0.73, 0.73)));
    auto* green_mat  = new lambertian(new const_texture(vec3(0.12, 0.45, 0.15)));
    auto* light_mat  = new diffuse_light(new const_texture(vec3(15.0)));
    int   mat        = 0;
    materials[mat++] = red_mat;
    materials[mat++] = white_mat;
    materials[mat++] = green_mat;
    materials[mat++] = light_mat;

    // create geometry
    printf("create geometry\n");
    int geom           = 0;
    geometries[geom++] = new yz_rect(0, 555, 0, 555, 555, green_mat);
    geometries[geom++] = new yz_rect(0, 555, 0, 555, 0, red_mat);
    geometries[geom++] = new xz_rect(0, 555, 0, 555, 0, white_mat);
    geometries[geom++] = new xy_rect(0, 555, 0, 555, 555, white_mat);
    geometries[geom++] = new xz_rect(0, 555, 0, 555, 555, white_mat);
    // 2 box
    auto* dieletric = new sphere(vec3(210, 100, 150), 80, new dielectric(1.5));
    auto* volume =
        new constant_medium(dieletric, 0.02, new const_texture(vec3(0, 0, 0.8)), randState);
    geometries[geom++] = new SSS_volume(dieletric, volume);
    geometries[geom++] =
        Transform(new box(vec3(265, 0, 295), vec3(430, 330, 460), white_mat), vec3(0), -45.0f);

    // light
    geometries[geom++] = new flip_face(new xz_rect(213, 343, 227, 332, 554, light_mat));

    // hitable_list
    geometryList[0] = new hitable_list(geometries, geom);

    printf("finish!\n");

    // create bvh
    //  we are not going to use BVH right now
    //  bvh             = new bvh_node(*geometries, 0, 1, randState);
}

/**
 * @brief Scene
 *
 */
class Scene {
public:
    __host__ Scene()
    {
        std::cout << "Scene(): Scene initlization." << std::endl;

        // allocate cuda memory
        std::cout << "Scene(): Allocate cuda memory." << std::endl;
        auto totalMen = sizeof(class camera*) + numGeometries * sizeof(hitable*) +
                        numMaterial * sizeof(material*) + sizeof(bvh_node*) * sizeof(hitable_list);
        std::cout << "Scene(): Total memory allocated: " << totalMen << " bytes." << std::endl;
        checkCudaErrors(cudaMalloc((void**)&camera, sizeof(class camera*)));
        checkCudaErrors(cudaMalloc((void**)&geometries, numGeometries * sizeof(hitable*)));
        checkCudaErrors(cudaMalloc((void**)&materials, numMaterial * sizeof(material*)));
        checkCudaErrors(cudaMalloc((void**)&bvh, sizeof(bvh_node*)));
        checkCudaErrors(cudaMalloc((void**)&geometryList, sizeof(hitable_list*)));
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize());

        std::cout << "Scene(): Create world." << std::endl;
        auto randState = Curand::GetInstance()->GetCurandState();
        CreateWorld<<<1, 1>>>(camera, geometries, geometryList, materials, bvh,
                              randState);  // create world
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize());
    }

    __host__ ~Scene()
    // FIXME - 这里可能有问题，geometries和materials都是指向gpu内存的指针，可能会越界访问
    {
        for (int i = 0; i < numGeometries; i++)
        {
            delete geometries[i];
        }
        for (int i = 0; i < numMaterial; i++)
        {
            delete materials[i];
        }
        delete[] camera, bvh;

        checkCudaErrors(cudaFree(camera));
        checkCudaErrors(cudaFree(geometries));
        checkCudaErrors(cudaFree(materials));
        checkCudaErrors(cudaFree(bvh));
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize());
    }

    __host__ __device__ camera**       GetCamera() const { return camera; }
    __host__ __device__ hitable_list** GetGeometryList() const { return geometryList; }

private:
    const int      numGeometries = 8;
    const int      numMaterial   = 4;
    camera**       camera;
    hitable**      geometries;
    hitable_list** geometryList;
    material**     materials;
    bvh_node**     bvh;
};

// ANCHOR - RTRenderer---------------------------------------------------------------------------

__device__ inline vec3 DefaultToneMappingFunc(const vec3& x)
{
    return x.gamma_correction();
}

__device__ vec3 rayColor(const ray&   r,
                         hitable*     world,
                         curandState* randState,
                         const vec3&  background = {0})
{
    ray  cur_ray         = r;
    vec3 cur_attenuation = vec3(1);
    for (int i = 0; i < maxDepth; i++)
    {
        hit_record rec;
        if (world->hit(cur_ray, 0.001f, FLT_MAX, rec))
        {
            ray  scattered;
            vec3 attenuation;

            vec3 emitted = rec.mat_ptr->emitted(rec.u, rec.v, rec.p);
            if (dot(cur_ray.direction(), rec.normal) > 0) { emitted = vec3{0}; }

            if (rec.mat_ptr->scatter(cur_ray, rec, attenuation, scattered, randState))
            {
                cur_attenuation *= attenuation;
                cur_ray = scattered;
            }
            else
            {
                return cur_attenuation * emitted;  // 自发光材质，没有scatter
            }
        }
        else  // sky light
        {
            vec3  unit_direction = unit_vector(cur_ray.direction());
            float t              = 0.5f * (unit_direction.y() + 1.0f);
            vec3  sky_light      = (1.0f - t) * vec3(1.0, 1.0, 1.0) + t * vec3(0.5, 0.7, 1.0);

            // return cur_attenuation * (background + sky_light * 0.05);
            return cur_attenuation * background;
        }
    }
    return vec3(0);  // exceeded recursion
}

__global__ void _StandardRender(int                  width,
                                int                  height,
                                int                  spp,
                                vec3* const          frameBuffer,
                                camera** const       camera,
                                hitable_list** const geometryList,
                                curandState* const   randStatePerPixel)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if ((i >= width) || (j >= height)) { return; }
    int pixel_index = j * width + i;

    auto* randState = randStatePerPixel + pixel_index;

    vec3 col(0);
    for (int s = 0; s < spp; s++)
    {
        float u = float(i + curand_uniform(randState)) / float(width);
        float v = float(j + curand_uniform(randState)) / float(height);
        ray   r = camera[0]->get_ray(u, v, randState);
        col += rayColor(r, *geometryList, randState);
    }

    frameBuffer[pixel_index] = DefaultToneMappingFunc(col / float(spp));  // Tone Mapping
}

__device__ vec3 PTRayColor(const ray&   r,
                           hitable*     world,
                           vec3         cameraOrigin,
                           vec3         cameraLookDir,
                           float        maxZDepth,
                           int          pixelIndex,
                           vec3* const  albedoFB,
                           vec3* const  positionFB,
                           float* const depthFB,
                           vec3* const  normalFB,
                           vec3* const  uvFB,
                           curandState* randState,
                           const vec3&  background = {0})
{
    ray  cur_ray         = r;
    vec3 cur_attenuation = vec3(1);
    for (int i = 0; i < maxDepth; i++)
    {
        hit_record rec;

        // status
        bool hit_anything = world->hit(cur_ray, 0.001f, FLT_MAX, rec);
        bool isScatter;
        ray  scattered;
        vec3 attenuation;
        vec3 emitted;

        if (hit_anything)
        {
            emitted = rec.mat_ptr->emitted(rec.u, rec.v, rec.p);
            if (dot(cur_ray.direction(), rec.normal) > 0) { emitted = vec3{0}; }

            isScatter = rec.mat_ptr->scatter(cur_ray, rec, attenuation, scattered, randState);
            if (isScatter)
            {
                cur_attenuation *= attenuation;
                cur_ray = scattered;
            }
            else
            {
                return cur_attenuation * emitted;  // 自发光材质，没有scatter
            }
        }
        else  // sky light
        {
            vec3  unit_direction = unit_vector(cur_ray.direction());
            float t              = 0.5f * (unit_direction.y() + 1.0f);
            vec3  sky_light      = (1.0f - t) * vec3(1.0, 1.0, 1.0) + t * vec3(0.5, 0.7, 1.0);

            // return cur_attenuation * (background + sky_light * 0.05);
            return cur_attenuation * background;
        }

        if (i == 0)  // first bounce
        {
            // albedo
            if (hit_anything)
            {
                if (isScatter) { albedoFB[pixelIndex] = attenuation; }       // none emissive
                else { albedoFB[pixelIndex] = emitted.make_unit_vector(); }  // emissive
            }
            else { albedoFB[pixelIndex] = vec3(0); }

            positionFB[pixelIndex] = hit_anything ? rec.p : vec3(0);
            depthFB[pixelIndex] =
                hit_anything ? dot(cameraLookDir, rec.p - cameraOrigin) / maxZDepth : 1.0f;
            normalFB[pixelIndex] = hit_anything ? rec.normal : vec3(0);
            uvFB[pixelIndex]     = hit_anything ? vec3(rec.u, rec.v, 0) : vec3(0);
        }
    }
    return vec3(0);  // exceeded recursion
}

__device__ vec3 WorldPos2ScreenPos(const mat4& MVP_T, const vec4& worldPos)
{
    vec4 ans = {dot(MVP_T.e[0], worldPos), dot(MVP_T.e[1], worldPos), dot(MVP_T.e[2], worldPos),
                dot(MVP_T.e[3], worldPos)};
    return {ans.e[0] / ans.e[3], ans.e[1] / ans.e[3], ans.e[2] / ans.e[3]};
}

__global__ void _PathTracing(int                  width,
                             int                  height,
                             float                maxZDepth,
                             vec3* const          fullRenderingFB,
                             vec3* const          albedoFB,
                             vec3* const          irradianceFB,
                             vec3* const          positionFB,
                             float* const         depthFB,
                             vec3* const          normalFB,
                             vec3* const          uvFB,
                             vec3* const          motionFB,
                             camera** const       camera,
                             hitable_list** const geometryList,
                             curandState* const   randStatePerPixel)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if ((i >= width) || (j >= height)) { return; }
    int pixel_index = j * width + i;

    auto* randState = randStatePerPixel + pixel_index;

    vec3      col(0);
    const int spp = 1;
    for (int s = 0; s < spp; s++)
    {
        float u = float(i + curand_uniform(randState)) / float(width);
        float v = float(j + curand_uniform(randState)) / float(height);
        ray   r = camera[0]->get_ray(u, v, randState);
        col += PTRayColor(r,                           //
                          *geometryList,               //
                          camera[0]->CameraOrigin(),   //
                          camera[0]->CameraLookDir(),  //
                          maxZDepth,                   //
                          pixel_index,                 //
                          albedoFB,                    //
                          positionFB,                  //
                          depthFB,                     //
                          normalFB,                    //
                          uvFB,                        //
                          randState);                  //
    }

    vec3 renderColor = col / float(spp);
    // fullRenderingFB
    fullRenderingFB[pixel_index] = DefaultToneMappingFunc(renderColor);  // Tone Mapping
    // irradianceFB
    irradianceFB[pixel_index] = renderColor / albedoFB[pixel_index];
    // motionFB, prevPositionFB
    // TODO - 这一帧的屏幕空间坐标减去上一帧的坐标
    mat4 MVP_T            = ((MovingCamera*)camera[0])->GetMVP_T();
    mat4 prevMVP_T        = ((MovingCamera*)camera[0])->GetPrevMVP_T();
    vec3 screenPos        = WorldPos2ScreenPos(MVP_T, vec4(positionFB[pixel_index], 1.0f));
    vec3 prevScreenPos    = WorldPos2ScreenPos(prevMVP_T, vec4(positionFB[pixel_index], 1.0f));
    motionFB[pixel_index] = screenPos - prevScreenPos;
}

/**
 * @brief
 *
 */
class RTRenderer {
public:
    using ToneMappingFuncPtr = vec3 (*)(const vec3&);

    __host__ RTRenderer(Scene* _scene, GBufferPool* _gBufferPool)
        : scene(_scene), gBufferPool(_gBufferPool)
    {
        std::cout << "RTRenderer(): RTRenderer initlization." << std::endl;
    }
    __host__ ~RTRenderer()
    {
        delete scene;
        delete gBufferPool;
    }

    __host__ Scene*       GetScene() const { return scene; }
    __host__ GBufferPool* GetGBufferPool() const { return gBufferPool; }

    __host__ void StandardRender(ImageBuffer<float>* const renderTarget) const
    {
        auto*  randStatePerPixel = Curand::GetInstance()->GetCurandStatePerPixel();
        auto*  frameBuffer       = reinterpret_cast<vec3*>(renderTarget->GetUnifiedMenData());
        auto** camera            = scene->GetCamera();
        auto** geomtryList       = scene->GetGeometryList();
        /**FIXME - scene->GetCamera(); 越界访问问题
        GetGeometryList(): { return *geomtryList; }
        geomtryList本身存储的是gpu专用地址，但是解引用的时候，代码在cpu上，解引用得到的对象在gpu上
         */

        _StandardRender<<<blocks, threads>>>(imageWidth, imageHeight, SPP, frameBuffer, camera,
                                             geomtryList, randStatePerPixel);
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize());
    }

    __host__ void PathTracing(std::map<GBufferType, ImageBuffer<float>*> gBuffermap) const
    {
        auto*  randStatePerPixel = Curand::GetInstance()->GetCurandStatePerPixel();
        auto** camera            = scene->GetCamera();
        auto** geomtryList       = scene->GetGeometryList();

        _PathTracing<<<blocks, threads>>>(
            imageWidth,   //
            imageHeight,  //
            maxZDepth,    //
            reinterpret_cast<vec3*>(
                gBuffermap[GBufferType::FullRendering]->GetUnifiedMenData()),                    //
            reinterpret_cast<vec3*>(gBuffermap[GBufferType::Albedo]->GetUnifiedMenData()),       //
            reinterpret_cast<vec3*>(gBuffermap[GBufferType::Irrandiance]->GetUnifiedMenData()),  //
            reinterpret_cast<vec3*>(gBuffermap[GBufferType::Position]->GetUnifiedMenData()),     //
            reinterpret_cast<float*>(gBuffermap[GBufferType::Depth]->GetUnifiedMenData()),       //
            reinterpret_cast<vec3*>(gBuffermap[GBufferType::Normal]->GetUnifiedMenData()),       //
            reinterpret_cast<vec3*>(gBuffermap[GBufferType::UV]->GetUnifiedMenData()),           //
            reinterpret_cast<vec3*>(gBuffermap[GBufferType::Motion]->GetUnifiedMenData()),       //
            camera,                                                                              //
            geomtryList,                                                                         //
            randStatePerPixel                                                                    //
        );
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize());
    }

private:
    Scene*       scene;
    GBufferPool* gBufferPool;
};

// ANCHOR - SVGFApplication----------------------------------------------------------------------

/**
 * @brief
 *
 */
class SVGFApplication {
public:
    __host__ SVGFApplication() : scene(new Scene())
    {
        std::cout << "SVGFApplication(): SVGFApplication initlization." << std::endl;
        std::set<GBufferType> gBufferTypes = {
            GBufferType::FullRendering,  //
            GBufferType::Albedo,         //
            GBufferType::Irrandiance,    //
            GBufferType::Position,       //
            GBufferType::Depth,          //
            GBufferType::Normal,         //
            GBufferType::UV,             //
            GBufferType::Motion          //
        };
        gBufferPool = new GBufferPool(gBufferTypes);
        renderer    = new RTRenderer(scene, gBufferPool);
    }
    __host__ ~SVGFApplication()
    {
        delete renderer;
        delete scene;
        delete gBufferPool;
    }

    __host__ void Run()
    {
        StandardRender();

        // SVGF
        PathTracing();
        Reconstruction();
        PostProcessing();
    }

private:
    RTRenderer*  renderer;
    GBufferPool* gBufferPool;
    Scene*       scene;

    __host__ void StandardRender()
    {
        // import an image as a test
        auto* earth_img = new ImageBuffer<unsigned char>("earth.jpg");

        std::cout << "StandardRender(): " << "Rendering a " << imageWidth << "x" << imageHeight
                  << " image with " << SPP << " samples per pixel ";
        std::cout << "in " << threads.x << "x" << threads.y << " blocks.\n";

        // start to render
        auto renderTarget = GBufferType::FullRendering;
        {
            Timer timer("StandardRender()");
            auto  gBufferMap = gBufferPool->PopFrontBufferPtr();
            renderer->StandardRender(gBufferMap.find(renderTarget)->second);
            gBufferPool->UpdateGBufferPool(gBufferMap);
        }

        //  Output FB as Image
        {
            auto frameBuffer = gBufferPool->GetBackBufferPtr().find(renderTarget)->second;
            frameBuffer->Save(savePath);
        }

        delete earth_img;  // release image

        std::cout << "StandardRender(): Rendering Finished!" << std::endl;
    }

    __host__ void PathTracing()
    {
        std::cout << "Path Tracing(): start to generate GBuffer" << std::endl;

        // start to render
        {
            // TODO - 设置摄像机时间

            Timer timer("PathTracing()");
            auto  gBufferMap = gBufferPool->PopFrontBufferPtr();
            renderer->PathTracing(gBufferMap);
            gBufferPool->UpdateGBufferPool(gBufferMap);
            // TODO - 第一帧的时候没有prevPotion
        }

        //  Output FB as Image
        {
            auto gBufferMap = gBufferPool->GetBackBufferPtr();

            // fullRendering
            gBufferMap.find(GBufferType::FullRendering)
                ->second->Save("PathTracing_FullRendering.png");
            // albedo
            gBufferMap.find(GBufferType::Albedo)->second->Save("PathTracing_Albedo.png");
            // Irradiance
            gBufferMap.find(GBufferType::Irrandiance)->second->Save("PathTracing_Irradiance.png");
            // normal
            gBufferMap.find(GBufferType::Normal)->second->Save("PathTracing_Normal.png");
            // depth
            gBufferMap.find(GBufferType::Depth)->second->Save("PathTracing_Depth.png");
            // position
            gBufferMap.find(GBufferType::Position)->second->Save("PathTracing_Position.png");
            // uv
            gBufferMap.find(GBufferType::UV)->second->Save("PathTracing_UV.png");
            // motion
            gBufferMap.find(GBufferType::Motion)->second->Save("PathTracing_Motion.png");
        }

        std::cout << "PathTracing(): Rendering Finished!" << std::endl;
    }

    __host__ void Reconstruction()
    {
        // Input Images: Motion, Color, Normal, Depth, Mesh ID
        // History Input Images
        // Temporal accumulation -> Integrated Color, Integrated Moments
        // Variance estimation
        // Wavelet Filtering
    }

    __host__ void PostProcessing() {}
};

__host__ int main(int argc, char** argv)
{
    try
    {
        auto* app = new SVGFApplication();
        app->Run();
        delete app;
    }
    catch (std::runtime_error e)
    {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
