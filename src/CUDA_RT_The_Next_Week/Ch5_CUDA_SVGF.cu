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
#include <cstdlib>
#include <cstring>

// c++
#include <chrono>
#include <functional>
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
const int       SPP      = 100;
const int       maxDepth = 50;
// cuda concurrency
const dim3   threads(8, 8);
const dim3   blocks(imageWidth / threads.x + 1, imageHeight / threads.y + 1);
const size_t curandSeed = 3777;
std::mutex   CurandMutex;
// SVGF core
const int gBufferHistorySize = 5;
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

    __host__ void DestoryInstance() { delete singleton; }

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
    __host__ ImageBuffer(int width = ::imageWidth, int height = ::imageHeight, int channels = 3)
    {
        std::cout << "ImageBuffer(): ImageBuffer initlization Empty Buffer. " << std::endl;
        size_t _byteSize = width * height * channels * sizeof(DataType);
        std::cout << "ImageBUffer(): Buffer size: " << _byteSize << "bytes. With Width: " << width
                  << " Height: " << height << " Channels: " << channels << std::endl;

        DataType* ptr = AllocateHostData<DataType>(_byteSize);
        memset(ptr, 0, _byteSize);
        Init(ptr, width, height, channels);
    }

    /**
     * @brief create a new image buffer with data from file
     *
     * @param filePath
     */
    __host__ ImageBuffer(const std::string& filePath)
    {
        std::cout << "ImageBuffer(): ImageBuffer initlization from File. " << std::endl;
        int            width, height, channels;
        unsigned char* imageData = stbi_load(filePath.c_str(), &width, &height, &channels, 0);
        if (imageData == nullptr)
        {
            std::cerr << "load image failed" << std::endl;
            ReleaseHostData(imageData);
            return;
        }

        // transfer data type to DataType
        const size_t _byteSize = width * height * channels * sizeof(unsigned char);
        DataType*    _pHostData =
            DataTypeConvertion<unsigned char, DataType>(imageData, _byteSize, true);

        Init(_pHostData, width, height, channels);
    }
    __host__ ImageBuffer(DataType* _pHostData, int width, int height, int channels)
    {
        if (!_pHostData)
        {
            std::cerr << "Error: Host data is null" << std::endl;
            return;
        }

        Init(_pHostData, width, height, channels);
    }

    __host__ ~ImageBuffer()
    {
        if (isInHost()) { ReleaseHostData(pHostData); }
        if (isInDevice()) { ReleaseDeviceData(pDeviceData); }
    }

    __host__ void Load(const std::string& filePath) = delete;
    __host__ void Save(const std::string& filePath)
    {
        // 优先从device中保存数据，如果device中没有数据，则从host中保存数据
        if (isInDevice()) { TransferToHost(); }
        if (!isInHost())
        {
            std::cerr << "Error: ImageBuffer is either not in host or device memory" << std::endl;
            return;
        }

        unsigned char* dataToSave =
            DataTypeConvertion<DataType, unsigned char>(pHostData, byteSize, false);
        stbi_write_png(filePath.c_str(), imageWidth, imageHeight, imageChannels, dataToSave, 0);
        stbi_image_free(dataToSave);
    }

    __host__ bool TransferToDevice()
    {
        if (!isInHost())
        {
            std::cerr << "Error: ImageBuffer is not in host memory" << std::endl;
            return false;
        }
        if (!isInDevice()) { pDeviceData = AllocateDeviceData<DataType>(byteSize); }
        checkCudaErrors(cudaMemcpy(pDeviceData, pHostData, byteSize, cudaMemcpyHostToDevice));
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize());
        return true;
    }
    __host__ bool TransferToHost()
    {
        if (!isInDevice())
        {
            std::cerr << "Error: ImageBuffer is not in device memory" << std::endl;
            return false;
        }
        if (!isInHost()) { pHostData = AllocateHostData<DataType>(byteSize); }
        checkCudaErrors(cudaMemcpy(pHostData, pDeviceData, byteSize, cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize());
        return true;
    }

    __host__ __device__ DataType* GetHostData() const { return pHostData; }
    __host__ __device__ DataType* GetDeviceData() const { return pDeviceData; }
    void                          GetImageSize(int*    _imageWidth    = nullptr,
                                               int*    _imageHeight   = nullptr,
                                               int*    _imageChannels = nullptr,
                                               int*    _numPixels     = nullptr,
                                               size_t* _byteSize      = nullptr) const
    {
        if (_imageWidth != nullptr) { *_imageWidth = imageWidth; }
        if (_imageHeight != nullptr) { *_imageHeight = imageHeight; }
        if (_imageChannels != nullptr) { *_imageChannels = imageChannels; }
        if (_numPixels != nullptr) { *_numPixels = numPixels; }
        if (_byteSize != nullptr) { *_byteSize = byteSize; }
    }

    __host__ bool isInHost() const { return (pHostData != nullptr); }
    __host__ bool isInDevice() const { return (pDeviceData != nullptr); }

    template <typename SourceType, typename TargetType>
    __host__ TargetType*
    DataTypeConvertion(SourceType* pSource, size_t sourceByteSize, bool releaseSource = false)
    {
        if (pSource == nullptr)
        {
            std::cerr << "Error: Invalid pointer when converting data type" << std::endl;
            return nullptr;
        }
        if (static_cast<bool>(std::is_same<SourceType, TargetType>::value))
        {
            return reinterpret_cast<TargetType*>(pSource);  // 类型相同
        }

        const size_t numElements    = sourceByteSize / sizeof(SourceType);
        const size_t targetByteSize = numElements * sizeof(TargetType);
        TargetType*  pTarget        = (TargetType*)malloc(targetByteSize);
        for (int i = 0; i < numElements; i++)
        {
            pTarget[i] = static_cast<DataType>(pSource[i]);
        }

        if (releaseSource) { ReleaseHostData(pSource); }
        return pTarget;
    }

private:
    DataType* pHostData;
    DataType* pDeviceData;

    int    imageWidth;
    int    imageHeight;
    int    imageChannels;
    int    numPixels;
    size_t byteSize;

    __host__ void Init(DataType* _pHostData, int width, int height, int channels = 3)
    {
        std::cout << "ImageBuffer()::Init(): function called" << std::endl;

        if (!_pHostData)
        {
            std::cerr << "Error: Host data is null" << std::endl;
            return;
        }

        pHostData     = _pHostData;
        imageWidth    = width;
        imageHeight   = height;
        imageChannels = channels;
        numPixels     = width * height;
        byteSize      = numPixels * channels * sizeof(DataType);

        TransferToDevice();  // transfer data from host to device
    }

    __host__ void ReleaseHostData(DataType* dst)
    {
        if (dst != nullptr)
        {
            free(dst);
            dst = nullptr;
        }
    }
    __host__ void ReleaseDeviceData(DataType* dst)
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
    __host__ TargetDataType* AllocateHostData(size_t byteSize)
    {
        return (TargetDataType*)malloc(byteSize);
    }
    template <typename TargetDataType = DataType>
    __host__ TargetDataType* AllocateDeviceData(size_t byteSize)
    {
        TargetDataType* ptr = nullptr;
        checkCudaErrors(cudaMallocManaged((void**)&ptr, byteSize));  // FIXME -
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
        Irradiance    = 1,
        Depth         = 1,
        Normal        = 3,
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

// ANCHOR - Scene---------------------------------------------------------------------------

__device__ hitable* Transform(hitable* geometry, const vec3& offset = {0}, float rotateYAngle = 0)
{
    return new translate(new rotate_y(geometry, rotateYAngle), offset);
}

__global__ static void CreateWorld(class camera** const camera,
                                   hitable** const      geometries,
                                   material** const     materials,
                                   bvh_node** const     bvh,
                                   curandState* const   randState)
{
    if (threadIdx.x != 0 && blockIdx.x != 0) { return; }

    // create camera
    vec3  lookfrom(278, 278, -800);
    vec3  lookat(278, 278, 0);
    vec3  vup(0, 1, 0);
    auto  dist_to_focus = 10.0;
    auto  aperture      = 0.0;
    auto  vfov          = 40.0;
    auto  aspectRatio   = (float)imageWidth / (float)imageHeight;
    float time0 = 0.0, time1 = 1.0;
    camera[0] = new class camera(lookfrom, lookat, vup, vfov, aspectRatio, aperture, dist_to_focus,
                                 time0, time1);

    // create materials
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

    // create bvh
    //  we are not going to use BVH right now
    //  bvh             = new bvh_node(*geometries, 0, 1, randState);
}

/**
 * @brief
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
                        numMaterial * sizeof(material*) + sizeof(bvh_node*);
        std::cout << "Scene(): Total memory allocated: " << totalMen << " bytes." << std::endl;
        checkCudaErrors(cudaMalloc((void**)&camera, sizeof(class camera*)));
        checkCudaErrors(cudaMalloc((void**)&geometries, numGeometries * sizeof(hitable*)));
        checkCudaErrors(cudaMalloc((void**)&materials, numMaterial * sizeof(material*)));
        checkCudaErrors(cudaMalloc((void**)&bvh, sizeof(bvh_node*)));
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize());

        std::cout << "Scene(): Create world." << std::endl;
        auto randState = Curand::GetInstance()->GetCurandState();
        CreateWorld<<<0, 0>>>(camera, geometries, materials, bvh, randState);  // create world
    }
    __host__ ~Scene()
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

    __device__ camera*      GetCamera() const { return camera[0]; }
    __device__ hitable_list GetHittableList() const
    {
        return hitable_list(geometries, numGeometries);
    }

private:
    const int  numGeometries = 8;
    const int  numMaterial   = 4;
    camera**   camera;
    hitable**  geometries;
    material** materials;
    bvh_node** bvh;
};

// ANCHOR - RTRenderer---------------------------------------------------------------------------

__device__ static inline vec3 DefaultToneMappingFunc(const vec3& x)
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

__global__ void _StandardRender(ImageBuffer<float>* const renderTarget,
                                Scene* const              scene,
                                curandState* const        randStatePerPixel)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if ((i >= imageWidth) || (j >= imageHeight)) { return; }
    int pixel_index = j * imageWidth + i;

    auto* randState = randStatePerPixel + pixel_index;

    vec3 col(0);
    auto geometries = scene->GetHittableList();
    for (int s = 0; s < SPP; s++)
    {
        float u = float(i + curand_uniform(randState)) / float(imageWidth);
        float v = float(j + curand_uniform(randState)) / float(imageHeight);
        ray   r = scene->GetCamera()->get_ray(u, v, randState);
        col += rayColor(r, &geometries, randState);
    }

    reinterpret_cast<vec3*>(renderTarget->GetDeviceData())[pixel_index] =
        DefaultToneMappingFunc(col / float(SPP));  // Tone Mapping
}

/**
 * @brief
 *
 */
class RTRenderer {
public:
    using ToneMappingFunc    = std::function<vec3(const vec3&)>;
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

    void StandardRender(ImageBuffer<float>* const renderTarget) const
    {
        auto* randStatePerPixel = Curand::GetInstance()->GetCurandStatePerPixel();
        _StandardRender<<<blocks, threads>>>(renderTarget, scene, randStatePerPixel);
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize());
    }

private:
    Scene*       scene;
    GBufferPool* gBufferPool;

    // __global__ static void RenderGBuffer();
    // __global__ static void TemporalAA();
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
        gBufferPool = new GBufferPool({GBufferPool::GBufferType::FullRendering});
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

        // PreSVGFPocess();
        // SVGFPipline();
        // PostSVGFPocess();
    }

private:
    RTRenderer*  renderer;
    GBufferPool* gBufferPool;
    Scene*       scene;

    __host__ void StandardRender()
    {
        std::cout << "Rendering a " << imageWidth << "x" << imageHeight << " image with " << SPP
                  << " samples per pixel ";
        std::cout << "in " << threads.x << "x" << threads.y << " blocks.\n";

        // import an image as a test
        auto* earth_img = new ImageBuffer<unsigned char>("earth.jpg");

        // start to render
        auto renderTarget = GBufferPool::GBufferType::FullRendering;
        {
            auto start = std::chrono::system_clock::now();

            auto gBufferMap = gBufferPool->PopFrontBufferPtr();
            renderer->StandardRender(gBufferMap.find(renderTarget)->second);
            gBufferPool->UpdateGBufferPool(gBufferMap);

            auto stop     = std::chrono::system_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
            std::cout << "Time:  " << duration.count() / 1000.0f << " s\n";
        }

        //  Output FB as Image
        {
            auto frameBuffer = gBufferPool->GetBackBufferPtr().find(renderTarget)->second;
            frameBuffer->Save(savePath);
        }
    }

    __host__ void PreSVGFPocess();

    __host__ void SVGFPipline()
    {
        //......
        // ReconstructionFilter();
        //......
    }

    __host__ void ReconstructionFilter()
    {
        // Input Images: Motion, Color, Normal, Depth, Mesh ID
        // History Input Images
        // Temporal accumulation -> Integrated Color, Integrated Moments
        // Variance estimation
        // Wavelet Filtering
    }

    __host__ void PostSVGFPocess();
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
