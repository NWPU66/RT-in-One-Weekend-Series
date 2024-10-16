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
#include <cstddef>
#include <sal.h>
#ifdef DONOTUSE
#    include <__clang_cuda_runtime_wrapper.h>
#endif
#include <cstdlib>

// c++
#include <iostream>
#include <memory>
#include <queue>
#include <stdexcept>
#include <string>
#include <vector>

// cuda
#include "cuda_runtime.h"
#include "curand_kernel.h"

// user
#include "raytracinginoneweekendincuda/bvh.h"
#include "raytracinginoneweekendincuda/camera.h"
#include "raytracinginoneweekendincuda/hitable.h"
#include "raytracinginoneweekendincuda/hitable_list.h"

// 3rdparty
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

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
// basic rendering settings
const int     imageWidth  = 800;
const int     imageHeight = 400;
constexpr int numPixels   = imageWidth * imageHeight;
const int     SPP         = 100;
const int     maxDepth    = 50;
// save settings

// FIXME - 注意cudaDeviceSynchronize的问题
/**
 * @brief
 *
 */
template <typename RenderDataType = float, typename SaveDataType = unsigned char>
class ImageBuffer {
public:
    ImageBuffer(const std::string& filePath)
    {
        // TODO -
        //......
        Init(pHostData, width, height, channels);
    }
    ImageBuffer(RenderDataType* pHostData, int width, int height, int channels)
    {
        Init(pHostData, width, height, channels);
    }
    ~ImageBuffer()
    {
        ReleaseHostData();
        ReleaseDeviceData();
        checkCudaErrors(cudaDeviceSynchronize());
    }

    __host__ void Save(const std::string& filePath);
    __host__ void Load(const std::string& filePath);

    __host__ void TransferToDevice()
    {
        if (!isInHost()) { AllocateHostData(); }
        if (!isInDevice()) { AllocateDeviceData(); }
        checkCudaErrors(cudaMemcpy(pDeviceData, pHostData, byteSize, cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize());
    }
    __host__ void TransferToHost()
    {
        if (!isInHost()) { AllocateHostData(); }
        if (!isInDevice()) { AllocateDeviceData(); }
        checkCudaErrors(cudaMemcpy(pHostData, pDeviceData, byteSize, cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize());
    }

    __host__ __device__ RenderDataType* GetHostData() { return pHostData; }
    __host__ __device__ RenderDataType* GetDeviceData() { return pDeviceData; }

    __host__ __device__ bool isInHost() const { return (pHostData != nullptr); }
    __host__ __device__ bool isInDevice() const { return (pDeviceData != nullptr); }

private:
    RenderDataType* pHostData;
    RenderDataType* pDeviceData;
    int             imageWidth;
    int             imageHeight;
    int             imageChannels;
    int             numPixels;
    size_t          byteSize;

    void Init(RenderDataType* pHostData, int width, int height, int channels)
    {
        pHostData     = pHostData;
        imageWidth    = width;
        imageHeight   = height;
        imageChannels = channels;
        numPixels     = width * height;
        byteSize      = numPixels * imageChannels * sizeof(RenderDataType);

        AllocateDeviceData(byteSize);
        TransferToDevice();
    }

    __host__ void ReleaseHostData()
    {
        if (!isInHost()) { return; }
        free(pHostData);
        pHostData = nullptr;
    }
    __host__ void ReleaseDeviceData()
    {
        if (!isInDevice()) { return; }
        checkCudaErrors(cudaFree(pDeviceData));
        checkCudaErrors(cudaGetLastError());
        pDeviceData = nullptr;
    }

    __host__ void AllocateHostData(size_t byteSize)
    {
        pHostData = (RenderDataType*)malloc(byteSize);
    }
    __host__ void AllocateDeviceData(size_t byteSize)
    {
        checkCudaErrors(cudaMalloc((void**)&pDeviceData, byteSize));
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize());
    }
};

/**
 * @brief
 *
 */
class GBufferPool {
public:
    GBufferPool();
    ~GBufferPool();

    void UpdateGBufferPool()
    {
        // ANCHOR - 从队首取出一个buffer指针，直接修改指针上的内存区，然后放回队尾
    }

private:
    std::queue<ImageBuffer*> Albedo;
    std::queue<ImageBuffer*> Irradiance;  // Irradiance = source / texture albedo
    std::queue<ImageBuffer*> Depth;
    std::queue<ImageBuffer*> Normal;
    std::queue<ImageBuffer*> Motion;
};

/**
 * @brief
 *
 */
class Scene {
public:
    Scene();
    ~Scene();

    camera* GetCamera() const { return camera; }

private:
    camera*               camera;
    std::vector<hitable*> geometries;
    hitable_list*         geometryList;
    bvh_node*             bvh;
};

/**
 * @brief
 *
 */
class RTRenderer {
public:
    RTRenderer();
    ~RTRenderer();

    Scene* GetScene() const { return scene; }

private:
    Scene*       scene;
    GBufferPool* gBufferPool;

    void RenderGBuffer();
    void ToneMapping();
    void TemporalAA();
};

/**
 * @brief
 *
 */
class SVGFApplication {
public:
    __host__ SVGFApplication() : renderer(new RTRenderer()) {}
    __host__ ~SVGFApplication() { delete renderer; }

    __host__ void Run()
    {
        PreSVGFPocess();
        SVGFPipline();
        PostSVGFPocess();
    }

private:
    RTRenderer* renderer;

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

int main(int argc, char** argv)
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
