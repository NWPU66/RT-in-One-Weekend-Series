// #include <cstdlib>

// #include <chrono>
// #include <iostream>

// #include "cuda_runtime.h"
// #include "device_launch_parameters.h"

// void Add(int n, float* x, float* y)
// {
//     int index  = threadIdx.x;
//     int stride = blockDim.x;

//     for (int i = index; i < n; i += stride)
//     {
//         y[i] += x[i];
//         std::cout << "\rScanlines remaining: " << i << ' ' << std::flush;
//     }
// }

// __global__ void GPU_Add(int n, float* x, float* y)
// {
//     for (int i = 0; i < n; i++)
//     {
//         y[i] += x[i];
//         // std::cout << "\rScanlines remaining: " << i << ' ' << std::flush;
//     }
// }

// int main(int argc, char** argv)
// {
//     constexpr int N = 1 << 25;

//     // auto *x = new float[N], *y = new float[N];

//     // 分配CUDA内存
//     float *   x, *y;
//     cudaError err = cudaMallocManaged(&x, N * sizeof(float));
//     err           = cudaMallocManaged(&y, N * sizeof(float));

//     for (int i = 0; i < N; i++)
//     {
//         x[i] = 1.0f;
//         y[i] = 2.0f;
//     }

//     // 计时
//     auto start = std::chrono::system_clock::now();

//     // Add(N, x, y);
//     // 启动CUDA核函数
//     GPU_Add<<<1, 256>>>(N, x, y);
//     std::cout << "GPU_Add<<<1, 1>>>(N, x, y);" << std::endl;
//     cudaDeviceSynchronize();
//     std::cout << "cudaDeviceSynchronize();" << std::endl;

//     // 计时
//     auto end       = std::chrono::system_clock::now();
//     using timeType = std::chrono::milliseconds;
//     auto duration  = std::chrono::duration_cast<timeType>(end - start);
//     std::cout << "Time: " << duration.count() << "ms" << std::endl;

//     float maxError = 0.0f;
//     for (int i = 0; i < N; i++)
//     {
//         maxError = fmax(maxError, fabs(y[i] - 3.0f));
//     }

//     std::cout << "Max error: " << maxError << std::endl;

//     // delete[] x, y;
//     cudaFree(x);
//     cudaFree(y);

//     return EXIT_SUCCESS;
// }

#include <cstdio>
#include <cstdlib>

#include <iostream>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

template <typename T> __global__ void matAdd_cuda(T* a, T* b, T* sum)
{
    int i  = blockIdx.x * blockDim.x + threadIdx.x;
    sum[i] = a[i] + b[i];
}

float* matAdd(float* a, float* b, int length)
{
    int device = 0;  // 设置使用第0块GPU进行运算
    cudaSetDevice(device);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, device);
    int    threadMaxSize = deviceProp.maxThreadsPerBlock;
    int    blockSize     = (length + threadMaxSize - 1) / threadMaxSize;
    dim3   thread(threadMaxSize), block(blockSize);
    int    size = length * sizeof(float);
    float* sum  = (float*)malloc(size);

    // 开辟显存空间
    float *sumGPU, *aGPU, *bGPU;
    cudaMalloc((void**)&sumGPU, size);
    cudaMalloc((void**)&aGPU, size);
    cudaMalloc((void**)&bGPU, size);

    // 内存->显存
    cudaMemcpy((void*)aGPU, (void*)a, size, cudaMemcpyHostToDevice);
    cudaMemcpy((void*)bGPU, (void*)b, size, cudaMemcpyHostToDevice);

    // 运算
    matAdd_cuda<float><<<block, thread>>>(aGPU, bGPU, sumGPU);
    cudaThreadSynchronize();

    // 显存->内存
    cudaMemcpy(sum, sumGPU, size, cudaMemcpyDeviceToHost);

    // 释放显存
    cudaFree(sumGPU);
    cudaFree(aGPU);
    cudaFree(bGPU);

    return sum;
}

int main(int argc, char** argv)
{
    // 创建数组
    const int length = 10;
    float     a[length], b[length];
    for (int i = 0; i < length; i++)
    {
        a[i] = 1;
        b[i] = 2;
    }

    float* c = matAdd(a, b, length);

    // 输出查看是否完成计算
    for (int i = 0; i < length; i++)
    {
        std::cout << a[i] << " " << b[i] << " " << c[i] << std::endl;
    }

    return EXIT_SUCCESS;
}
