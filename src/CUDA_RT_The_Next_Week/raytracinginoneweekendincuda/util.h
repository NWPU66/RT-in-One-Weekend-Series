#ifndef UTILH
#define UTILH

#include <cstddef>
#include <math.h>

#include <functional>

template <typename T>  // double or float
__host__ __device__ inline T mapping(T x, T tar_min, T tar_max, T src_min = 0.0, T src_max = 1.0)
{
    return tar_min + (tar_max - tar_min) * ((x - src_min) / (src_max - src_min));
}

template <typename T>
__host__ __device__ void bubble_sort(T* x, size_t start, size_t end, bool cmp(const T, const T))
{
    for (int i = start; i < end - 1; i++)
    {
        bool swapped = false;
        for (int j = start; j < end - i - 1; j++)
        {
            if (!cmp(x[j], x[j + 1]))
            {
                swapped  = true;
                T temp   = x[j];
                x[j]     = x[j + 1];
                x[j + 1] = temp;
            }
        }

        if (!swapped) { break; }
    }
}

#endif