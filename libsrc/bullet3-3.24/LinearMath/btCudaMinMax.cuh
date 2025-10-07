/*
CUDA Conversion of Bullet Physics btMinMax utilities
Copyright (c) 2003-2006 Erwin Coumans  https://bulletphysics.org
CUDA Conversion: 2025
*/

#ifndef BT_CUDA_MINMAX_CUH
#define BT_CUDA_MINMAX_CUH

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "btCudaScalar.cuh"

/**
 * CUDA-optimized min/max operations
 */
template<class T>
__device__ __host__ inline const T& btCudaMin(const T& a, const T& b)
{
    return a < b ? a : b;
}

template<class T>
__device__ __host__ inline const T& btCudaMax(const T& a, const T& b)
{
    return a > b ? a : b;
}

template<class T>
__device__ __host__ inline const T& btCudaClamp(const T& value, const T& min, const T& max)
{
    return btCudaMin(btCudaMax(value, min), max);
}

template<class T>
__device__ __host__ inline void btCudaSwap(T& a, T& b)
{
    T tmp = a;
    a = b;
    b = tmp;
}

template<class T>
__device__ __host__ inline void btCudaSetMin(T& a, const T& b)
{
    if (b < a) {
        a = b;
    }
}

template<class T>
__device__ __host__ inline void btCudaSetMax(T& a, const T& b)
{
    if (a < b) {
        a = b;
    }
}

/**
 * Specialized for btCudaScalar
 */
__device__ __host__ inline btCudaScalar btCudaMinScalar(btCudaScalar a, btCudaScalar b)
{
#ifdef __CUDA_ARCH__
    return fminf(a, b);
#else
    return a < b ? a : b;
#endif
}

__device__ __host__ inline btCudaScalar btCudaMaxScalar(btCudaScalar a, btCudaScalar b)
{
#ifdef __CUDA_ARCH__
    return fmaxf(a, b);
#else
    return a > b ? a : b;
#endif
}

#endif // BT_CUDA_MINMAX_CUH
