/*
CUDA Conversion of Bullet Physics btScalar
Copyright (c) 2003-2006 Erwin Coumans  https://bulletphysics.org
CUDA Conversion: 2025

This software is provided 'as-is', without any express or implied warranty.
*/

#ifndef BT_CUDA_SCALAR_CUH
#define BT_CUDA_SCALAR_CUH

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <float.h>
#include <math.h>

// CUDA scalar type configuration
#ifdef BT_USE_DOUBLE_PRECISION
typedef double btCudaScalar;
#define CUDA_EPSILON DBL_EPSILON
#define CUDA_INFINITY DBL_MAX
#else
typedef float btCudaScalar;
#define CUDA_EPSILON FLT_EPSILON
#define CUDA_INFINITY FLT_MAX
#endif

// CUDA math constants
#define CUDA_PI 3.14159265358979323846f
#define CUDA_2_PI (2.0f * CUDA_PI)
#define CUDA_HALF_PI (CUDA_PI * 0.5f)
#define CUDA_RADS_PER_DEG (CUDA_PI / 180.0f)
#define CUDA_DEGS_PER_RAD (180.0f / CUDA_PI)

/**
 * CUDA-optimized math functions
 */
template <typename T>
__device__ __host__ inline T btCudaMin(const T& a, const T& b)
{
#ifdef __CUDA_ARCH__
    return fminf(a, b);
#else
    return (a < b) ? a : b;
#endif
}

template <typename T>
__device__ __host__ inline T btCudaMax(const T& a, const T& b)
{
#ifdef __CUDA_ARCH__
    return fmaxf(a, b);
#else
    return (a > b) ? a : b;
#endif
}

__device__ __host__ inline btCudaScalar btCudaSqrt(btCudaScalar x)
{
#ifdef __CUDA_ARCH__
    return sqrtf(x);
#else
    return sqrt(x);
#endif
}

__device__ __host__ inline btCudaScalar btCudaFabs(btCudaScalar x)
{
#ifdef __CUDA_ARCH__
    return fabsf(x);
#else
    return fabs(x);
#endif
}

__device__ __host__ inline btCudaScalar btCudaCos(btCudaScalar x)
{
#ifdef __CUDA_ARCH__
    return cosf(x);
#else
    return cos(x);
#endif
}

__device__ __host__ inline btCudaScalar btCudaSin(btCudaScalar x)
{
#ifdef __CUDA_ARCH__
    return sinf(x);
#else
    return sin(x);
#endif
}

__device__ __host__ inline btCudaScalar btCudaTan(btCudaScalar x)
{
#ifdef __CUDA_ARCH__
    return tanf(x);
#else
    return tan(x);
#endif
}

__device__ __host__ inline btCudaScalar btCudaAcos(btCudaScalar x)
{
#ifdef __CUDA_ARCH__
    return acosf(x);
#else
    return acos(x);
#endif
}

__device__ __host__ inline btCudaScalar btCudaAsin(btCudaScalar x)
{
#ifdef __CUDA_ARCH__
    return asinf(x);
#else
    return asin(x);
#endif
}

__device__ __host__ inline btCudaScalar btCudaAtan(btCudaScalar x)
{
#ifdef __CUDA_ARCH__
    return atanf(x);
#else
    return atan(x);
#endif
}

__device__ __host__ inline btCudaScalar btCudaAtan2(btCudaScalar y, btCudaScalar x)
{
#ifdef __CUDA_ARCH__
    return atan2f(y, x);
#else
    return atan2(y, x);
#endif
}

__device__ __host__ inline btCudaScalar btCudaPow(btCudaScalar x, btCudaScalar y)
{
#ifdef __CUDA_ARCH__
    return powf(x, y);
#else
    return pow(x, y);
#endif
}

/**
 * Utility functions
 */
__device__ __host__ inline bool btCudaEqual(btCudaScalar a, btCudaScalar b)
{
    return btCudaFabs(a - b) <= CUDA_EPSILON;
}

__device__ __host__ inline bool btCudaGreaterEqual(btCudaScalar a, btCudaScalar b)
{
    return a >= b - CUDA_EPSILON;
}

__device__ __host__ inline btCudaScalar btCudaClamp(btCudaScalar value, btCudaScalar min, btCudaScalar max)
{
    return btCudaMin(btCudaMax(value, min), max);
}

__device__ __host__ inline btCudaScalar btCudaDegrees(btCudaScalar radians)
{
    return radians * CUDA_DEGS_PER_RAD;
}

__device__ __host__ inline btCudaScalar btCudaRadians(btCudaScalar degrees)
{
    return degrees * CUDA_RADS_PER_DEG;
}

/**
 * Fast reciprocal square root - GPU optimized
 */
__device__ __host__ inline btCudaScalar btCudaRsqrt(btCudaScalar x)
{
#ifdef __CUDA_ARCH__
    return rsqrtf(x);  // Use hardware accelerated rsqrt on GPU
#else
    return btCudaScalar(1.0) / btCudaSqrt(x);
#endif
}

#endif // BT_CUDA_SCALAR_CUH
