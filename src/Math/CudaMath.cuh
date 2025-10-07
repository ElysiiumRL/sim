#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <curand_kernel.h>
#include <vector_types.h>
#include <vector_functions.h>
#include <math_functions.h>

#include "Math.h"

#ifdef RS_CUDA_ENABLED

namespace RocketSim {
namespace CudaMath {

// CUDA Vector Types for faster GPU computation
struct CudaVec3 {
    float x, y, z;
    
    __host__ __device__ CudaVec3() : x(0), y(0), z(0) {}
    __host__ __device__ CudaVec3(float x, float y, float z) : x(x), y(y), z(z) {}
    __host__ __device__ CudaVec3(const Vec& v) : x(v.x), y(v.y), z(v.z) {}
    
    __host__ __device__ operator Vec() const { return Vec(x, y, z); }
    
    __device__ CudaVec3 operator+(const CudaVec3& other) const {
        return CudaVec3(x + other.x, y + other.y, z + other.z);
    }
    
    __device__ CudaVec3 operator-(const CudaVec3& other) const {
        return CudaVec3(x - other.x, y - other.y, z - other.z);
    }
    
    __device__ CudaVec3 operator*(float scalar) const {
        return CudaVec3(x * scalar, y * scalar, z * scalar);
    }
    
    __device__ float dot(const CudaVec3& other) const {
        return x * other.x + y * other.y + z * other.z;
    }
    
    __device__ CudaVec3 cross(const CudaVec3& other) const {
        return CudaVec3(
            y * other.z - z * other.y,
            z * other.x - x * other.z,
            x * other.y - y * other.x
        );
    }
    
    __device__ float length() const {
        return sqrtf(x * x + y * y + z * z);
    }
    
    __device__ float lengthSq() const {
        return x * x + y * y + z * z;
    }
    
    __device__ CudaVec3 normalized() const {
        float len = length();
        return len > 0 ? *this * (1.0f / len) : CudaVec3();
    }
};

struct CudaRotMat {
    float m[3][3];
    
    __host__ __device__ CudaRotMat() {
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                m[i][j] = (i == j) ? 1.0f : 0.0f;
            }
        }
    }
    
    __host__ __device__ CudaRotMat(const RotMat& mat) {
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                m[i][j] = mat[i][j];
            }
        }
    }
    
    __device__ CudaVec3 operator*(const CudaVec3& vec) const {
        return CudaVec3(
            m[0][0] * vec.x + m[0][1] * vec.y + m[0][2] * vec.z,
            m[1][0] * vec.x + m[1][1] * vec.y + m[1][2] * vec.z,
            m[2][0] * vec.x + m[2][1] * vec.y + m[2][2] * vec.z
        );
    }
};

// High-performance physics state for GPU
struct CudaPhysState {
    CudaVec3 pos;
    CudaVec3 vel;
    CudaVec3 angVel;
    CudaRotMat rotMat;
    
    __host__ __device__ CudaPhysState() {}
    __host__ __device__ CudaPhysState(const PhysState& state) 
        : pos(state.pos), vel(state.vel), angVel(state.angVel), rotMat(state.rotMat) {}
};

// CUDA Physics constants for faster access
__constant__ float CUDA_GRAVITY = -650.0f;
__constant__ float CUDA_BALL_RADIUS = 92.75f;
__constant__ float CUDA_CAR_HEIGHT = 36.16f;
__constant__ float CUDA_BOOST_ACCEL = 991.666f;

// Fast math functions
__device__ inline float fastSqrtf(float x) {
    return __fsqrt_rn(x);
}

__device__ inline float fastInvSqrtf(float x) {
    return __frsqrt_rn(x);
}

__device__ inline float fastSin(float x) {
    return __sinf(x);
}

__device__ inline float fastCos(float x) {
    return __cosf(x);
}

} // namespace CudaMath
} // namespace RocketSim

#endif // RS_CUDA_ENABLED
