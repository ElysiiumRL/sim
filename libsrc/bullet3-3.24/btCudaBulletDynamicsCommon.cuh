/*
CUDA Bullet Physics - Complete Common Header
Copyright (c) 2003-2006 Erwin Coumans  https://bulletphysics.org
CUDA Conversion: 2025

This software is provided 'as-is', without any express or implied warranty.
*/

#ifndef BT_CUDA_BULLET_DYNAMICS_COMMON_CUH
#define BT_CUDA_BULLET_DYNAMICS_COMMON_CUH

// Include CUDA runtime
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// Core CUDA Math Library
#include "LinearMath/btCudaScalar.cuh"
#include "LinearMath/btCudaMinMax.cuh"
#include "LinearMath/btCudaVector3.cuh"
#include "LinearMath/btCudaQuaternion.cuh"
#include "LinearMath/btCudaMatrix3x3.cuh"
#include "LinearMath/btCudaTransform.cuh"

// CUDA Collision Detection
#include "BulletCollision/CollisionShapes/btCudaCollisionShape.cuh"
#include "BulletCollision/CollisionDispatch/btCudaCollisionObject.cuh"
#include "BulletCollision/BroadphaseCollision/btCudaBroadphase.cuh"

// CUDA Dynamics
#include "BulletDynamics/Dynamics/btCudaRigidBody.cuh"
#include "BulletDynamics/ConstraintSolver/btCudaConstraintSolver.cuh"
#include "BulletDynamics/Dynamics/btCudaDiscreteDynamicsWorld.cuh"

/**
 * CUDA Bullet Physics Initialization
 */
inline bool initCudaBulletPhysics()
{
    // Check CUDA availability
    int deviceCount;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);
    
    if (error != cudaSuccess || deviceCount == 0) {
        printf("CUDA Bullet Physics: No CUDA devices found\n");
        return false;
    }
    
    // Get device properties
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    
    printf("CUDA Bullet Physics Initialized:\n");
    printf("  Device: %s\n", prop.name);
    printf("  Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("  Global Memory: %.1f MB\n", prop.totalGlobalMem / (1024.0f * 1024.0f));
    printf("  Shared Memory per Block: %zu bytes\n", prop.sharedMemPerBlock);
    printf("  Max Threads per Block: %d\n", prop.maxThreadsPerBlock);
    
    return true;
}

/**
 * CUDA Error Checking Utility
 */
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            printf("CUDA error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(error)); \
            exit(1); \
        } \
    } while(0)

/**
 * CUDA Kernel Launch Helper
 */
template<typename... Args>
inline void launchCudaKernel(void(*kernel)(Args...), dim3 gridSize, dim3 blockSize, Args... args)
{
    kernel<<<gridSize, blockSize>>>(args...);
    CUDA_CHECK(cudaGetLastError());
}

/**
 * CUDA Memory Management Helpers
 */
template<typename T>
inline void allocateCudaArray(T** ptr, size_t count)
{
    CUDA_CHECK(cudaMalloc(ptr, count * sizeof(T)));
}

template<typename T>
inline void freeCudaArray(T* ptr)
{
    if (ptr) {
        CUDA_CHECK(cudaFree(ptr));
    }
}

template<typename T>
inline void copyCudaArrayToHost(T* hostPtr, const T* devicePtr, size_t count)
{
    CUDA_CHECK(cudaMemcpy(hostPtr, devicePtr, count * sizeof(T), cudaMemcpyDeviceToHost));
}

template<typename T>
inline void copyCudaArrayToDevice(T* devicePtr, const T* hostPtr, size_t count)
{
    CUDA_CHECK(cudaMemcpy(devicePtr, hostPtr, count * sizeof(T), cudaMemcpyHostToDevice));
}

/**
 * CUDA Bullet Physics Configuration
 */
struct btCudaPhysicsConfig
{
    int maxRigidBodies = 1024;
    int maxCollisionObjects = 2048;
    int maxContactPoints = 4096;
    int maxConstraints = 2048;
    btCudaVector3 gravity = btCudaVector3(0, -9.81f, 0);
    btCudaScalar timeStep = 1.0f / 60.0f;
    int maxSubSteps = 1;
    
    // CUDA execution configuration
    int blockSize = 256;
    bool enableDebugOutput = false;
    bool enableProfiling = false;
};

/**
 * Complete CUDA Bullet Physics World Factory
 */
inline btCudaDiscreteDynamicsWorld* createCudaBulletWorld(const btCudaPhysicsConfig& config = btCudaPhysicsConfig())
{
    if (!initCudaBulletPhysics()) {
        return nullptr;
    }
    
    btCudaDiscreteDynamicsWorld* world = new btCudaDiscreteDynamicsWorld(
        config.maxRigidBodies,
        config.maxCollisionObjects,
        config.maxContactPoints,
        config.maxConstraints
    );
    
    world->setGravity(config.gravity);
    
    if (config.enableDebugOutput) {
        printf("CUDA Bullet World Created:\n");
        printf("  Max Rigid Bodies: %d\n", config.maxRigidBodies);
        printf("  Max Collision Objects: %d\n", config.maxCollisionObjects);
        printf("  Max Contact Points: %d\n", config.maxContactPoints);
        printf("  Max Constraints: %d\n", config.maxConstraints);
        printf("  Gravity: (%.2f, %.2f, %.2f)\n", 
               config.gravity.getX(), config.gravity.getY(), config.gravity.getZ());
    }
    
    return world;
}

/**
 * Utility functions for common physics objects
 */
inline btCudaRigidBodyConstructionInfo createCudaBoxBody(
    const btCudaVector3& halfExtents, 
    btCudaScalar mass, 
    const btCudaTransform& transform = btCudaTransform::getIdentity())
{
    // Create box shape (would need proper implementation)
    btCudaBoxShape* boxShape = new btCudaBoxShape();
    boxShape->init(halfExtents);
    
    // Calculate inertia
    btCudaVector3 inertia;
    if (mass != 0.0f) {
        btCudaScalar m = mass / 3.0f;
        inertia.setValue(
            m * (halfExtents.getY() * halfExtents.getY() + halfExtents.getZ() * halfExtents.getZ()),
            m * (halfExtents.getX() * halfExtents.getX() + halfExtents.getZ() * halfExtents.getZ()),
            m * (halfExtents.getX() * halfExtents.getX() + halfExtents.getY() * halfExtents.getY())
        );
    } else {
        inertia.setZero();
    }
    
    btCudaRigidBodyConstructionInfo info;
    info.init(mass, reinterpret_cast<btCudaCollisionShape*>(boxShape), inertia, transform);
    return info;
}

inline btCudaRigidBodyConstructionInfo createCudaSphereBody(
    btCudaScalar radius,
    btCudaScalar mass,
    const btCudaTransform& transform = btCudaTransform::getIdentity())
{
    // Create sphere shape
    btCudaSphereShape* sphereShape = new btCudaSphereShape();
    sphereShape->init(radius);
    
    // Calculate inertia
    btCudaVector3 inertia;
    if (mass != 0.0f) {
        btCudaScalar i = 0.4f * mass * radius * radius;
        inertia.setValue(i, i, i);
    } else {
        inertia.setZero();
    }
    
    btCudaRigidBodyConstructionInfo info;
    info.init(mass, reinterpret_cast<btCudaCollisionShape*>(sphereShape), inertia, transform);
    return info;
}

/**
 * Performance monitoring utilities
 */
class btCudaPerformanceMonitor
{
private:
    cudaEvent_t startEvent, stopEvent;
    float totalTime;
    int frameCount;
    
public:
    btCudaPerformanceMonitor() : totalTime(0.0f), frameCount(0)
    {
        cudaEventCreate(&startEvent);
        cudaEventCreate(&stopEvent);
    }
    
    ~btCudaPerformanceMonitor()
    {
        cudaEventDestroy(startEvent);
        cudaEventDestroy(stopEvent);
    }
    
    void startFrame()
    {
        cudaEventRecord(startEvent);
    }
    
    void endFrame()
    {
        cudaEventRecord(stopEvent);
        cudaEventSynchronize(stopEvent);
        
        float frameTime;
        cudaEventElapsedTime(&frameTime, startEvent, stopEvent);
        totalTime += frameTime;
        frameCount++;
    }
    
    float getAverageFrameTime() const
    {
        return frameCount > 0 ? totalTime / frameCount : 0.0f;
    }
    
    float getAverageFPS() const
    {
        float avgTime = getAverageFrameTime();
        return avgTime > 0.0f ? 1000.0f / avgTime : 0.0f;
    }
    
    void reset()
    {
        totalTime = 0.0f;
        frameCount = 0;
    }
    
    void printStats() const
    {
        printf("CUDA Physics Performance:\n");
        printf("  Average frame time: %.3f ms\n", getAverageFrameTime());
        printf("  Average FPS: %.1f\n", getAverageFPS());
        printf("  Total frames: %d\n", frameCount);
    }
};

#endif // BT_CUDA_BULLET_DYNAMICS_COMMON_CUH
