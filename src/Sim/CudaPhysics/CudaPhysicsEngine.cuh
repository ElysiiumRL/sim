#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <vector>
#include <memory>

#include "../../Math/CudaMath.cuh"
#include "../Car/Car.h"
#include "../Ball/Ball.h"

#ifdef RS_CUDA_ENABLED

namespace RocketSim {
namespace CudaPhysics {

// GPU memory-aligned structures for maximum performance
struct alignas(16) CudaCarState {
    CudaMath::CudaPhysState physState;
    float boost;
    bool isOnGround;
    bool hasJumped;
    bool hasDoubleJumped;
    bool hasFlipped;
    float jumpTime;
    float flipTime;
    CudaMath::CudaVec3 flipRelTorque;
    float wheelContactTime[4];
    int team;
    uint32_t id;
};

struct alignas(16) CudaBallState {
    CudaMath::CudaPhysState physState;
    float radius;
    float mass;
    int chargeLevel;
    float accumulatedHitForce;
    bool hasDamaged;
};

struct alignas(16) CudaCarControls {
    float throttle;
    float steer;
    float pitch;
    float yaw;
    float roll;
    bool jump;
    bool boost;
    bool handbrake;
};

struct alignas(16) CudaCollisionInfo {
    CudaMath::CudaVec3 contactPoint;
    CudaMath::CudaVec3 normal;
    float penetration;
    int objectA;
    int objectB;
    int type; // 0=car-ball, 1=car-car, 2=car-world, 3=ball-world
};

// GPU Physics Engine Class
class CudaPhysicsEngine {
private:
    // Device memory pointers
    CudaCarState* d_carStates;
    CudaBallState* d_ballState;
    CudaCarControls* d_carControls;
    CudaCollisionInfo* d_collisions;
    float* d_arenaVertices;
    int* d_arenaIndices;
    
    // Host memory for synchronization
    std::vector<CudaCarState> h_carStates;
    CudaBallState h_ballState;
    std::vector<CudaCarControls> h_carControls;
    std::vector<CudaCollisionInfo> h_collisions;
    
    // Configuration
    int maxCars;
    int maxCollisions;
    float tickTime;
    int arenaVertexCount;
    int arenaIndexCount;
    
    // CUDA streams for parallel execution
    cudaStream_t physicsStream;
    cudaStream_t collisionStream;
    cudaStream_t memoryStream;
    
public:
    CudaPhysicsEngine(int maxCars = 64, int maxCollisions = 1024);
    ~CudaPhysicsEngine();
    
    // Initialize GPU resources
    bool Initialize();
    void Cleanup();
    
    // Data transfer methods
    void UploadCarStates(const std::vector<Car*>& cars);
    void UploadBallState(const Ball* ball);
    void UploadCarControls(const std::vector<CarControls>& controls);
    void UploadArenaGeometry(const std::vector<float>& vertices, const std::vector<int>& indices);
    
    void DownloadCarStates(std::vector<Car*>& cars);
    void DownloadBallState(Ball* ball);
    void DownloadCollisions(std::vector<CudaCollisionInfo>& collisions);
    
    // Main simulation methods
    void SimulatePhysicsStep(float deltaTime, int numCars);
    void UpdateCarPhysics(int numCars);
    void UpdateBallPhysics();
    void DetectCollisions(int numCars);
    void ResolveCollisions();
    
    // Async methods for maximum performance
    void BeginPhysicsStep(float deltaTime, int numCars);
    void EndPhysicsStep();
    
    // Performance monitoring
    float GetLastStepTime() const { return lastStepTime; }
    long long GetTotalSteps() const { return totalSteps; }
    
private:
    float lastStepTime;
    long long totalSteps;
    
    // CUDA event timing
    cudaEvent_t startEvent, stopEvent;
    
    // Error checking
    bool CheckCudaError(const char* operation);
};

// CUDA Kernel declarations
extern "C" {
    void LaunchCarPhysicsKernel(
        CudaCarState* carStates,
        CudaCarControls* carControls,
        float deltaTime,
        int numCars,
        cudaStream_t stream
    );
    
    void LaunchBallPhysicsKernel(
        CudaBallState* ballState,
        float deltaTime,
        cudaStream_t stream
    );
    
    void LaunchCollisionDetectionKernel(
        CudaCarState* carStates,
        CudaBallState* ballState,
        float* arenaVertices,
        int* arenaIndices,
        CudaCollisionInfo* collisions,
        int numCars,
        int arenaVertexCount,
        int arenaIndexCount,
        int maxCollisions,
        cudaStream_t stream
    );
    
    void LaunchCollisionResponseKernel(
        CudaCarState* carStates,
        CudaBallState* ballState,
        CudaCollisionInfo* collisions,
        int numCollisions,
        float deltaTime,
        cudaStream_t stream
    );
    
    void LaunchBatchCarUpdateKernel(
        CudaCarState* carStates,
        CudaCarControls* carControls,
        CudaCollisionInfo* collisions,
        float deltaTime,
        int numCars,
        int numCollisions,
        cudaStream_t stream
    );
}

} // namespace CudaPhysics
} // namespace RocketSim

#endif // RS_CUDA_ENABLED
