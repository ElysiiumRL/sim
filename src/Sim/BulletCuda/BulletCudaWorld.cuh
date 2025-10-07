#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <vector>
#include <memory>

#include "../../Math/CudaMath.cuh"
#include "../../../libsrc/bullet3-3.24/BulletDynamics/Dynamics/btDiscreteDynamicsWorld.h"

#ifdef RS_CUDA_ENABLED

namespace RocketSim {
namespace BulletCuda {

// CUDA-accelerated replacement for btDiscreteDynamicsWorld
// This replaces the core Bullet Physics simulation loop with GPU kernels

struct CudaRigidBody {
    CudaMath::CudaVec3 position;
    CudaMath::CudaVec3 linearVelocity;
    CudaMath::CudaVec3 angularVelocity;
    CudaMath::CudaRotMat orientation;
    
    float mass;
    float invMass;
    CudaMath::CudaVec3 localInertia;
    CudaMath::CudaVec3 invInertia;
    
    float friction;
    float restitution;
    float linearDamping;
    float angularDamping;
    
    int collisionShape; // Index to shape array
    int objectType; // 0=car, 1=ball, 2=static
    uint32_t id;
    
    bool isActive;
    bool isKinematic;
    bool isStatic;
};

struct CudaCollisionShape {
    int shapeType; // 0=box, 1=sphere, 2=mesh
    CudaMath::CudaVec3 halfExtents; // For box
    float radius; // For sphere
    int meshVertexOffset; // For mesh
    int meshVertexCount;
    int meshIndexOffset;
    int meshIndexCount;
};

struct CudaConstraint {
    int bodyA, bodyB;
    CudaMath::CudaVec3 pivotA, pivotB;
    float breakingThreshold;
    bool isActive;
};

struct CudaContactPoint {
    CudaMath::CudaVec3 pointA, pointB;
    CudaMath::CudaVec3 normal;
    float distance;
    float appliedImpulse;
    int bodyA, bodyB;
    bool isValid;
};

class BulletCudaWorld {
public:
    BulletCudaWorld(int maxBodies = 256, int maxContacts = 2048, int maxConstraints = 128);
    ~BulletCudaWorld();
    
    // Initialize CUDA physics world
    bool Initialize();
    void Cleanup();
    
    // Body management
    int AddRigidBody(const CudaRigidBody& body);
    void RemoveRigidBody(int bodyId);
    void UpdateRigidBody(int bodyId, const CudaRigidBody& body);
    CudaRigidBody GetRigidBody(int bodyId);
    
    // Shape management
    int AddCollisionShape(const CudaCollisionShape& shape);
    void SetMeshData(const std::vector<float>& vertices, const std::vector<int>& indices);
    
    // Physics simulation - THIS IS THE KEY REPLACEMENT FOR BULLET
    void StepSimulation(float deltaTime, int maxSubSteps = 1, float fixedTimeStep = 1.0f/60.0f);
    
    // World properties
    void SetGravity(const CudaMath::CudaVec3& gravity);
    CudaMath::CudaVec3 GetGravity() const { return h_gravity; }
    
    // Contact and collision queries
    std::vector<CudaContactPoint> GetContactPoints();
    bool RayTest(const CudaMath::CudaVec3& from, const CudaMath::CudaVec3& to, int& hitBody, CudaMath::CudaVec3& hitPoint);
    
    // Performance monitoring
    float GetLastStepTime() const { return lastStepTime; }
    long long GetStepCount() const { return stepCount; }
    
    // Synchronization with CPU Bullet world (for compatibility)
    void SyncFromBulletWorld(btDiscreteDynamicsWorld* world);
    void SyncToBulletWorld(btDiscreteDynamicsWorld* world);
    
private:
    // Device memory pointers
    CudaRigidBody* d_bodies;
    CudaCollisionShape* d_shapes;
    CudaContactPoint* d_contacts;
    CudaConstraint* d_constraints;
    float* d_meshVertices;
    int* d_meshIndices;
    
    // Host memory for synchronization
    std::vector<CudaRigidBody> h_bodies;
    std::vector<CudaCollisionShape> h_shapes;
    std::vector<CudaContactPoint> h_contacts;
    std::vector<CudaConstraint> h_constraints;
    CudaMath::CudaVec3 h_gravity;
    
    // Configuration
    int maxBodies, maxContacts, maxConstraints;
    int numBodies, numShapes, numConstraints;
    int meshVertexCount, meshIndexCount;
    
    // CUDA streams for parallel execution
    cudaStream_t integrationStream;
    cudaStream_t collisionStream;
    cudaStream_t constraintStream;
    
    // Performance tracking
    float lastStepTime;
    long long stepCount;
    cudaEvent_t startEvent, stopEvent;
    
    // Internal methods
    void IntegrateMotion(float deltaTime);
    void BroadPhaseCollision();
    void NarrowPhaseCollision();
    void SolveConstraints(float deltaTime);
    void UpdateTransforms();
    
    bool CheckCudaError(const char* operation);
};

// CUDA Kernel declarations for Bullet Physics replacement
extern "C" {
    void LaunchIntegrationKernel(
        CudaRigidBody* bodies,
        int numBodies,
        CudaMath::CudaVec3 gravity,
        float deltaTime,
        cudaStream_t stream
    );
    
    void LaunchBroadPhaseKernel(
        CudaRigidBody* bodies,
        CudaCollisionShape* shapes,
        CudaContactPoint* contacts,
        int numBodies,
        int maxContacts,
        cudaStream_t stream
    );
    
    void LaunchNarrowPhaseKernel(
        CudaRigidBody* bodies,
        CudaCollisionShape* shapes,
        CudaContactPoint* contacts,
        float* meshVertices,
        int* meshIndices,
        int numBodies,
        int numContacts,
        cudaStream_t stream
    );
    
    void LaunchConstraintSolverKernel(
        CudaRigidBody* bodies,
        CudaContactPoint* contacts,
        CudaConstraint* constraints,
        int numBodies,
        int numContacts,
        int numConstraints,
        float deltaTime,
        int numIterations,
        cudaStream_t stream
    );
    
    void LaunchRocketLeaguePhysicsKernel(
        CudaRigidBody* bodies,
        CudaContactPoint* contacts,
        int numBodies,
        int numContacts,
        float deltaTime,
        cudaStream_t stream
    );
}

} // namespace BulletCuda
} // namespace RocketSim

#endif // RS_CUDA_ENABLED
