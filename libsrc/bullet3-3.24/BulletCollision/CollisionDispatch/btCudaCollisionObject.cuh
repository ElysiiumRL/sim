/*
CUDA Conversion of Bullet Physics btCollisionObject
Copyright (c) 2003-2006 Erwin Coumans  https://bulletphysics.org
CUDA Conversion: 2025

This software is provided 'as-is', without any express or implied warranty.
*/

#ifndef BT_CUDA_COLLISION_OBJECT_CUH
#define BT_CUDA_COLLISION_OBJECT_CUH

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "../../LinearMath/btCudaTransform.cuh"
#include "../../LinearMath/btCudaVector3.cuh"
#include "../CollisionShapes/btCudaCollisionShape.cuh"

// Activation states for GPU collision objects
#define CUDA_ACTIVE_TAG 1
#define CUDA_ISLAND_SLEEPING 2
#define CUDA_WANTS_DEACTIVATION 3
#define CUDA_DISABLE_DEACTIVATION 4
#define CUDA_DISABLE_SIMULATION 5

// Collision object types
#define CUDA_COLLISION_OBJECT 1
#define CUDA_RIGID_BODY 2
#define CUDA_GHOST_OBJECT 4
#define CUDA_SOFT_BODY 8
#define CUDA_HF_FLUID 16
#define CUDA_CHARACTER_OBJECT 32

/**
 * CUDA-accelerated Collision Object
 * Manages collision detection data on GPU
 */
struct btCudaCollisionObject
{
    // Transform data
    btCudaTransform m_worldTransform;
    btCudaTransform m_interpolationWorldTransform;
    
    // Bounding volume
    btCudaVector3 m_aabbMin;
    btCudaVector3 m_aabbMax;
    
    // Collision shape (pointer to GPU memory)
    btCudaCollisionShape* m_collisionShape;
    
    // Object properties
    int m_objectType;
    int m_activationState;
    int m_islandTag;
    int m_companionId;
    int m_worldArrayIndex;
    
    // Collision filtering
    short int m_collisionFilterGroup;
    short int m_collisionFilterMask;
    
    // Friction and restitution
    btCudaScalar m_friction;
    btCudaScalar m_rollingFriction;
    btCudaScalar m_spinningFriction;
    btCudaScalar m_restitution;
    
    // Deactivation
    btCudaScalar m_deactivationTime;
    
    // User data (index to GPU array)
    int m_userIndex;
    
    /**
     * GPU device functions
     */
    __device__ void setWorldTransform(const btCudaTransform& worldTrans)
    {
        m_worldTransform = worldTrans;
        updateAabb();
    }
    
    __device__ const btCudaTransform& getWorldTransform() const
    {
        return m_worldTransform;
    }
    
    __device__ void setInterpolationWorldTransform(const btCudaTransform& trans)
    {
        m_interpolationWorldTransform = trans;
    }
    
    __device__ const btCudaTransform& getInterpolationWorldTransform() const
    {
        return m_interpolationWorldTransform;
    }
    
    __device__ void setCollisionShape(btCudaCollisionShape* collisionShape)
    {
        m_collisionShape = collisionShape;
        updateAabb();
    }
    
    __device__ const btCudaCollisionShape* getCollisionShape() const
    {
        return m_collisionShape;
    }
    
    __device__ btCudaCollisionShape* getCollisionShape()
    {
        return m_collisionShape;
    }
    
    __device__ void setActivationState(int newState)
    {
        if (m_activationState != CUDA_DISABLE_DEACTIVATION && m_activationState != CUDA_DISABLE_SIMULATION) {
            m_activationState = newState;
        }
    }
    
    __device__ int getActivationState() const
    {
        return m_activationState;
    }
    
    __device__ void forceActivationState(int newState)
    {
        m_activationState = newState;
    }
    
    __device__ void activate(bool forceActivation = false)
    {
        if (forceActivation || !(m_collisionFilterGroup & (CUDA_DISABLE_DEACTIVATION | CUDA_DISABLE_SIMULATION))) {
            setActivationState(CUDA_ACTIVE_TAG);
            m_deactivationTime = btCudaScalar(0.0);
        }
    }
    
    __device__ bool isActive() const
    {
        return (getActivationState() != CUDA_ISLAND_SLEEPING && getActivationState() != CUDA_DISABLE_SIMULATION);
    }
    
    __device__ void setRestitution(btCudaScalar rest)
    {
        m_restitution = rest;
    }
    
    __device__ btCudaScalar getRestitution() const
    {
        return m_restitution;
    }
    
    __device__ void setFriction(btCudaScalar frict)
    {
        m_friction = frict;
    }
    
    __device__ btCudaScalar getFriction() const
    {
        return m_friction;
    }
    
    __device__ void setRollingFriction(btCudaScalar frict)
    {
        m_rollingFriction = frict;
    }
    
    __device__ btCudaScalar getRollingFriction() const
    {
        return m_rollingFriction;
    }
    
    __device__ void setSpinningFriction(btCudaScalar frict)
    {
        m_spinningFriction = frict;
    }
    
    __device__ btCudaScalar getSpinningFriction() const
    {
        return m_spinningFriction;
    }
    
    /**
     * Update AABB based on current transform and collision shape
     */
    __device__ void updateAabb()
    {
        if (m_collisionShape) {
            m_collisionShape->getAabb(m_worldTransform, m_aabbMin, m_aabbMax);
        }
    }
    
    /**
     * Get AABB
     */
    __device__ void getAabb(btCudaVector3& aabbMin, btCudaVector3& aabbMax) const
    {
        aabbMin = m_aabbMin;
        aabbMax = m_aabbMax;
    }
    
    /**
     * Check if point is inside AABB
     */
    __device__ bool isPointInsideAabb(const btCudaVector3& point) const
    {
        return (point.getX() >= m_aabbMin.getX() && point.getX() <= m_aabbMax.getX() &&
                point.getY() >= m_aabbMin.getY() && point.getY() <= m_aabbMax.getY() &&
                point.getZ() >= m_aabbMin.getZ() && point.getZ() <= m_aabbMax.getZ());
    }
    
    /**
     * Check AABB overlap with another collision object
     */
    __device__ bool testAabbAgainstAabb(const btCudaCollisionObject& other) const
    {
        return !(m_aabbMax.getX() < other.m_aabbMin.getX() || other.m_aabbMax.getX() < m_aabbMin.getX() ||
                 m_aabbMax.getY() < other.m_aabbMin.getY() || other.m_aabbMax.getY() < m_aabbMin.getY() ||
                 m_aabbMax.getZ() < other.m_aabbMin.getZ() || other.m_aabbMax.getZ() < m_aabbMin.getZ());
    }
    
    /**
     * Collision filtering
     */
    __device__ void setCollisionFlags(int flags)
    {
        m_objectType = flags;
    }
    
    __device__ int getCollisionFlags() const
    {
        return m_objectType;
    }
    
    __device__ bool hasContactResponse() const
    {
        return (m_objectType & CUDA_COLLISION_OBJECT) != 0;
    }
    
    /**
     * Initialize default values
     */
    __device__ __host__ void init()
    {
        m_worldTransform.setIdentity();
        m_interpolationWorldTransform.setIdentity();
        m_aabbMin.setValue(-1e30f, -1e30f, -1e30f);
        m_aabbMax.setValue(1e30f, 1e30f, 1e30f);
        m_collisionShape = nullptr;
        m_objectType = CUDA_COLLISION_OBJECT;
        m_activationState = CUDA_ACTIVE_TAG;
        m_islandTag = -1;
        m_companionId = -1;
        m_worldArrayIndex = -1;
        m_collisionFilterGroup = 1;
        m_collisionFilterMask = -1;
        m_friction = btCudaScalar(0.5);
        m_rollingFriction = btCudaScalar(0.0);
        m_spinningFriction = btCudaScalar(0.0);
        m_restitution = btCudaScalar(0.0);
        m_deactivationTime = btCudaScalar(0.0);
        m_userIndex = -1;
    }
};

/**
 * GPU kernel for updating collision object AABBs
 */
__global__ void updateCollisionObjectAabbs(btCudaCollisionObject* objects, int numObjects)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numObjects) return;
    
    objects[idx].updateAabb();
}

/**
 * GPU kernel for AABB overlap testing
 */
__global__ void testAabbOverlaps(btCudaCollisionObject* objects, int numObjects, 
                                bool* overlapResults, int maxPairs)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= maxPairs) return;
    
    // Calculate pair indices (i, j) from linear index
    int i = 0, j = 0;
    int remaining = idx;
    while (remaining >= (numObjects - i - 1)) {
        remaining -= (numObjects - i - 1);
        i++;
    }
    j = i + 1 + remaining;
    
    if (i < numObjects && j < numObjects) {
        overlapResults[idx] = objects[i].testAabbAgainstAabb(objects[j]);
    }
}

/**
 * GPU kernel for collision object activation
 */
__global__ void activateCollisionObjects(btCudaCollisionObject* objects, int numObjects, 
                                        btCudaScalar timeStep)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numObjects) return;
    
    btCudaCollisionObject& obj = objects[idx];
    
    if (obj.getActivationState() == CUDA_WANTS_DEACTIVATION) {
        obj.m_deactivationTime += timeStep;
        if (obj.m_deactivationTime > btCudaScalar(2.0)) {
            obj.setActivationState(CUDA_ISLAND_SLEEPING);
        }
    }
}

#endif // BT_CUDA_COLLISION_OBJECT_CUH
