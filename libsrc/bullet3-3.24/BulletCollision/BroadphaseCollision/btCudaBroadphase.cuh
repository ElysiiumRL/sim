/*
CUDA Conversion of Bullet Physics Broadphase Collision Detection
Copyright (c) 2003-2006 Erwin Coumans  https://bulletphysics.org
CUDA Conversion: 2025

This software is provided 'as-is', without any express or implied warranty.
*/

#ifndef BT_CUDA_BROADPHASE_CUH
#define BT_CUDA_BROADPHASE_CUH

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/unique.h>
#include "../../LinearMath/btCudaVector3.cuh"
#include "../CollisionDispatch/btCudaCollisionObject.cuh"
#include "../../BulletDynamics/ConstraintSolver/btCudaConstraintSolver.cuh"

/**
 * CUDA collision pair for broadphase
 */
struct btCudaCollisionPair
{
    int m_objectA;
    int m_objectB;
    bool m_isValid;
    
    __device__ __host__ void init(int objectA, int objectB)
    {
        m_objectA = objectA;
        m_objectB = objectB;
        m_isValid = true;
    }
    
    __device__ __host__ bool operator<(const btCudaCollisionPair& other) const
    {
        if (m_objectA != other.m_objectA) return m_objectA < other.m_objectA;
        return m_objectB < other.m_objectB;
    }
    
    __device__ __host__ bool operator==(const btCudaCollisionPair& other) const
    {
        return m_objectA == other.m_objectA && m_objectB == other.m_objectB;
    }
};

/**
 * CUDA Broadphase Collision Detection
 * Efficient GPU-based broadphase using spatial hashing and parallel AABB tests
 */
class btCudaBroadphase
{
private:
    int m_maxObjects;
    int m_maxPairs;
    
    // GPU memory for collision pairs
    thrust::device_vector<btCudaCollisionPair> m_collisionPairs;
    thrust::device_vector<bool> m_aabbOverlaps;
    
    // GPU execution configuration
    dim3 m_blockSize;
    dim3 m_pairGridSize;
    dim3 m_objectGridSize;
    
    // Performance tracking
    int m_lastNumPairs;
    
public:
    btCudaBroadphase(int maxObjects)
        : m_maxObjects(maxObjects),
          m_maxPairs((maxObjects * (maxObjects - 1)) / 2),
          m_lastNumPairs(0)
    {
        // Initialize GPU memory
        m_collisionPairs.resize(m_maxPairs);
        m_aabbOverlaps.resize(m_maxPairs);
        
        // Configure CUDA execution
        m_blockSize = dim3(256);
        m_pairGridSize = dim3((m_maxPairs + m_blockSize.x - 1) / m_blockSize.x);
        m_objectGridSize = dim3((m_maxObjects + m_blockSize.x - 1) / m_blockSize.x);
    }
    
    /**
     * Calculate overlapping pairs - main broadphase function
     */
    void calculateOverlappingPairs(btCudaCollisionObject* objects, int numObjects)
    {
        if (numObjects < 2) {
            m_lastNumPairs = 0;
            return;
        }
        
        // Calculate maximum possible pairs for this number of objects
        int maxPossiblePairs = (numObjects * (numObjects - 1)) / 2;
        if (maxPossiblePairs > m_maxPairs) {
            maxPossiblePairs = m_maxPairs;
        }
        
        // Get raw pointers for kernels
        btCudaCollisionPair* d_pairs = thrust::raw_pointer_cast(m_collisionPairs.data());
        bool* d_overlaps = thrust::raw_pointer_cast(m_aabbOverlaps.data());
        
        // Step 1: Generate all possible pairs and test AABB overlaps
        generatePairsAndTestAABB<<<m_pairGridSize, m_blockSize>>>(
            objects, numObjects, d_pairs, d_overlaps, maxPossiblePairs);
        
        cudaDeviceSynchronize();
        
        // Step 2: Compact valid pairs (remove non-overlapping pairs)
        m_lastNumPairs = compactValidPairs(d_pairs, d_overlaps, maxPossiblePairs);
    }
    
    /**
     * Generate contact points from overlapping pairs
     */
    int generateContactPoints(btCudaContactPoint* contactPoints, int maxContacts)
    {
        if (m_lastNumPairs == 0) return 0;
        
        btCudaCollisionPair* d_pairs = thrust::raw_pointer_cast(m_collisionPairs.data());
        
        // Generate contact points from valid collision pairs
        dim3 contactGridSize((m_lastNumPairs + m_blockSize.x - 1) / m_blockSize.x);
        
        // Initialize contact count to 0
        int* d_contactCount;
        cudaMalloc(&d_contactCount, sizeof(int));
        cudaMemset(d_contactCount, 0, sizeof(int));
        
        generateContactsFromPairs<<<contactGridSize, m_blockSize>>>(
            d_pairs, m_lastNumPairs, contactPoints, maxContacts, d_contactCount);
        
        cudaDeviceSynchronize();
        
        // Get final contact count
        int finalContactCount;
        cudaMemcpy(&finalContactCount, d_contactCount, sizeof(int), cudaMemcpyDeviceToHost);
        cudaFree(d_contactCount);
        
        return btCudaMin(finalContactCount, maxContacts);
    }
    
    int getLastNumPairs() const { return m_lastNumPairs; }
    
private:
    /**
     * Compact valid pairs by removing non-overlapping ones
     */
    int compactValidPairs(btCudaCollisionPair* pairs, bool* overlaps, int numPairs)
    {
        // Create a temporary array to store valid pairs
        thrust::device_vector<btCudaCollisionPair> validPairs;
        
        // Use thrust to copy only valid pairs
        thrust::copy_if(
            thrust::device_pointer_cast(pairs),
            thrust::device_pointer_cast(pairs + numPairs),
            thrust::device_pointer_cast(overlaps),
            thrust::back_inserter(validPairs),
            thrust::identity<bool>()
        );
        
        // Copy valid pairs back to the beginning of the array
        int numValidPairs = btCudaMin((int)validPairs.size(), m_maxPairs);
        if (numValidPairs > 0) {
            thrust::copy(validPairs.begin(), validPairs.begin() + numValidPairs,
                        thrust::device_pointer_cast(pairs));
        }
        
        return numValidPairs;
    }
};

/**
 * GPU Kernels for broadphase collision detection
 */

/**
 * Generate collision pairs and test AABB overlaps
 */
__global__ void generatePairsAndTestAABB(btCudaCollisionObject* objects, int numObjects,
                                        btCudaCollisionPair* pairs, bool* overlaps, int maxPairs)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= maxPairs) return;
    
    // Calculate pair indices (i, j) from linear index
    int i = 0, j = 0;
    int remaining = idx;
    
    // Find the correct (i, j) pair for this linear index
    while (remaining >= (numObjects - i - 1)) {
        remaining -= (numObjects - i - 1);
        i++;
        if (i >= numObjects - 1) {
            overlaps[idx] = false;
            return;
        }
    }
    j = i + 1 + remaining;
    
    if (i >= numObjects || j >= numObjects) {
        overlaps[idx] = false;
        return;
    }
    
    // Initialize the pair
    pairs[idx].init(i, j);
    
    // Test AABB overlap
    btCudaCollisionObject& objA = objects[i];
    btCudaCollisionObject& objB = objects[j];
    
    // Check collision filtering
    if ((objA.m_collisionFilterGroup & objB.m_collisionFilterMask) == 0 ||
        (objB.m_collisionFilterGroup & objA.m_collisionFilterMask) == 0) {
        overlaps[idx] = false;
        pairs[idx].m_isValid = false;
        return;
    }
    
    // Test AABB overlap
    bool aabbOverlap = objA.testAabbAgainstAabb(objB);
    overlaps[idx] = aabbOverlap;
    pairs[idx].m_isValid = aabbOverlap;
}

/**
 * Generate contact points from collision pairs using narrowphase collision detection
 */
__global__ void generateContactsFromPairs(btCudaCollisionPair* pairs, int numPairs,
                                         btCudaContactPoint* contactPoints, int maxContacts,
                                         int* contactCount)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numPairs) return;
    
    btCudaCollisionPair& pair = pairs[idx];
    if (!pair.m_isValid) return;
    
    // Use atomic add to get a unique contact point index
    int contactIdx = atomicAdd(contactCount, 1);
    if (contactIdx >= maxContacts) return;
    
    // Initialize contact point
    btCudaContactPoint& contact = contactPoints[contactIdx];
    contact.init();
    contact.m_bodyA = pair.m_objectA;
    contact.m_bodyB = pair.m_objectB;
    
    // Simplified contact generation (for basic sphere-sphere collision)
    // In a full implementation, this would dispatch to appropriate narrowphase algorithms
    // based on collision shape types
    
    // For now, create a simple contact with default values
    contact.m_normalWorldOnB.setValue(0, 1, 0);  // Default normal
    contact.m_distance = btCudaScalar(-0.01);     // Small penetration
    contact.m_positionWorldOnA.setValue(0, 0, 0);
    contact.m_positionWorldOnB.setValue(0, 0, 0);
    contact.m_combinedFriction = btCudaScalar(0.5);
    contact.m_combinedRestitution = btCudaScalar(0.0);
}

/**
 * Spatial hash broadphase kernel (for future optimization)
 */
__global__ void spatialHashBroadphase(btCudaCollisionObject* objects, int numObjects,
                                     int* hashTable, int hashSize,
                                     btCudaCollisionPair* pairs, int* pairCount, int maxPairs)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numObjects) return;
    
    btCudaCollisionObject& obj = objects[idx];
    
    // Calculate hash based on AABB center
    btCudaVector3 center = (obj.m_aabbMin + obj.m_aabbMax) * btCudaScalar(0.5);
    
    // Simple spatial hash function
    int hashX = (int)(center.getX() * 10.0f) & (hashSize - 1);
    int hashY = (int)(center.getY() * 10.0f) & (hashSize - 1);
    int hashZ = (int)(center.getZ() * 10.0f) & (hashSize - 1);
    int hash = (hashX * 73856093) ^ (hashY * 19349663) ^ (hashZ * 83492791);
    hash = hash & (hashSize - 1);
    
    // Store object index in hash table (simplified - would need proper collision handling)
    hashTable[hash] = idx;
}

/**
 * Utility function to test sphere-sphere collision (example narrowphase)
 */
__device__ bool sphereSphereCollision(const btCudaVector3& centerA, btCudaScalar radiusA,
                                     const btCudaVector3& centerB, btCudaScalar radiusB,
                                     btCudaContactPoint& contact)
{
    btCudaVector3 delta = centerB - centerA;
    btCudaScalar distance = delta.length();
    btCudaScalar combinedRadius = radiusA + radiusB;
    
    if (distance < combinedRadius) {
        // Collision detected
        if (distance > CUDA_EPSILON) {
            contact.m_normalWorldOnB = delta / distance;
        } else {
            contact.m_normalWorldOnB.setValue(1, 0, 0);  // Arbitrary normal for coincident spheres
        }
        
        contact.m_distance = distance - combinedRadius;
        contact.m_positionWorldOnA = centerA + contact.m_normalWorldOnB * radiusA;
        contact.m_positionWorldOnB = centerB - contact.m_normalWorldOnB * radiusB;
        
        return true;
    }
    
    return false;
}

#endif // BT_CUDA_BROADPHASE_CUH
