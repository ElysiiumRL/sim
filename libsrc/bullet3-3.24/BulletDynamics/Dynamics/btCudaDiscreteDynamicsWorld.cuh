/*
CUDA Conversion of Bullet Physics btDiscreteDynamicsWorld
Copyright (c) 2003-2006 Erwin Coumans  https://bulletphysics.org
CUDA Conversion: 2025

This software is provided 'as-is', without any express or implied warranty.
*/

#ifndef BT_CUDA_DISCRETE_DYNAMICS_WORLD_CUH
#define BT_CUDA_DISCRETE_DYNAMICS_WORLD_CUH

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include "btCudaRigidBody.cuh"
#include "../../BulletCollision/CollisionDispatch/btCudaCollisionObject.cuh"
#include "../../LinearMath/btCudaVector3.cuh"
#include "../ConstraintSolver/btCudaConstraintSolver.cuh"
#include "../../BulletCollision/BroadphaseCollision/btCudaBroadphase.cuh"

// Forward declarations
struct btCudaContactPoint;
struct btCudaConstraint;

/**
 * CUDA Discrete Dynamics World
 * Main physics simulation class running entirely on GPU
 */
class btCudaDiscreteDynamicsWorld
{
private:
    // GPU memory arrays
    thrust::device_vector<btCudaRigidBody> m_rigidBodies;
    thrust::device_vector<btCudaCollisionObject> m_collisionObjects;
    thrust::device_vector<btCudaContactPoint> m_contactPoints;
    thrust::device_vector<btCudaConstraint> m_constraints;
    thrust::device_vector<btCudaTransform> m_predictedTransforms;
    
    // GPU pointers for kernels
    btCudaRigidBody* d_rigidBodies;
    btCudaCollisionObject* d_collisionObjects;
    btCudaContactPoint* d_contactPoints;
    btCudaConstraint* d_constraints;
    btCudaTransform* d_predictedTransforms;
    
    // World properties
    btCudaVector3 m_gravity;
    btCudaScalar m_fixedTimeStep;
    int m_maxSubSteps;
    
    // Simulation state
    int m_numRigidBodies;
    int m_numCollisionObjects;
    int m_numContactPoints;
    int m_numConstraints;
    int m_maxRigidBodies;
    int m_maxCollisionObjects;
    int m_maxContactPoints;
    int m_maxConstraints;
    
    // Components
    btCudaConstraintSolver* m_constraintSolver;
    btCudaBroadphase* m_broadphase;
    
    // CUDA execution configuration
    dim3 m_blockSize;
    dim3 m_rigidBodyGridSize;
    dim3 m_collisionGridSize;
    dim3 m_contactGridSize;
    
    // Performance monitoring
    cudaEvent_t m_startEvent, m_stopEvent;
    float m_lastStepTime;
    
public:
    /**
     * Constructor
     */
    btCudaDiscreteDynamicsWorld(int maxRigidBodies = 1024, int maxCollisionObjects = 2048, 
                               int maxContactPoints = 4096, int maxConstraints = 2048)
        : m_maxRigidBodies(maxRigidBodies),
          m_maxCollisionObjects(maxCollisionObjects),
          m_maxContactPoints(maxContactPoints),
          m_maxConstraints(maxConstraints),
          m_numRigidBodies(0),
          m_numCollisionObjects(0),
          m_numContactPoints(0),
          m_numConstraints(0),
          m_fixedTimeStep(1.0f / 60.0f),
          m_maxSubSteps(1),
          m_constraintSolver(nullptr),
          m_broadphase(nullptr),
          m_lastStepTime(0.0f)
    {
        m_gravity.setValue(0, -9.81f, 0);
        
        // Initialize CUDA execution configuration
        m_blockSize = dim3(256);
        m_rigidBodyGridSize = dim3((m_maxRigidBodies + m_blockSize.x - 1) / m_blockSize.x);
        m_collisionGridSize = dim3((m_maxCollisionObjects + m_blockSize.x - 1) / m_blockSize.x);
        m_contactGridSize = dim3((m_maxContactPoints + m_blockSize.x - 1) / m_blockSize.x);
        
        initializeGPUMemory();
        initializeComponents();
        
        // Create CUDA events for timing
        cudaEventCreate(&m_startEvent);
        cudaEventCreate(&m_stopEvent);
    }
    
    /**
     * Destructor
     */
    ~btCudaDiscreteDynamicsWorld()
    {
        cleanup();
    }
    
    /**
     * Initialize GPU memory and components
     */
    void initializeGPUMemory()
    {
        try {
            // Allocate GPU memory
            m_rigidBodies.resize(m_maxRigidBodies);
            m_collisionObjects.resize(m_maxCollisionObjects);
            m_contactPoints.resize(m_maxContactPoints);
            m_constraints.resize(m_maxConstraints);
            m_predictedTransforms.resize(m_maxRigidBodies);
            
            // Get raw pointers for kernels
            d_rigidBodies = thrust::raw_pointer_cast(m_rigidBodies.data());
            d_collisionObjects = thrust::raw_pointer_cast(m_collisionObjects.data());
            d_contactPoints = thrust::raw_pointer_cast(m_contactPoints.data());
            d_constraints = thrust::raw_pointer_cast(m_constraints.data());
            d_predictedTransforms = thrust::raw_pointer_cast(m_predictedTransforms.data());
            
        } catch (const std::exception& e) {
            printf("CUDA memory allocation failed: %s\n", e.what());
            throw;
        }
    }
    
    void initializeComponents()
    {
        // Initialize constraint solver
        m_constraintSolver = new btCudaConstraintSolver(m_maxConstraints, m_maxContactPoints);
        
        // Initialize broadphase collision detection
        m_broadphase = new btCudaBroadphase(m_maxCollisionObjects);
    }
    
    /**
     * Main physics simulation step - COMPLETELY ON GPU
     */
    int stepSimulation(btCudaScalar timeStep, int maxSubSteps = 1, btCudaScalar fixedTimeStep = btCudaScalar(1.) / btCudaScalar(60.))
    {
        cudaEventRecord(m_startEvent);
        
        m_fixedTimeStep = fixedTimeStep;
        m_maxSubSteps = maxSubSteps;
        
        // Calculate number of substeps
        int numSimulationSubSteps = 0;
        
        if (maxSubSteps) {
            // Fixed timestep with substeps
            btCudaScalar localTime = timeStep;
            if (localTime >= fixedTimeStep) {
                numSimulationSubSteps = int(localTime / fixedTimeStep);
                numSimulationSubSteps = btCudaMin(numSimulationSubSteps, maxSubSteps);
            }
        } else {
            // Variable timestep
            fixedTimeStep = timeStep;
            localTime = timeStep;
            if (localTime > btCudaScalar(0.)) {
                numSimulationSubSteps = 1;
                maxSubSteps = 1;
            }
        }
        
        // Perform simulation substeps
        for (int i = 0; i < numSimulationSubSteps; i++) {
            internalSingleStepSimulation(fixedTimeStep);
        }
        
        cudaEventRecord(m_stopEvent);
        cudaEventSynchronize(m_stopEvent);
        cudaEventElapsedTime(&m_lastStepTime, m_startEvent, m_stopEvent);
        
        return numSimulationSubSteps;
    }
    
    /**
     * Internal single step simulation - ALL GPU KERNELS
     */
    void internalSingleStepSimulation(btCudaScalar timeStep)
    {
        // 1. Clear forces and update transforms
        clearForces();
        
        // 2. Apply gravity
        applyGravity(timeStep);
        
        // 3. Integrate velocities
        integrateVelocities(timeStep);
        
        // 4. Predict transforms
        predictTransforms(timeStep);
        
        // 5. Broadphase collision detection
        performBroadphaseCollision();
        
        // 6. Narrowphase collision detection
        performNarrowphaseCollision();
        
        // 7. Solve constraints and contacts
        solveConstraints(timeStep);
        
        // 8. Update transforms
        updateTransforms();
        
        // 9. Process callbacks and cleanup
        processCallbacks();
        
        // Synchronize to ensure all GPU operations complete
        cudaDeviceSynchronize();
    }
    
    /**
     * GPU Simulation Steps
     */
    void clearForces()
    {
        if (m_numRigidBodies > 0) {
            clearRigidBodyForces<<<m_rigidBodyGridSize, m_blockSize>>>(d_rigidBodies, m_numRigidBodies);
        }
    }
    
    void applyGravity(btCudaScalar timeStep)
    {
        if (m_numRigidBodies > 0) {
            applyGravityToRigidBodies<<<m_rigidBodyGridSize, m_blockSize>>>(d_rigidBodies, m_numRigidBodies, m_gravity, timeStep);
        }
    }
    
    void integrateVelocities(btCudaScalar timeStep)
    {
        if (m_numRigidBodies > 0) {
            integrateRigidBodyVelocities<<<m_rigidBodyGridSize, m_blockSize>>>(d_rigidBodies, m_numRigidBodies, timeStep);
        }
    }
    
    void predictTransforms(btCudaScalar timeStep)
    {
        if (m_numRigidBodies > 0) {
            predictRigidBodyTransforms<<<m_rigidBodyGridSize, m_blockSize>>>(d_rigidBodies, d_predictedTransforms, m_numRigidBodies, timeStep);
        }
    }
    
    void performBroadphaseCollision()
    {
        if (m_broadphase && m_numCollisionObjects > 0) {
            m_broadphase->calculateOverlappingPairs(d_collisionObjects, m_numCollisionObjects);
        }
    }
    
    void performNarrowphaseCollision()
    {
        if (m_broadphase) {
            m_numContactPoints = m_broadphase->generateContactPoints(d_contactPoints, m_maxContactPoints);
        }
    }
    
    void solveConstraints(btCudaScalar timeStep)
    {
        if (m_constraintSolver && (m_numContactPoints > 0 || m_numConstraints > 0)) {
            m_constraintSolver->solveGroup(d_rigidBodies, m_numRigidBodies,
                                          d_contactPoints, m_numContactPoints,
                                          d_constraints, m_numConstraints,
                                          timeStep);
        }
    }
    
    void updateTransforms()
    {
        if (m_numRigidBodies > 0) {
            updateRigidBodyTransforms<<<m_rigidBodyGridSize, m_blockSize>>>(d_rigidBodies, d_predictedTransforms, m_numRigidBodies);
        }
    }
    
    void processCallbacks()
    {
        // Update AABBs
        if (m_numCollisionObjects > 0) {
            updateCollisionObjectAabbs<<<m_collisionGridSize, m_blockSize>>>(d_collisionObjects, m_numCollisionObjects);
        }
        
        // Process sleeping/activation
        if (m_numRigidBodies > 0) {
            activateCollisionObjects<<<m_rigidBodyGridSize, m_blockSize>>>(
                reinterpret_cast<btCudaCollisionObject*>(d_rigidBodies), m_numRigidBodies, m_fixedTimeStep);
        }
    }
    
    /**
     * Object management
     */
    int addRigidBody(const btCudaRigidBodyConstructionInfo& constructionInfo)
    {
        if (m_numRigidBodies >= m_maxRigidBodies) {
            printf("Error: Maximum rigid bodies reached\n");
            return -1;
        }
        
        // Initialize rigid body on host
        btCudaRigidBody hostBody;
        hostBody.init(constructionInfo);
        
        // Copy to GPU
        m_rigidBodies[m_numRigidBodies] = hostBody;
        
        // Also add to collision objects
        addCollisionObject(reinterpret_cast<const btCudaCollisionObject&>(hostBody));
        
        return m_numRigidBodies++;
    }
    
    int addCollisionObject(const btCudaCollisionObject& collisionObject)
    {
        if (m_numCollisionObjects >= m_maxCollisionObjects) {
            printf("Error: Maximum collision objects reached\n");
            return -1;
        }
        
        m_collisionObjects[m_numCollisionObjects] = collisionObject;
        return m_numCollisionObjects++;
    }
    
    void removeRigidBody(int index)
    {
        if (index >= 0 && index < m_numRigidBodies) {
            // Shift remaining bodies
            for (int i = index; i < m_numRigidBodies - 1; i++) {
                m_rigidBodies[i] = m_rigidBodies[i + 1];
            }
            m_numRigidBodies--;
        }
    }
    
    /**
     * Property setters/getters
     */
    void setGravity(const btCudaVector3& gravity) { m_gravity = gravity; }
    const btCudaVector3& getGravity() const { return m_gravity; }
    
    int getNumRigidBodies() const { return m_numRigidBodies; }
    int getNumCollisionObjects() const { return m_numCollisionObjects; }
    int getNumContactPoints() const { return m_numContactPoints; }
    
    float getLastStepTime() const { return m_lastStepTime; }
    
    /**
     * Get rigid body data for host access
     */
    void getRigidBodyData(thrust::host_vector<btCudaRigidBody>& hostBodies)
    {
        hostBodies.resize(m_numRigidBodies);
        thrust::copy(m_rigidBodies.begin(), m_rigidBodies.begin() + m_numRigidBodies, hostBodies.begin());
    }
    
    void setRigidBodyData(const thrust::host_vector<btCudaRigidBody>& hostBodies)
    {
        m_numRigidBodies = btCudaMin((int)hostBodies.size(), m_maxRigidBodies);
        thrust::copy(hostBodies.begin(), hostBodies.begin() + m_numRigidBodies, m_rigidBodies.begin());
    }
    
    /**
     * Get contact points for analysis
     */
    void getContactPoints(thrust::host_vector<btCudaContactPoint>& hostContacts)
    {
        hostContacts.resize(m_numContactPoints);
        thrust::copy(m_contactPoints.begin(), m_contactPoints.begin() + m_numContactPoints, hostContacts.begin());
    }
    
    /**
     * Performance and debugging
     */
    void printPerformanceInfo()
    {
        printf("CUDA Physics World Performance:\n");
        printf("  Last step time: %.3f ms\n", m_lastStepTime);
        printf("  Rigid bodies: %d/%d\n", m_numRigidBodies, m_maxRigidBodies);
        printf("  Collision objects: %d/%d\n", m_numCollisionObjects, m_maxCollisionObjects);
        printf("  Contact points: %d/%d\n", m_numContactPoints, m_maxContactPoints);
        printf("  SPS (Steps Per Second): %.1f\n", 1000.0f / m_lastStepTime);
    }
    
    /**
     * Cleanup
     */
    void cleanup()
    {
        if (m_constraintSolver) {
            delete m_constraintSolver;
            m_constraintSolver = nullptr;
        }
        
        if (m_broadphase) {
            delete m_broadphase;
            m_broadphase = nullptr;
        }
        
        cudaEventDestroy(m_startEvent);
        cudaEventDestroy(m_stopEvent);
    }
    
    /**
     * Synchronization utility
     */
    void synchronize()
    {
        cudaDeviceSynchronize();
    }
    
    /**
     * Memory usage information
     */
    size_t getGPUMemoryUsage() const
    {
        return (m_rigidBodies.size() * sizeof(btCudaRigidBody)) +
               (m_collisionObjects.size() * sizeof(btCudaCollisionObject)) +
               (m_contactPoints.size() * sizeof(btCudaContactPoint)) +
               (m_constraints.size() * sizeof(btCudaConstraint)) +
               (m_predictedTransforms.size() * sizeof(btCudaTransform));
    }
};

#endif // BT_CUDA_DISCRETE_DYNAMICS_WORLD_CUH
