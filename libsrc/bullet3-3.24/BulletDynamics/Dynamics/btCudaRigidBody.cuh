/*
CUDA Conversion of Bullet Physics btRigidBody
Copyright (c) 2003-2006 Erwin Coumans  https://bulletphysics.org
CUDA Conversion: 2025

This software is provided 'as-is', without any express or implied warranty.
*/

#ifndef BT_CUDA_RIGID_BODY_CUH
#define BT_CUDA_RIGID_BODY_CUH

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "../../LinearMath/btCudaTransform.cuh"
#include "../../LinearMath/btCudaVector3.cuh"
#include "../../LinearMath/btCudaMatrix3x3.cuh"
#include "../../BulletCollision/CollisionDispatch/btCudaCollisionObject.cuh"

// Rigid body flags
#define CUDA_DISABLE_WORLD_GRAVITY 1
#define CUDA_ENABLE_GYROSCOPIC_FORCE_EXPLICIT 2
#define CUDA_ENABLE_GYROSCOPIC_FORCE_IMPLICIT_WORLD 4
#define CUDA_ENABLE_GYROSCOPIC_FORCE_IMPLICIT_BODY 8
#define CUDA_ENABLE_GYROSCOPIC_FORCE (CUDA_ENABLE_GYROSCOPIC_FORCE_IMPLICIT_BODY)

/**
 * CUDA Rigid Body Construction Info
 */
struct btCudaRigidBodyConstructionInfo
{
    btCudaScalar m_mass;
    btCudaTransform m_startWorldTransform;
    btCudaCollisionShape* m_collisionShape;
    btCudaVector3 m_localInertia;
    btCudaScalar m_linearDamping;
    btCudaScalar m_angularDamping;
    btCudaScalar m_friction;
    btCudaScalar m_rollingFriction;
    btCudaScalar m_spinningFriction;
    btCudaScalar m_restitution;
    btCudaScalar m_linearSleepingThreshold;
    btCudaScalar m_angularSleepingThreshold;
    bool m_additionalDamping;
    btCudaScalar m_additionalDampingFactor;
    btCudaScalar m_additionalLinearDampingThresholdSqr;
    btCudaScalar m_additionalAngularDampingThresholdSqr;
    btCudaScalar m_additionalAngularDampingFactor;
    
    __device__ __host__ void init(btCudaScalar mass, btCudaCollisionShape* collisionShape, const btCudaVector3& localInertia,
                                 const btCudaTransform& startTransform = btCudaTransform::getIdentity())
    {
        m_mass = mass;
        m_collisionShape = collisionShape;
        m_localInertia = localInertia;
        m_startWorldTransform = startTransform;
        m_linearDamping = btCudaScalar(0.0);
        m_angularDamping = btCudaScalar(0.0);
        m_friction = btCudaScalar(0.5);
        m_rollingFriction = btCudaScalar(0.0);
        m_spinningFriction = btCudaScalar(0.0);
        m_restitution = btCudaScalar(0.0);
        m_linearSleepingThreshold = btCudaScalar(0.8);
        m_angularSleepingThreshold = btCudaScalar(1.0);
        m_additionalDamping = false;
        m_additionalDampingFactor = btCudaScalar(0.005);
        m_additionalLinearDampingThresholdSqr = btCudaScalar(0.01);
        m_additionalAngularDampingThresholdSqr = btCudaScalar(0.01);
        m_additionalAngularDampingFactor = btCudaScalar(0.01);
    }
};

/**
 * CUDA Rigid Body
 * Complete GPU-accelerated rigid body dynamics
 */
struct btCudaRigidBody : public btCudaCollisionObject
{
    // Dynamic properties
    btCudaMatrix3x3 m_invInertiaTensorWorld;
    btCudaVector3 m_linearVelocity;
    btCudaVector3 m_angularVelocity;
    btCudaScalar m_inverseMass;
    btCudaVector3 m_linearFactor;
    btCudaVector3 m_angularFactor;
    
    // Accumulated forces and torques
    btCudaVector3 m_gravity;
    btCudaVector3 m_gravity_acceleration;
    btCudaVector3 m_invInertiaLocal;
    btCudaVector3 m_totalForce;
    btCudaVector3 m_totalTorque;
    
    // Damping
    btCudaScalar m_linearDamping;
    btCudaScalar m_angularDamping;
    bool m_additionalDamping;
    btCudaScalar m_additionalDampingFactor;
    btCudaScalar m_additionalLinearDampingThresholdSqr;
    btCudaScalar m_additionalAngularDampingThresholdSqr;
    btCudaScalar m_additionalAngularDampingFactor;
    
    // Sleeping thresholds
    btCudaScalar m_linearSleepingThreshold;
    btCudaScalar m_angularSleepingThreshold;
    
    // Flags
    int m_rigidbodyFlags;
    int m_debugBodyId;
    
    /**
     * Initialize rigid body from construction info
     */
    __device__ __host__ void init(const btCudaRigidBodyConstructionInfo& constructionInfo)
    {
        // Initialize base collision object
        btCudaCollisionObject::init();
        setCollisionShape(constructionInfo.m_collisionShape);
        setWorldTransform(constructionInfo.m_startWorldTransform);
        
        // Set dynamic properties
        m_inverseMass = constructionInfo.m_mass != btCudaScalar(0.0) ? btCudaScalar(1.0) / constructionInfo.m_mass : btCudaScalar(0.0);
        m_invInertiaLocal = constructionInfo.m_localInertia;
        
        // Set damping
        m_linearDamping = constructionInfo.m_linearDamping;
        m_angularDamping = constructionInfo.m_angularDamping;
        m_additionalDamping = constructionInfo.m_additionalDamping;
        m_additionalDampingFactor = constructionInfo.m_additionalDampingFactor;
        m_additionalLinearDampingThresholdSqr = constructionInfo.m_additionalLinearDampingThresholdSqr;
        m_additionalAngularDampingThresholdSqr = constructionInfo.m_additionalAngularDampingThresholdSqr;
        m_additionalAngularDampingFactor = constructionInfo.m_additionalAngularDampingFactor;
        
        // Set sleeping thresholds
        m_linearSleepingThreshold = constructionInfo.m_linearSleepingThreshold;
        m_angularSleepingThreshold = constructionInfo.m_angularSleepingThreshold;
        
        // Initialize other properties
        m_linearVelocity.setZero();
        m_angularVelocity.setZero();
        m_linearFactor.setValue(1, 1, 1);
        m_angularFactor.setValue(1, 1, 1);
        m_gravity.setZero();
        m_gravity_acceleration.setZero();
        m_totalForce.setZero();
        m_totalTorque.setZero();
        m_rigidbodyFlags = 0;
        m_debugBodyId = -1;
        
        setCollisionFlags(getCollisionFlags() | CUDA_RIGID_BODY);
        
        // Update inverse inertia tensor
        updateInverseInertiaTensor();
    }
    
    /**
     * Physics integration step - GPU optimized
     */
    __device__ void integrateVelocities(btCudaScalar step)
    {
        if (isStaticOrKinematicObject()) {
            return;
        }
        
        // Apply gravity
        m_linearVelocity += m_gravity_acceleration * step;
        
        // Apply accumulated forces
        m_linearVelocity += m_totalForce * (m_inverseMass * step);
        m_angularVelocity += m_invInertiaTensorWorld * m_totalTorque * step;
        
        // Apply damping
        applyDamping(step);
        
        // Clear accumulated forces
        clearForces();
    }
    
    __device__ void predictIntegratedTransform(btCudaScalar timeStep, btCudaTransform& predictedTransform) const
    {
        predictedTransform = getWorldTransform();
        predictedTransform.integrateTransform(m_linearVelocity, m_angularVelocity, timeStep);
    }
    
    __device__ void proceedToTransform(const btCudaTransform& newTrans)
    {
        setCenterOfMassTransform(newTrans);
    }
    
    /**
     * Force and torque application - GPU optimized
     */
    __device__ void applyCentralForce(const btCudaVector3& force)
    {
        m_totalForce += force * m_linearFactor;
    }
    
    __device__ void applyTorque(const btCudaVector3& torque)
    {
        m_totalTorque += torque * m_angularFactor;
    }
    
    __device__ void applyForce(const btCudaVector3& force, const btCudaVector3& relativePos)
    {
        applyCentralForce(force);
        applyTorque(relativePos.cross(force * m_linearFactor));
    }
    
    __device__ void applyCentralImpulse(const btCudaVector3& impulse)
    {
        m_linearVelocity += impulse * m_inverseMass * m_linearFactor;
    }
    
    __device__ void applyTorqueImpulse(const btCudaVector3& torque)
    {
        m_angularVelocity += m_invInertiaTensorWorld * torque * m_angularFactor;
    }
    
    __device__ void applyImpulse(const btCudaVector3& impulse, const btCudaVector3& relativePos)
    {
        if (m_inverseMass != btCudaScalar(0.0)) {
            applyCentralImpulse(impulse);
            if (m_angularFactor.length2() != btCudaScalar(0.0)) {
                applyTorqueImpulse(relativePos.cross(impulse * m_linearFactor));
            }
        }
    }
    
    __device__ void clearForces()
    {
        m_totalForce.setZero();
        m_totalTorque.setZero();
    }
    
    /**
     * Velocity getters/setters - GPU optimized
     */
    __device__ const btCudaVector3& getLinearVelocity() const { return m_linearVelocity; }
    __device__ const btCudaVector3& getAngularVelocity() const { return m_angularVelocity; }
    
    __device__ void setLinearVelocity(const btCudaVector3& linVel)
    {
        m_linearVelocity = linVel;
    }
    
    __device__ void setAngularVelocity(const btCudaVector3& angVel)
    {
        m_angularVelocity = angVel;
    }
    
    /**
     * Mass and inertia - GPU optimized
     */
    __device__ btCudaScalar getInverseMass() const { return m_inverseMass; }
    __device__ btCudaScalar getMass() const { return m_inverseMass == btCudaScalar(0.0) ? btCudaScalar(0.0) : btCudaScalar(1.0) / m_inverseMass; }
    
    __device__ void setMassProps(btCudaScalar mass, const btCudaVector3& inertia)
    {
        m_inverseMass = mass == btCudaScalar(0.0) ? btCudaScalar(0.0) : btCudaScalar(1.0) / mass;
        m_invInertiaLocal = inertia;
        updateInverseInertiaTensor();
    }
    
    __device__ const btCudaMatrix3x3& getInvInertiaTensorWorld() const { return m_invInertiaTensorWorld; }
    
    /**
     * Center of mass transform - GPU optimized
     */
    __device__ void setCenterOfMassTransform(const btCudaTransform& xform)
    {
        if (isKinematicObject()) {
            setInterpolationWorldTransform(getWorldTransform());
        } else {
            setInterpolationWorldTransform(xform);
        }
        setWorldTransform(xform);
        updateInverseInertiaTensor();
    }
    
    __device__ const btCudaTransform& getCenterOfMassTransform() const
    {
        return getWorldTransform();
    }
    
    /**
     * Damping application - GPU optimized
     */
    __device__ void applyDamping(btCudaScalar timeStep)
    {
        // Linear damping
        m_linearVelocity *= btCudaPow(btCudaScalar(1.0) - m_linearDamping, timeStep);
        
        // Angular damping
        m_angularVelocity *= btCudaPow(btCudaScalar(1.0) - m_angularDamping, timeStep);
        
        if (m_additionalDamping) {
            // Additional damping for stability
            if (m_linearVelocity.length2() < m_additionalLinearDampingThresholdSqr) {
                m_linearVelocity *= btCudaScalar(1.0) - m_additionalDampingFactor;
            }
            
            if (m_angularVelocity.length2() < m_additionalAngularDampingThresholdSqr) {
                m_angularVelocity *= btCudaScalar(1.0) - m_additionalAngularDampingFactor;
            }
        }
    }
    
    /**
     * Object state checks - GPU optimized
     */
    __device__ bool isStaticObject() const
    {
        return (getCollisionFlags() & CUDA_RIGID_BODY) && (m_inverseMass == btCudaScalar(0.0));
    }
    
    __device__ bool isKinematicObject() const
    {
        return (getCollisionFlags() & CUDA_RIGID_BODY) && (m_inverseMass == btCudaScalar(0.0));
    }
    
    __device__ bool isStaticOrKinematicObject() const
    {
        return isStaticObject() || isKinematicObject();
    }
    
    __device__ bool hasContactResponse() const
    {
        return !isStaticOrKinematicObject();
    }
    
    /**
     * Gravity - GPU optimized
     */
    __device__ void setGravity(const btCudaVector3& acceleration)
    {
        if (m_inverseMass != btCudaScalar(0.0)) {
            m_gravity = acceleration * (btCudaScalar(1.0) / m_inverseMass);
        }
        m_gravity_acceleration = acceleration;
    }
    
    __device__ const btCudaVector3& getGravity() const { return m_gravity_acceleration; }
    
    /**
     * Factors for constraining motion - GPU optimized
     */
    __device__ void setLinearFactor(const btCudaVector3& linearFactor) { m_linearFactor = linearFactor; }
    __device__ const btCudaVector3& getLinearFactor() const { return m_linearFactor; }
    
    __device__ void setAngularFactor(const btCudaVector3& angularFactor) { m_angularFactor = angularFactor; }
    __device__ void setAngularFactor(btCudaScalar angularFactor) { m_angularFactor.setValue(angularFactor, angularFactor, angularFactor); }
    __device__ const btCudaVector3& getAngularFactor() const { return m_angularFactor; }
    
    /**
     * Sleeping - GPU optimized
     */
    __device__ void setSleepingThresholds(btCudaScalar linear, btCudaScalar angular)
    {
        m_linearSleepingThreshold = linear;
        m_angularSleepingThreshold = angular;
    }
    
    __device__ bool wantsSleeping()
    {
        if (getActivationState() == CUDA_DISABLE_DEACTIVATION) {
            return false;
        }
        
        if ((m_linearVelocity.length2() < m_linearSleepingThreshold * m_linearSleepingThreshold) &&
            (m_angularVelocity.length2() < m_angularSleepingThreshold * m_angularSleepingThreshold)) {
            return true;
        }
        return false;
    }
    
    /**
     * Update inverse inertia tensor in world space - GPU optimized
     */
    __device__ void updateInverseInertiaTensor()
    {
        btCudaMatrix3x3 m;
        m.setValue(m_invInertiaLocal.getX(), 0, 0,
                  0, m_invInertiaLocal.getY(), 0,
                  0, 0, m_invInertiaLocal.getZ());
        
        m_invInertiaTensorWorld = getWorldTransform().getBasis() * m * getWorldTransform().getBasis().transpose();
    }
    
    /**
     * Get velocity at a point - GPU optimized
     */
    __device__ btCudaVector3 getVelocityInLocalPoint(const btCudaVector3& relativePos) const
    {
        return m_linearVelocity + m_angularVelocity.cross(relativePos);
    }
    
    __device__ btCudaVector3 getPushVelocity() const { return m_linearVelocity; }
    __device__ btCudaVector3 getTurnVelocity() const { return m_angularVelocity; }
    
    /**
     * Compute impulse denominator for contact solving - GPU optimized
     */
    __device__ btCudaScalar computeImpulseDenominator(const btCudaVector3& pos, const btCudaVector3& normal) const
    {
        btCudaVector3 r0 = pos - getCenterOfMassTransform().getOrigin();
        btCudaVector3 c0 = r0.cross(normal);
        btCudaVector3 vec = (c0 * getInvInertiaTensorWorld()).cross(r0);
        return m_inverseMass + normal.dot(vec);
    }
    
    /**
     * Compute angular impulse denominator - GPU optimized
     */
    __device__ btCudaScalar computeAngularImpulseDenominator(const btCudaVector3& axis) const
    {
        btCudaVector3 vec = axis * getInvInertiaTensorWorld();
        return axis.dot(vec);
    }
};

/**
 * GPU kernels for rigid body operations
 */

/**
 * Integrate velocities for all rigid bodies
 */
__global__ void integrateRigidBodyVelocities(btCudaRigidBody* rigidBodies, int numBodies, btCudaScalar timeStep)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numBodies) return;
    
    rigidBodies[idx].integrateVelocities(timeStep);
}

/**
 * Predict integrated transforms for all rigid bodies
 */
__global__ void predictRigidBodyTransforms(btCudaRigidBody* rigidBodies, btCudaTransform* predictedTransforms,
                                          int numBodies, btCudaScalar timeStep)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numBodies) return;
    
    rigidBodies[idx].predictIntegratedTransform(timeStep, predictedTransforms[idx]);
}

/**
 * Apply gravity to all rigid bodies
 */
__global__ void applyGravityToRigidBodies(btCudaRigidBody* rigidBodies, int numBodies, 
                                         const btCudaVector3 gravity, btCudaScalar timeStep)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numBodies) return;
    
    if (!rigidBodies[idx].isStaticOrKinematicObject()) {
        rigidBodies[idx].setGravity(gravity);
    }
}

/**
 * Clear forces for all rigid bodies
 */
__global__ void clearRigidBodyForces(btCudaRigidBody* rigidBodies, int numBodies)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numBodies) return;
    
    rigidBodies[idx].clearForces();
}

/**
 * Update rigid body transforms
 */
__global__ void updateRigidBodyTransforms(btCudaRigidBody* rigidBodies, const btCudaTransform* newTransforms, int numBodies)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numBodies) return;
    
    rigidBodies[idx].proceedToTransform(newTransforms[idx]);
}

#endif // BT_CUDA_RIGID_BODY_CUH
