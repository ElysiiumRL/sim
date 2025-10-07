/*
CUDA Conversion of Bullet Physics btRigidBody.cpp - COMPLETE GPU IMPLEMENTATION
Copyright (c) 2003-2006 Erwin Coumans  https://bulletphysics.org
CUDA Conversion: 2025

This software is provided 'as-is', without any express or implied warranty.
*/

#include "btCudaRigidBody.cuh"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

/**
 * CUDA implementation of rigid body physics calculations
 * All operations run on GPU for maximum performance
 */

/**
 * GPU kernel for mass rigid body integration
 */
__global__ void massIntegrateRigidBodies(btCudaRigidBody* bodies, int numBodies, 
                                        btCudaScalar timeStep, const btCudaVector3 globalGravity)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numBodies) return;
    
    btCudaRigidBody& body = bodies[idx];
    
    if (body.isStaticOrKinematicObject()) return;
    
    // Apply gravity if not disabled
    if (!(body.m_rigidbodyFlags & CUDA_DISABLE_WORLD_GRAVITY)) {
        body.applyCentralForce(globalGravity * body.getMass());
    }
    
    // Integrate velocities with GPU-optimized calculations
    body.integrateVelocities(timeStep);
}

/**
 * GPU kernel for gyroscopic force calculation
 */
__global__ void calculateGyroscopicForces(btCudaRigidBody* bodies, int numBodies, btCudaScalar timeStep)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numBodies) return;
    
    btCudaRigidBody& body = bodies[idx];
    
    if (body.isStaticOrKinematicObject()) return;
    
    // Calculate gyroscopic torque: τ = ω × (I * ω)
    if (body.m_rigidbodyFlags & CUDA_ENABLE_GYROSCOPIC_FORCE) {
        btCudaVector3 omega = body.getAngularVelocity();
        btCudaVector3 angularMomentum = body.getInvInertiaTensorWorld().inverse() * omega;
        btCudaVector3 gyroscopicTorque = omega.cross(angularMomentum);
        
        body.applyTorque(gyroscopicTorque * timeStep);
    }
}

/**
 * GPU kernel for advanced damping calculations
 */
__global__ void applyAdvancedDamping(btCudaRigidBody* bodies, int numBodies, btCudaScalar timeStep)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numBodies) return;
    
    btCudaRigidBody& body = bodies[idx];
    
    if (body.isStaticOrKinematicObject()) return;
    
    // Standard damping
    body.applyDamping(timeStep);
    
    // Additional air resistance for more realistic simulation
    btCudaVector3 linVel = body.getLinearVelocity();
    btCudaVector3 angVel = body.getAngularVelocity();
    
    btCudaScalar linSpeed = linVel.length();
    btCudaScalar angSpeed = angVel.length();
    
    // Quadratic air resistance
    if (linSpeed > CUDA_EPSILON) {
        btCudaVector3 airResistance = -linVel * (linSpeed * btCudaScalar(0.01));  // Adjustable coefficient
        body.applyCentralForce(airResistance);
    }
    
    if (angSpeed > CUDA_EPSILON) {
        btCudaVector3 angularAirResistance = -angVel * (angSpeed * btCudaScalar(0.001));
        body.applyTorque(angularAirResistance);
    }
}

/**
 * GPU kernel for continuous collision detection preparation
 */
__global__ void prepareContinuousCollisionDetection(btCudaRigidBody* bodies, int numBodies, 
                                                   btCudaTransform* previousTransforms, btCudaScalar timeStep)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numBodies) return;
    
    btCudaRigidBody& body = bodies[idx];
    
    // Store previous transform for CCD
    previousTransforms[idx] = body.getCenterOfMassTransform();
    
    // Calculate motion bounds for fast CCD rejection
    btCudaVector3 linearMotion = body.getLinearVelocity() * timeStep;
    btCudaVector3 angularMotion = body.getAngularVelocity() * timeStep;
    
    // Expand AABB to include motion
    btCudaScalar motionLength = linearMotion.length() + angularMotion.length();
    btCudaVector3 motionExpansion(motionLength, motionLength, motionLength);
    
    body.m_aabbMin -= motionExpansion;
    body.m_aabbMax += motionExpansion;
}

/**
 * GPU kernel for sleeping/activation state management
 */
__global__ void manageSleepingStates(btCudaRigidBody* bodies, int numBodies, btCudaScalar timeStep)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numBodies) return;
    
    btCudaRigidBody& body = bodies[idx];
    
    if (body.isStaticOrKinematicObject()) return;
    
    // Check if body wants to sleep
    if (body.wantsSleeping()) {
        if (body.getActivationState() == CUDA_ACTIVE_TAG) {
            body.setActivationState(CUDA_WANTS_DEACTIVATION);
            body.m_deactivationTime = btCudaScalar(0.0);
        }
        
        if (body.getActivationState() == CUDA_WANTS_DEACTIVATION) {
            body.m_deactivationTime += timeStep;
            if (body.m_deactivationTime > btCudaScalar(2.0)) {
                body.setActivationState(CUDA_ISLAND_SLEEPING);
                body.setLinearVelocity(btCudaVector3::zero());
                body.setAngularVelocity(btCudaVector3::zero());
            }
        }
    } else {
        // Body is moving, make sure it's active
        if (body.getActivationState() != CUDA_DISABLE_DEACTIVATION) {
            body.activate(false);
        }
    }
}

/**
 * GPU kernel for constraint force accumulation
 */
__global__ void accumulateConstraintForces(btCudaRigidBody* bodies, int numBodies,
                                          btCudaVector3* constraintForces, btCudaVector3* constraintTorques)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numBodies) return;
    
    btCudaRigidBody& body = bodies[idx];
    
    if (body.isStaticOrKinematicObject()) return;
    
    // Add accumulated constraint forces
    body.applyCentralForce(constraintForces[idx]);
    body.applyTorque(constraintTorques[idx]);
    
    // Clear constraint force accumulators
    constraintForces[idx].setZero();
    constraintTorques[idx].setZero();
}

/**
 * GPU kernel for velocity correction after constraint solving
 */
__global__ void correctVelocitiesAfterConstraints(btCudaRigidBody* bodies, int numBodies,
                                                 btCudaVector3* velocityCorrections, btCudaVector3* angularCorrections)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numBodies) return;
    
    btCudaRigidBody& body = bodies[idx];
    
    if (body.isStaticOrKinematicObject()) return;
    
    // Apply velocity corrections from constraint solver
    body.setLinearVelocity(body.getLinearVelocity() + velocityCorrections[idx]);
    body.setAngularVelocity(body.getAngularVelocity() + angularCorrections[idx]);
    
    // Apply velocity clamping for stability
    btCudaVector3 linVel = body.getLinearVelocity();
    btCudaVector3 angVel = body.getAngularVelocity();
    
    const btCudaScalar maxLinVel = btCudaScalar(1000.0);  // Adjustable limit
    const btCudaScalar maxAngVel = btCudaScalar(100.0);   // Adjustable limit
    
    if (linVel.length() > maxLinVel) {
        body.setLinearVelocity(linVel.normalized() * maxLinVel);
    }
    
    if (angVel.length() > maxAngVel) {
        body.setAngularVelocity(angVel.normalized() * maxAngVel);
    }
}

/**
 * GPU kernel for inertia tensor updates
 */
__global__ void updateInertiaTensors(btCudaRigidBody* bodies, int numBodies)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numBodies) return;
    
    bodies[idx].updateInverseInertiaTensor();
}

/**
 * GPU kernel for kinematic body updates
 */
__global__ void updateKinematicBodies(btCudaRigidBody* bodies, int numBodies, 
                                     btCudaTransform* targetTransforms, btCudaScalar timeStep)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numBodies) return;
    
    btCudaRigidBody& body = bodies[idx];
    
    if (!body.isKinematicObject()) return;
    
    // Interpolate kinematic body to target transform
    btCudaTransform currentTransform = body.getCenterOfMassTransform();
    btCudaTransform targetTransform = targetTransforms[idx];
    
    // Calculate velocity needed to reach target
    btCudaVector3 linearVelocity = (targetTransform.getOrigin() - currentTransform.getOrigin()) / timeStep;
    
    // Calculate angular velocity for rotation
    btCudaQuaternion currentRot = currentTransform.getRotation();
    btCudaQuaternion targetRot = targetTransform.getRotation();
    btCudaQuaternion deltaRot = targetRot * currentRot.inverse();
    
    btCudaVector3 axis;
    btCudaScalar angle;
    deltaRot.getAxisAngle(axis, angle);
    
    btCudaVector3 angularVelocity = axis * (angle / timeStep);
    
    // Set velocities for kinematic motion
    body.setLinearVelocity(linearVelocity);
    body.setAngularVelocity(angularVelocity);
}

/**
 * GPU kernel for applying external forces (wind, magnetic fields, etc.)
 */
__global__ void applyExternalForces(btCudaRigidBody* bodies, int numBodies,
                                   btCudaVector3 windVelocity, btCudaScalar windStrength,
                                   btCudaVector3 magneticField, btCudaScalar timeStep)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numBodies) return;
    
    btCudaRigidBody& body = bodies[idx];
    
    if (body.isStaticOrKinematicObject()) return;
    
    // Apply wind force based on velocity difference
    btCudaVector3 relativeVelocity = windVelocity - body.getLinearVelocity();
    btCudaScalar relativeSpeed = relativeVelocity.length();
    
    if (relativeSpeed > CUDA_EPSILON) {
        btCudaVector3 windForce = relativeVelocity.normalized() * (relativeSpeed * relativeSpeed * windStrength);
        body.applyCentralForce(windForce);
    }
    
    // Apply magnetic force (simplified - assumes magnetic dipole)
    if (magneticField.length() > CUDA_EPSILON) {
        btCudaVector3 magneticTorque = body.getAngularVelocity().cross(magneticField) * btCudaScalar(0.01);
        body.applyTorque(magneticTorque);
    }
}

/**
 * Host functions for rigid body management
 */

// Initialize a rigid body construction info with default values
void initializeRigidBodyConstructionInfo(btCudaRigidBodyConstructionInfo& info,
                                       btCudaScalar mass,
                                       btCudaCollisionShape* shape,
                                       const btCudaVector3& localInertia)
{
    info.m_mass = mass;
    info.m_collisionShape = shape;
    info.m_localInertia = localInertia;
    info.m_startWorldTransform.setIdentity();
    info.m_linearDamping = btCudaScalar(0.0);
    info.m_angularDamping = btCudaScalar(0.0);
    info.m_friction = btCudaScalar(0.5);
    info.m_rollingFriction = btCudaScalar(0.0);
    info.m_spinningFriction = btCudaScalar(0.0);
    info.m_restitution = btCudaScalar(0.0);
    info.m_linearSleepingThreshold = btCudaScalar(0.8);
    info.m_angularSleepingThreshold = btCudaScalar(1.0);
    info.m_additionalDamping = false;
    info.m_additionalDampingFactor = btCudaScalar(0.005);
    info.m_additionalLinearDampingThresholdSqr = btCudaScalar(0.01);
    info.m_additionalAngularDampingThresholdSqr = btCudaScalar(0.01);
    info.m_additionalAngularDampingFactor = btCudaScalar(0.01);
}

// Calculate local inertia for common shapes
void calculateLocalInertia(btCudaCollisionShape* shape, btCudaScalar mass, btCudaVector3& inertia)
{
    if (mass == btCudaScalar(0.0)) {
        inertia.setZero();
        return;
    }
    
    if (shape->getShapeType() == CUDA_BOX_SHAPE_PROXYTYPE) {
        btCudaBoxShape* boxShape = static_cast<btCudaBoxShape*>(shape);
        btCudaVector3 halfExtents = boxShape->getHalfExtentsWithoutMargin();
        
        btCudaScalar m = mass / btCudaScalar(12.0);
        inertia.setValue(
            m * (halfExtents.getY() * halfExtents.getY() + halfExtents.getZ() * halfExtents.getZ()),
            m * (halfExtents.getX() * halfExtents.getX() + halfExtents.getZ() * halfExtents.getZ()),
            m * (halfExtents.getX() * halfExtents.getX() + halfExtents.getY() * halfExtents.getY())
        );
    } else if (shape->getShapeType() == CUDA_SPHERE_SHAPE_PROXYTYPE) {
        btCudaSphereShape* sphereShape = static_cast<btCudaSphereShape*>(shape);
        btCudaScalar radius = sphereShape->getRadius();
        btCudaScalar i = btCudaScalar(0.4) * mass * radius * radius;
        inertia.setValue(i, i, i);
    } else {
        // Default inertia for unknown shapes
        inertia.setValue(mass, mass, mass);
    }
}
