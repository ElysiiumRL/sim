/*
CUDA Conversion of Bullet Physics Constraint Solver
Copyright (c) 2003-2006 Erwin Coumans  https://bulletphysics.org
CUDA Conversion: 2025

This software is provided 'as-is', without any express or implied warranty.
*/

#ifndef BT_CUDA_CONSTRAINT_SOLVER_CUH
#define BT_CUDA_CONSTRAINT_SOLVER_CUH

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "../../LinearMath/btCudaVector3.cuh"
#include "../../LinearMath/btCudaScalar.cuh"
#include "../Dynamics/btCudaRigidBody.cuh"

/**
 * CUDA Contact Point for constraint solving
 */
struct btCudaContactPoint
{
    btCudaVector3 m_positionWorldOnA;
    btCudaVector3 m_positionWorldOnB;
    btCudaVector3 m_normalWorldOnB;
    btCudaScalar m_distance;
    btCudaScalar m_combinedFriction;
    btCudaScalar m_combinedRestitution;
    int m_bodyA;
    int m_bodyB;
    
    // Solver data
    btCudaScalar m_appliedImpulse;
    btCudaScalar m_appliedImpulseLateral1;
    btCudaScalar m_appliedImpulseLateral2;
    btCudaVector3 m_lateralFrictionDir1;
    btCudaVector3 m_lateralFrictionDir2;
    
    __device__ void init()
    {
        m_positionWorldOnA.setZero();
        m_positionWorldOnB.setZero();
        m_normalWorldOnB.setZero();
        m_distance = btCudaScalar(0.0);
        m_combinedFriction = btCudaScalar(0.5);
        m_combinedRestitution = btCudaScalar(0.0);
        m_bodyA = -1;
        m_bodyB = -1;
        m_appliedImpulse = btCudaScalar(0.0);
        m_appliedImpulseLateral1 = btCudaScalar(0.0);
        m_appliedImpulseLateral2 = btCudaScalar(0.0);
        m_lateralFrictionDir1.setZero();
        m_lateralFrictionDir2.setZero();
    }
};

/**
 * CUDA Constraint structure
 */
struct btCudaConstraint
{
    int m_bodyA;
    int m_bodyB;
    btCudaVector3 m_pivotInA;
    btCudaVector3 m_pivotInB;
    btCudaVector3 m_constraintAxis;
    btCudaScalar m_lowerLimit;
    btCudaScalar m_upperLimit;
    btCudaScalar m_appliedImpulse;
    
    __device__ void init()
    {
        m_bodyA = -1;
        m_bodyB = -1;
        m_pivotInA.setZero();
        m_pivotInB.setZero();
        m_constraintAxis.setZero();
        m_lowerLimit = btCudaScalar(0.0);
        m_upperLimit = btCudaScalar(0.0);
        m_appliedImpulse = btCudaScalar(0.0);
    }
};

/**
 * CUDA Constraint Solver
 * Solves contacts and constraints in parallel on GPU
 */
class btCudaConstraintSolver
{
private:
    int m_maxConstraints;
    int m_maxContactPoints;
    dim3 m_blockSize;
    dim3 m_constraintGridSize;
    dim3 m_contactGridSize;
    
public:
    btCudaConstraintSolver(int maxConstraints, int maxContactPoints)
        : m_maxConstraints(maxConstraints), m_maxContactPoints(maxContactPoints)
    {
        m_blockSize = dim3(256);
        m_constraintGridSize = dim3((maxConstraints + m_blockSize.x - 1) / m_blockSize.x);
        m_contactGridSize = dim3((maxContactPoints + m_blockSize.x - 1) / m_blockSize.x);
    }
    
    /**
     * Main constraint solving function - GPU parallel
     */
    void solveGroup(btCudaRigidBody* rigidBodies, int numBodies,
                   btCudaContactPoint* contactPoints, int numContacts,
                   btCudaConstraint* constraints, int numConstraints,
                   btCudaScalar timeStep)
    {
        if (numContacts > 0) {
            // Solve contact constraints
            solveContactConstraints<<<m_contactGridSize, m_blockSize>>>(
                rigidBodies, numBodies, contactPoints, numContacts, timeStep);
        }
        
        if (numConstraints > 0) {
            // Solve joint constraints
            solveJointConstraints<<<m_constraintGridSize, m_blockSize>>>(
                rigidBodies, numBodies, constraints, numConstraints, timeStep);
        }
        
        cudaDeviceSynchronize();
    }
};

/**
 * GPU Kernels for constraint solving
 */

/**
 * Solve contact constraints using Sequential Impulse method
 */
__global__ void solveContactConstraints(btCudaRigidBody* rigidBodies, int numBodies,
                                       btCudaContactPoint* contacts, int numContacts,
                                       btCudaScalar timeStep)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numContacts) return;
    
    btCudaContactPoint& contact = contacts[idx];
    
    if (contact.m_bodyA < 0 || contact.m_bodyA >= numBodies ||
        contact.m_bodyB < 0 || contact.m_bodyB >= numBodies) {
        return;
    }
    
    btCudaRigidBody& bodyA = rigidBodies[contact.m_bodyA];
    btCudaRigidBody& bodyB = rigidBodies[contact.m_bodyB];
    
    // Skip if both bodies are static
    if (bodyA.isStaticOrKinematicObject() && bodyB.isStaticOrKinematicObject()) {
        return;
    }
    
    // Calculate relative position vectors
    btCudaVector3 rel_pos1 = contact.m_positionWorldOnA - bodyA.getCenterOfMassTransform().getOrigin();
    btCudaVector3 rel_pos2 = contact.m_positionWorldOnB - bodyB.getCenterOfMassTransform().getOrigin();
    
    // Calculate relative velocity
    btCudaVector3 vel1 = bodyA.getLinearVelocity() + bodyA.getAngularVelocity().cross(rel_pos1);
    btCudaVector3 vel2 = bodyB.getLinearVelocity() + bodyB.getAngularVelocity().cross(rel_pos2);
    btCudaVector3 relativeVelocity = vel1 - vel2;
    
    // Contact normal velocity
    btCudaScalar normalVelocity = relativeVelocity.dot(contact.m_normalWorldOnB);
    
    // Calculate impulse denominator
    btCudaScalar jacDiagABInv = btCudaScalar(0.0);
    
    if (!bodyA.isStaticOrKinematicObject()) {
        btCudaVector3 ftorqueAxis1 = rel_pos1.cross(contact.m_normalWorldOnB);
        jacDiagABInv += bodyA.getInverseMass() + contact.m_normalWorldOnB.dot(
            (ftorqueAxis1 * bodyA.getInvInertiaTensorWorld()).cross(rel_pos1));
    }
    
    if (!bodyB.isStaticOrKinematicObject()) {
        btCudaVector3 ftorqueAxis2 = rel_pos2.cross(-contact.m_normalWorldOnB);
        jacDiagABInv += bodyB.getInverseMass() + (-contact.m_normalWorldOnB).dot(
            (ftorqueAxis2 * bodyB.getInvInertiaTensorWorld()).cross(rel_pos2));
    }
    
    if (jacDiagABInv < CUDA_EPSILON) return;
    
    // Calculate impulse magnitude
    btCudaScalar impulse = btCudaScalar(0.0);
    
    if (contact.m_distance < btCudaScalar(0.0)) {
        // Penetration resolution
        btCudaScalar velocityImpulse = -normalVelocity / jacDiagABInv;
        btCudaScalar positionImpulse = contact.m_distance / (timeStep * jacDiagABInv);
        impulse = velocityImpulse + positionImpulse;
        
        // Add restitution
        if (normalVelocity < -btCudaScalar(1.0)) {
            impulse += -contact.m_combinedRestitution * normalVelocity / jacDiagABInv;
        }
        
        // Accumulate impulse
        btCudaScalar oldImpulse = contact.m_appliedImpulse;
        contact.m_appliedImpulse = btCudaMax(btCudaScalar(0.0), oldImpulse + impulse);
        impulse = contact.m_appliedImpulse - oldImpulse;
        
        // Apply impulse to bodies
        btCudaVector3 impulseVector = impulse * contact.m_normalWorldOnB;
        
        if (!bodyA.isStaticOrKinematicObject()) {
            bodyA.applyCentralImpulse(impulseVector);
            bodyA.applyTorqueImpulse(rel_pos1.cross(impulseVector));
        }
        
        if (!bodyB.isStaticOrKinematicObject()) {
            bodyB.applyCentralImpulse(-impulseVector);
            bodyB.applyTorqueImpulse(rel_pos2.cross(-impulseVector));
        }
    }
    
    // Friction solving (simplified)
    if (contact.m_combinedFriction > btCudaScalar(0.0) && impulse > btCudaScalar(0.0)) {
        // Calculate friction directions
        btCudaVector3 tangent = relativeVelocity - normalVelocity * contact.m_normalWorldOnB;
        btCudaScalar tangentSpeed = tangent.length();
        
        if (tangentSpeed > CUDA_EPSILON) {
            tangent /= tangentSpeed;
            
            // Calculate friction impulse
            btCudaScalar frictionImpulse = -tangentSpeed / jacDiagABInv;
            btCudaScalar maxFriction = contact.m_combinedFriction * contact.m_appliedImpulse;
            frictionImpulse = btCudaClamp(frictionImpulse, -maxFriction, maxFriction);
            
            // Apply friction impulse
            btCudaVector3 frictionVector = frictionImpulse * tangent;
            
            if (!bodyA.isStaticOrKinematicObject()) {
                bodyA.applyCentralImpulse(frictionVector);
                bodyA.applyTorqueImpulse(rel_pos1.cross(frictionVector));
            }
            
            if (!bodyB.isStaticOrKinematicObject()) {
                bodyB.applyCentralImpulse(-frictionVector);
                bodyB.applyTorqueImpulse(rel_pos2.cross(-frictionVector));
            }
        }
    }
}

/**
 * Solve joint constraints
 */
__global__ void solveJointConstraints(btCudaRigidBody* rigidBodies, int numBodies,
                                     btCudaConstraint* constraints, int numConstraints,
                                     btCudaScalar timeStep)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numConstraints) return;
    
    btCudaConstraint& constraint = constraints[idx];
    
    if (constraint.m_bodyA < 0 || constraint.m_bodyA >= numBodies ||
        constraint.m_bodyB < 0 || constraint.m_bodyB >= numBodies) {
        return;
    }
    
    btCudaRigidBody& bodyA = rigidBodies[constraint.m_bodyA];
    btCudaRigidBody& bodyB = rigidBodies[constraint.m_bodyB];
    
    // Skip if both bodies are static
    if (bodyA.isStaticOrKinematicObject() && bodyB.isStaticOrKinematicObject()) {
        return;
    }
    
    // Transform constraint points to world space
    btCudaVector3 pivotA = bodyA.getCenterOfMassTransform() * constraint.m_pivotInA;
    btCudaVector3 pivotB = bodyB.getCenterOfMassTransform() * constraint.m_pivotInB;
    
    // Calculate constraint error
    btCudaVector3 constraintError = pivotA - pivotB;
    btCudaScalar error = constraintError.dot(constraint.m_constraintAxis);
    
    // Calculate relative velocity along constraint axis
    btCudaVector3 velA = bodyA.getVelocityInLocalPoint(constraint.m_pivotInA);
    btCudaVector3 velB = bodyB.getVelocityInLocalPoint(constraint.m_pivotInB);
    btCudaVector3 relativeVelocity = velA - velB;
    btCudaScalar axisVelocity = relativeVelocity.dot(constraint.m_constraintAxis);
    
    // Calculate jacobian
    btCudaScalar jacDiagABInv = btCudaScalar(0.0);
    
    if (!bodyA.isStaticOrKinematicObject()) {
        jacDiagABInv += bodyA.computeImpulseDenominator(pivotA, constraint.m_constraintAxis);
    }
    
    if (!bodyB.isStaticOrKinematicObject()) {
        jacDiagABInv += bodyB.computeImpulseDenominator(pivotB, -constraint.m_constraintAxis);
    }
    
    if (jacDiagABInv < CUDA_EPSILON) return;
    
    // Calculate impulse
    btCudaScalar impulse = -(axisVelocity + error / timeStep) / jacDiagABInv;
    
    // Apply limits
    btCudaScalar oldImpulse = constraint.m_appliedImpulse;
    constraint.m_appliedImpulse = btCudaClamp(oldImpulse + impulse, constraint.m_lowerLimit, constraint.m_upperLimit);
    impulse = constraint.m_appliedImpulse - oldImpulse;
    
    // Apply impulse
    btCudaVector3 impulseVector = impulse * constraint.m_constraintAxis;
    
    if (!bodyA.isStaticOrKinematicObject()) {
        bodyA.applyImpulse(impulseVector, constraint.m_pivotInA);
    }
    
    if (!bodyB.isStaticOrKinematicObject()) {
        bodyB.applyImpulse(-impulseVector, constraint.m_pivotInB);
    }
}

#endif // BT_CUDA_CONSTRAINT_SOLVER_CUH
