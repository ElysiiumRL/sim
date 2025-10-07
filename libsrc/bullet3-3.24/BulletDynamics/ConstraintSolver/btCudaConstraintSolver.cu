/*
CUDA Conversion of Bullet Physics Constraint Solver - COMPLETE GPU IMPLEMENTATION
Copyright (c) 2003-2006 Erwin Coumans  https://bulletphysics.org
CUDA Conversion: 2025

This software is provided 'as-is', without any express or implied warranty.
*/

#include "btCudaConstraintSolver.cuh"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cooperative_groups.h>

using namespace cooperative_groups;

/**
 * CUDA Sequential Impulse Constraint Solver - COMPLETE GPU IMPLEMENTATION
 * This replaces the entire CPU constraint solver with GPU parallel algorithms
 */

/**
 * GPU kernel for parallel contact preparation
 */
__global__ void prepareContactConstraints(btCudaRigidBody* bodies, int numBodies,
                                        btCudaContactPoint* contacts, int numContacts,
                                        btCudaScalar* jacobianDiagonals, btCudaScalar timeStep)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numContacts) return;
    
    btCudaContactPoint& contact = contacts[idx];
    
    if (contact.m_bodyA < 0 || contact.m_bodyA >= numBodies ||
        contact.m_bodyB < 0 || contact.m_bodyB >= numBodies) {
        jacobianDiagonals[idx] = btCudaScalar(0.0);
        return;
    }
    
    btCudaRigidBody& bodyA = bodies[contact.m_bodyA];
    btCudaRigidBody& bodyB = bodies[contact.m_bodyB];
    
    // Calculate contact points relative to center of mass
    btCudaVector3 rel_pos1 = contact.m_positionWorldOnA - bodyA.getCenterOfMassTransform().getOrigin();
    btCudaVector3 rel_pos2 = contact.m_positionWorldOnB - bodyB.getCenterOfMassTransform().getOrigin();
    
    // Calculate jacobian diagonal for normal constraint
    btCudaScalar jacDiagABInv = btCudaScalar(0.0);
    
    if (!bodyA.isStaticOrKinematicObject()) {
        jacDiagABInv += bodyA.computeImpulseDenominator(contact.m_positionWorldOnA, contact.m_normalWorldOnB);
    }
    
    if (!bodyB.isStaticOrKinematicObject()) {
        jacDiagABInv += bodyB.computeImpulseDenominator(contact.m_positionWorldOnB, -contact.m_normalWorldOnB);
    }
    
    jacobianDiagonals[idx] = jacDiagABInv;
    
    // Initialize lateral friction directions
    btCudaVector3 normal = contact.m_normalWorldOnB;
    btCudaVector3 tangent1, tangent2;
    
    // Create orthonormal basis
    if (btCudaFabs(normal.getX()) > btCudaScalar(0.7)) {
        tangent1 = btCudaVector3(normal.getY(), -normal.getX(), btCudaScalar(0.0));
    } else {
        tangent1 = btCudaVector3(btCudaScalar(0.0), normal.getZ(), -normal.getY());
    }
    tangent1.normalize();
    tangent2 = normal.cross(tangent1);
    
    contact.m_lateralFrictionDir1 = tangent1;
    contact.m_lateralFrictionDir2 = tangent2;
}

/**
 * GPU kernel for parallel contact constraint solving using Sequential Impulse Method
 */
__global__ void solveContactConstraintsParallel(btCudaRigidBody* bodies, int numBodies,
                                               btCudaContactPoint* contacts, int numContacts,
                                               btCudaScalar* jacobianDiagonals, btCudaScalar timeStep,
                                               int iteration, int maxIterations)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numContacts) return;
    
    btCudaContactPoint& contact = contacts[idx];
    btCudaScalar jacDiagABInv = jacobianDiagonals[idx];
    
    if (jacDiagABInv < CUDA_EPSILON) return;
    
    btCudaRigidBody& bodyA = bodies[contact.m_bodyA];
    btCudaRigidBody& bodyB = bodies[contact.m_bodyB];
    
    // Calculate relative position vectors
    btCudaVector3 rel_pos1 = contact.m_positionWorldOnA - bodyA.getCenterOfMassTransform().getOrigin();
    btCudaVector3 rel_pos2 = contact.m_positionWorldOnB - bodyB.getCenterOfMassTransform().getOrigin();
    
    // Calculate relative velocity at contact point
    btCudaVector3 vel1 = bodyA.getLinearVelocity() + bodyA.getAngularVelocity().cross(rel_pos1);
    btCudaVector3 vel2 = bodyB.getLinearVelocity() + bodyB.getAngularVelocity().cross(rel_pos2);
    btCudaVector3 relativeVelocity = vel1 - vel2;
    
    // Solve normal constraint
    btCudaScalar normalVelocity = relativeVelocity.dot(contact.m_normalWorldOnB);
    
    // Calculate constraint violation with Baumgarte stabilization
    btCudaScalar baumgarte = btCudaScalar(0.2) / timeStep;
    btCudaScalar positionError = btCudaMin(btCudaScalar(0.0), contact.m_distance + btCudaScalar(0.02));  // Allow 2cm penetration
    btCudaScalar velocityError = normalVelocity;
    
    // Add restitution
    if (normalVelocity < -btCudaScalar(1.0)) {
        velocityError += contact.m_combinedRestitution * normalVelocity;
    }
    
    btCudaScalar deltaImpulse = -(velocityError + baumgarte * positionError) / jacDiagABInv;
    
    // Accumulate impulse with non-negativity constraint
    btCudaScalar oldImpulse = contact.m_appliedImpulse;
    contact.m_appliedImpulse = btCudaMax(btCudaScalar(0.0), oldImpulse + deltaImpulse);
    deltaImpulse = contact.m_appliedImpulse - oldImpulse;
    
    // Apply normal impulse
    btCudaVector3 impulse = deltaImpulse * contact.m_normalWorldOnB;
    
    if (!bodyA.isStaticOrKinematicObject()) {
        bodyA.applyCentralImpulse(impulse);
        bodyA.applyTorqueImpulse(rel_pos1.cross(impulse));
    }
    
    if (!bodyB.isStaticOrKinematicObject()) {
        bodyB.applyCentralImpulse(-impulse);
        bodyB.applyTorqueImpulse(rel_pos2.cross(-impulse));
    }
    
    // Solve friction constraints if normal impulse is positive
    if (contact.m_appliedImpulse > btCudaScalar(0.0) && contact.m_combinedFriction > btCudaScalar(0.0)) {
        solveFrictionConstraintGPU(bodyA, bodyB, contact, rel_pos1, rel_pos2, jacDiagABInv);
    }
}

/**
 * GPU device function for friction constraint solving
 */
__device__ void solveFrictionConstraintGPU(btCudaRigidBody& bodyA, btCudaRigidBody& bodyB,
                                          btCudaContactPoint& contact,
                                          const btCudaVector3& rel_pos1, const btCudaVector3& rel_pos2,
                                          btCudaScalar normalJacDiag)
{
    // Recalculate relative velocity after normal impulse
    btCudaVector3 vel1 = bodyA.getLinearVelocity() + bodyA.getAngularVelocity().cross(rel_pos1);
    btCudaVector3 vel2 = bodyB.getLinearVelocity() + bodyB.getAngularVelocity().cross(rel_pos2);
    btCudaVector3 relativeVelocity = vel1 - vel2;
    
    // Project relative velocity onto friction directions
    btCudaScalar lat1Vel = relativeVelocity.dot(contact.m_lateralFrictionDir1);
    btCudaScalar lat2Vel = relativeVelocity.dot(contact.m_lateralFrictionDir2);
    
    // Calculate friction impulses
    btCudaScalar maxFriction = contact.m_combinedFriction * contact.m_appliedImpulse;
    
    // Solve first lateral friction direction
    btCudaScalar deltaImpulseLat1 = -lat1Vel / normalJacDiag;
    btCudaScalar oldImpulseLat1 = contact.m_appliedImpulseLateral1;
    contact.m_appliedImpulseLateral1 = btCudaClamp(oldImpulseLat1 + deltaImpulseLat1, -maxFriction, maxFriction);
    deltaImpulseLat1 = contact.m_appliedImpulseLateral1 - oldImpulseLat1;
    
    // Apply first lateral impulse
    btCudaVector3 impulseLat1 = deltaImpulseLat1 * contact.m_lateralFrictionDir1;
    
    if (!bodyA.isStaticOrKinematicObject()) {
        bodyA.applyCentralImpulse(impulseLat1);
        bodyA.applyTorqueImpulse(rel_pos1.cross(impulseLat1));
    }
    
    if (!bodyB.isStaticOrKinematicObject()) {
        bodyB.applyCentralImpulse(-impulseLat1);
        bodyB.applyTorqueImpulse(rel_pos2.cross(-impulseLat1));
    }
    
    // Solve second lateral friction direction
    btCudaScalar deltaImpulseLat2 = -lat2Vel / normalJacDiag;
    btCudaScalar oldImpulseLat2 = contact.m_appliedImpulseLateral2;
    contact.m_appliedImpulseLateral2 = btCudaClamp(oldImpulseLat2 + deltaImpulseLat2, -maxFriction, maxFriction);
    deltaImpulseLat2 = contact.m_appliedImpulseLateral2 - oldImpulseLat2;
    
    // Apply second lateral impulse
    btCudaVector3 impulseLat2 = deltaImpulseLat2 * contact.m_lateralFrictionDir2;
    
    if (!bodyA.isStaticOrKinematicObject()) {
        bodyA.applyCentralImpulse(impulseLat2);
        bodyA.applyTorqueImpulse(rel_pos1.cross(impulseLat2));
    }
    
    if (!bodyB.isStaticOrKinematicObject()) {
        bodyB.applyCentralImpulse(-impulseLat2);
        bodyB.applyTorqueImpulse(rel_pos2.cross(-impulseLat2));
    }
}

/**
 * GPU kernel for joint constraint solving
 */
__global__ void solveJointConstraintsParallel(btCudaRigidBody* bodies, int numBodies,
                                             btCudaConstraint* constraints, int numConstraints,
                                             btCudaScalar timeStep, int iteration)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numConstraints) return;
    
    btCudaConstraint& constraint = constraints[idx];
    
    if (constraint.m_bodyA < 0 || constraint.m_bodyA >= numBodies ||
        constraint.m_bodyB < 0 || constraint.m_bodyB >= numBodies) {
        return;
    }
    
    btCudaRigidBody& bodyA = bodies[constraint.m_bodyA];
    btCudaRigidBody& bodyB = bodies[constraint.m_bodyB];
    
    // Transform constraint points to world space
    btCudaVector3 pivotA = bodyA.getCenterOfMassTransform() * constraint.m_pivotInA;
    btCudaVector3 pivotB = bodyB.getCenterOfMassTransform() * constraint.m_pivotInB;
    
    // Calculate constraint error
    btCudaVector3 constraintError = pivotA - pivotB;
    btCudaScalar error = constraintError.dot(constraint.m_constraintAxis);
    
    // Calculate relative velocity
    btCudaVector3 relA = constraint.m_pivotInA;
    btCudaVector3 relB = constraint.m_pivotInB;
    btCudaVector3 velA = bodyA.getVelocityInLocalPoint(relA);
    btCudaVector3 velB = bodyB.getVelocityInLocalPoint(relB);
    btCudaVector3 relativeVelocity = velA - velB;
    btCudaScalar axisVelocity = relativeVelocity.dot(constraint.m_constraintAxis);
    
    // Calculate effective mass
    btCudaScalar effectiveMass = btCudaScalar(0.0);
    
    if (!bodyA.isStaticOrKinematicObject()) {
        effectiveMass += bodyA.computeImpulseDenominator(pivotA, constraint.m_constraintAxis);
    }
    
    if (!bodyB.isStaticOrKinematicObject()) {
        effectiveMass += bodyB.computeImpulseDenominator(pivotB, -constraint.m_constraintAxis);
    }
    
    if (effectiveMass < CUDA_EPSILON) return;
    
    // Calculate impulse with Baumgarte stabilization
    btCudaScalar baumgarte = btCudaScalar(0.2) / timeStep;
    btCudaScalar deltaImpulse = -(axisVelocity + baumgarte * error) / effectiveMass;
    
    // Apply limits
    btCudaScalar oldImpulse = constraint.m_appliedImpulse;
    constraint.m_appliedImpulse = btCudaClamp(oldImpulse + deltaImpulse, constraint.m_lowerLimit, constraint.m_upperLimit);
    deltaImpulse = constraint.m_appliedImpulse - oldImpulse;
    
    // Apply impulse
    btCudaVector3 impulse = deltaImpulse * constraint.m_constraintAxis;
    
    if (!bodyA.isStaticOrKinematicObject()) {
        bodyA.applyImpulse(impulse, relA);
    }
    
    if (!bodyB.isStaticOrKinematicObject()) {
        bodyB.applyImpulse(-impulse, relB);
    }
}

/**
 * GPU kernel for warm starting contact constraints
 */
__global__ void warmStartContactConstraints(btCudaRigidBody* bodies, int numBodies,
                                          btCudaContactPoint* contacts, int numContacts)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numContacts) return;
    
    btCudaContactPoint& contact = contacts[idx];
    
    if (contact.m_bodyA < 0 || contact.m_bodyA >= numBodies ||
        contact.m_bodyB < 0 || contact.m_bodyB >= numBodies) {
        return;
    }
    
    btCudaRigidBody& bodyA = bodies[contact.m_bodyA];
    btCudaRigidBody& bodyB = bodies[contact.m_bodyB];
    
    // Apply cached impulses for warm starting
    btCudaVector3 rel_pos1 = contact.m_positionWorldOnA - bodyA.getCenterOfMassTransform().getOrigin();
    btCudaVector3 rel_pos2 = contact.m_positionWorldOnB - bodyB.getCenterOfMassTransform().getOrigin();
    
    // Apply normal impulse
    btCudaVector3 normalImpulse = contact.m_appliedImpulse * contact.m_normalWorldOnB;
    
    if (!bodyA.isStaticOrKinematicObject()) {
        bodyA.applyCentralImpulse(normalImpulse);
        bodyA.applyTorqueImpulse(rel_pos1.cross(normalImpulse));
    }
    
    if (!bodyB.isStaticOrKinematicObject()) {
        bodyB.applyCentralImpulse(-normalImpulse);
        bodyB.applyTorqueImpulse(rel_pos2.cross(-normalImpulse));
    }
    
    // Apply lateral impulses
    btCudaVector3 lateralImpulse1 = contact.m_appliedImpulseLateral1 * contact.m_lateralFrictionDir1;
    btCudaVector3 lateralImpulse2 = contact.m_appliedImpulseLateral2 * contact.m_lateralFrictionDir2;
    
    if (!bodyA.isStaticOrKinematicObject()) {
        bodyA.applyCentralImpulse(lateralImpulse1 + lateralImpulse2);
        bodyA.applyTorqueImpulse(rel_pos1.cross(lateralImpulse1 + lateralImpulse2));
    }
    
    if (!bodyB.isStaticOrKinematicObject()) {
        bodyB.applyCentralImpulse(-(lateralImpulse1 + lateralImpulse2));
        bodyB.applyTorqueImpulse(rel_pos2.cross(-(lateralImpulse1 + lateralImpulse2)));
    }
}

/**
 * GPU kernel for constraint force mixing (CFM) regularization
 */
__global__ void applyCFMRegularization(btCudaContactPoint* contacts, int numContacts,
                                      btCudaScalar* jacobianDiagonals, btCudaScalar cfm)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numContacts) return;
    
    // Add CFM to diagonal for regularization
    jacobianDiagonals[idx] += cfm;
}

/**
 * Host implementation of btCudaConstraintSolver
 */
void btCudaConstraintSolver::solveGroup(btCudaRigidBody* rigidBodies, int numBodies,
                                       btCudaContactPoint* contactPoints, int numContacts,
                                       btCudaConstraint* constraints, int numConstraints,
                                       btCudaScalar timeStep)
{
    if (numContacts == 0 && numConstraints == 0) return;
    
    // Allocate temporary GPU memory for solver data
    btCudaScalar* d_jacobianDiagonals = nullptr;
    if (numContacts > 0) {
        cudaMalloc(&d_jacobianDiagonals, numContacts * sizeof(btCudaScalar));
    }
    
    // Constraint solving parameters
    const int maxIterations = 10;
    const btCudaScalar cfm = btCudaScalar(1e-8);  // Constraint Force Mixing for regularization
    
    if (numContacts > 0) {
        // Prepare contact constraints
        prepareContactConstraints<<<m_contactGridSize, m_blockSize>>>(
            rigidBodies, numBodies, contactPoints, numContacts, d_jacobianDiagonals, timeStep);
        
        // Apply CFM regularization
        applyCFMRegularization<<<m_contactGridSize, m_blockSize>>>(
            contactPoints, numContacts, d_jacobianDiagonals, cfm);
        
        // Warm start with cached impulses
        warmStartContactConstraints<<<m_contactGridSize, m_blockSize>>>(
            rigidBodies, numBodies, contactPoints, numContacts);
        
        // Iterative constraint solving
        for (int iteration = 0; iteration < maxIterations; iteration++) {
            // Solve contact constraints
            solveContactConstraintsParallel<<<m_contactGridSize, m_blockSize>>>(
                rigidBodies, numBodies, contactPoints, numContacts, d_jacobianDiagonals,
                timeStep, iteration, maxIterations);
            
            // Synchronize between iterations for stability
            if (iteration < maxIterations - 1) {
                cudaDeviceSynchronize();
            }
        }
    }
    
    if (numConstraints > 0) {
        // Solve joint constraints
        for (int iteration = 0; iteration < maxIterations; iteration++) {
            solveJointConstraintsParallel<<<m_constraintGridSize, m_blockSize>>>(
                rigidBodies, numBodies, constraints, numConstraints, timeStep, iteration);
            
            if (iteration < maxIterations - 1) {
                cudaDeviceSynchronize();
            }
        }
    }
    
    // Cleanup temporary memory
    if (d_jacobianDiagonals) {
        cudaFree(d_jacobianDiagonals);
    }
    
    // Final synchronization
    cudaDeviceSynchronize();
}

/**
 * Advanced constraint solver with split impulse and position correction
 */
void btCudaConstraintSolver::solveGroupAdvanced(btCudaRigidBody* rigidBodies, int numBodies,
                                               btCudaContactPoint* contactPoints, int numContacts,
                                               btCudaConstraint* constraints, int numConstraints,
                                               btCudaScalar timeStep)
{
    // Split impulse method: solve velocity and position separately
    
    // Phase 1: Velocity solving
    solveGroup(rigidBodies, numBodies, contactPoints, numContacts, constraints, numConstraints, timeStep);
    
    // Phase 2: Position correction (non-penetration constraints)
    if (numContacts > 0) {
        solvePositionConstraints<<<m_contactGridSize, m_blockSize>>>(
            rigidBodies, numBodies, contactPoints, numContacts, timeStep);
    }
}

/**
 * GPU kernel for position constraint solving
 */
__global__ void solvePositionConstraints(btCudaRigidBody* bodies, int numBodies,
                                        btCudaContactPoint* contacts, int numContacts,
                                        btCudaScalar timeStep)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numContacts) return;
    
    btCudaContactPoint& contact = contacts[idx];
    
    if (contact.m_distance >= btCudaScalar(0.0)) return;  // Only solve penetrating contacts
    
    btCudaRigidBody& bodyA = bodies[contact.m_bodyA];
    btCudaRigidBody& bodyB = bodies[contact.m_bodyB];
    
    // Calculate position correction to resolve penetration
    btCudaScalar penetration = -contact.m_distance;
    btCudaScalar correction = penetration * btCudaScalar(0.8);  // Percentage of penetration to resolve
    
    // Split correction between bodies based on inverse mass
    btCudaScalar totalInvMass = bodyA.getInverseMass() + bodyB.getInverseMass();
    if (totalInvMass < CUDA_EPSILON) return;
    
    btCudaVector3 correctionVector = contact.m_normalWorldOnB * correction;
    
    if (!bodyA.isStaticOrKinematicObject()) {
        btCudaScalar ratioA = bodyA.getInverseMass() / totalInvMass;
        btCudaTransform newTransformA = bodyA.getCenterOfMassTransform();
        newTransformA.getOrigin() += correctionVector * ratioA;
        bodyA.setCenterOfMassTransform(newTransformA);
    }
    
    if (!bodyB.isStaticOrKinematicObject()) {
        btCudaScalar ratioB = bodyB.getInverseMass() / totalInvMass;
        btCudaTransform newTransformB = bodyB.getCenterOfMassTransform();
        newTransformB.getOrigin() -= correctionVector * ratioB;
        bodyB.setCenterOfMassTransform(newTransformB);
    }
}
