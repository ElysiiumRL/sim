/*
CUDA Conversion of Bullet Physics btTransform
Copyright (c) 2003-2006 Gino van den Bergen / Erwin Coumans  https://bulletphysics.org
CUDA Conversion: 2025

This software is provided 'as-is', without any express or implied warranty.
*/

#ifndef BT_CUDA_TRANSFORM_CUH
#define BT_CUDA_TRANSFORM_CUH

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "btCudaMatrix3x3.cuh"
#include "btCudaVector3.cuh"
#include "btCudaQuaternion.cuh"

/**
 * CUDA-accelerated Transform class
 * Represents a rigid body transformation (rotation + translation)
 * All operations optimized for GPU execution
 */
class btCudaTransform
{
private:
    btCudaMatrix3x3 m_basis;      // Rotation matrix
    btCudaVector3   m_origin;     // Translation vector

public:
    /**
     * Constructors
     */
    __device__ __host__ btCudaTransform() {}

    __device__ __host__ explicit btCudaTransform(const btCudaQuaternion& q, const btCudaVector3& c = btCudaVector3::zero())
        : m_basis(q), m_origin(c) {}

    __device__ __host__ explicit btCudaTransform(const btCudaMatrix3x3& b, const btCudaVector3& c = btCudaVector3::zero())
        : m_basis(b), m_origin(c) {}

    __device__ __host__ btCudaTransform(const btCudaTransform& other)
        : m_basis(other.m_basis), m_origin(other.m_origin) {}

    /**
     * Assignment operator
     */
    __device__ __host__ btCudaTransform& operator=(const btCudaTransform& other)
    {
        m_basis = other.m_basis;
        m_origin = other.m_origin;
        return *this;
    }

    /**
     * Element access
     */
    __device__ __host__ const btCudaMatrix3x3& getBasis() const { return m_basis; }
    __device__ __host__ btCudaMatrix3x3& getBasis() { return m_basis; }
    
    __device__ __host__ const btCudaVector3& getOrigin() const { return m_origin; }
    __device__ __host__ btCudaVector3& getOrigin() { return m_origin; }

    __device__ __host__ void setBasis(const btCudaMatrix3x3& basis) { m_basis = basis; }
    __device__ __host__ void setOrigin(const btCudaVector3& origin) { m_origin = origin; }

    /**
     * Get rotation as quaternion
     */
    __device__ __host__ btCudaQuaternion getRotation() const
    {
        return m_basis.getRotation();
    }

    /**
     * Set rotation from quaternion
     */
    __device__ __host__ void setRotation(const btCudaQuaternion& q)
    {
        m_basis.setRotation(q);
    }

    /**
     * Set identity transform
     */
    __device__ __host__ void setIdentity()
    {
        m_basis.setIdentity();
        m_origin.setZero();
    }

    /**
     * Static identity transform
     */
    __device__ __host__ static btCudaTransform getIdentity()
    {
        btCudaTransform identity;
        identity.setIdentity();
        return identity;
    }

    /**
     * Transform multiplication (composition) - GPU optimized
     */
    __device__ __host__ btCudaTransform operator*(const btCudaTransform& t) const
    {
        return btCudaTransform(m_basis * t.m_basis, (*this)(t.m_origin));
    }

    /**
     * In-place transform multiplication
     */
    __device__ __host__ btCudaTransform& operator*=(const btCudaTransform& t)
    {
        m_origin += m_basis * t.m_origin;
        m_basis *= t.m_basis;
        return *this;
    }

    /**
     * Transform a point - GPU optimized
     */
    __device__ __host__ btCudaVector3 operator()(const btCudaVector3& x) const
    {
        return m_basis * x + m_origin;
    }

    /**
     * Transform a vector (rotation only) - GPU optimized
     */
    __device__ __host__ btCudaVector3 operator*(const btCudaVector3& x) const
    {
        return m_basis * x;
    }

    /**
     * Inverse transform - GPU optimized
     */
    __device__ __host__ btCudaTransform inverse() const
    {
        btCudaMatrix3x3 inv = m_basis.transpose();
        return btCudaTransform(inv, inv * -m_origin);
    }

    /**
     * In-place inverse
     */
    __device__ __host__ btCudaTransform& invert()
    {
        m_basis = m_basis.transpose();
        m_origin = m_basis * -m_origin;
        return *this;
    }

    /**
     * Inverse transform a point - GPU optimized
     */
    __device__ __host__ btCudaVector3 invXform(const btCudaVector3& inVec) const
    {
        btCudaVector3 v = inVec - m_origin;
        return m_basis.transpose() * v;
    }

    /**
     * Interpolation between transforms - GPU optimized
     */
    __device__ __host__ void lerp(const btCudaTransform& t1, const btCudaTransform& t2, const btCudaScalar& t)
    {
        setOrigin(t1.getOrigin().lerp(t2.getOrigin(), t));
        setRotation(t1.getRotation().slerp(t2.getRotation(), t));
    }

    /**
     * Static interpolation function
     */
    __device__ __host__ static btCudaTransform lerp(const btCudaTransform& t1, const btCudaTransform& t2, const btCudaScalar& t)
    {
        btCudaTransform result;
        result.lerp(t1, t2, t);
        return result;
    }

    /**
     * Get Euler angles from transform
     */
    __device__ __host__ void getEulerYPR(btCudaScalar& yaw, btCudaScalar& pitch, btCudaScalar& roll) const
    {
        m_basis.getEulerYPR(yaw, pitch, roll);
    }

    /**
     * Set from Euler angles
     */
    __device__ __host__ void setEulerYPR(const btCudaScalar& yaw, const btCudaScalar& pitch, const btCudaScalar& roll)
    {
        setRotation(btCudaQuaternion(yaw, pitch, roll));
    }

    /**
     * Utility functions
     */
    __device__ __host__ void setFromOpenGLMatrix(const btCudaScalar* m)
    {
        m_basis.setValue(m[0], m[4], m[8],
                        m[1], m[5], m[9],
                        m[2], m[6], m[10]);
        m_origin.setValue(m[12], m[13], m[14]);
    }

    __device__ __host__ void getOpenGLMatrix(btCudaScalar* m) const
    {
        m[0] = m_basis[0][0]; m[4] = m_basis[0][1]; m[8] = m_basis[0][2]; m[12] = m_origin[0];
        m[1] = m_basis[1][0]; m[5] = m_basis[1][1]; m[9] = m_basis[1][2]; m[13] = m_origin[1];
        m[2] = m_basis[2][0]; m[6] = m_basis[2][1]; m[10] = m_basis[2][2]; m[14] = m_origin[2];
        m[3] = btCudaScalar(0.0); m[7] = btCudaScalar(0.0); m[11] = btCudaScalar(0.0); m[15] = btCudaScalar(1.0);
    }

    /**
     * Get transformed AABB - GPU optimized
     */
    __device__ __host__ void transformAabb(const btCudaVector3& localAabbMin, const btCudaVector3& localAabbMax,
                                          btCudaVector3& aabbMinOut, btCudaVector3& aabbMaxOut) const
    {
        btCudaVector3 center = (localAabbMax + localAabbMin) * btCudaScalar(0.5);
        btCudaVector3 extent = (localAabbMax - localAabbMin) * btCudaScalar(0.5);
        
        btCudaVector3 newCenter = (*this)(center);
        
        // Transform extent using absolute values of rotation matrix
        btCudaVector3 newExtent(
            btCudaFabs(m_basis[0][0]) * extent[0] + btCudaFabs(m_basis[0][1]) * extent[1] + btCudaFabs(m_basis[0][2]) * extent[2],
            btCudaFabs(m_basis[1][0]) * extent[0] + btCudaFabs(m_basis[1][1]) * extent[1] + btCudaFabs(m_basis[1][2]) * extent[2],
            btCudaFabs(m_basis[2][0]) * extent[0] + btCudaFabs(m_basis[2][1]) * extent[1] + btCudaFabs(m_basis[2][2]) * extent[2]
        );
        
        aabbMinOut = newCenter - newExtent;
        aabbMaxOut = newCenter + newExtent;
    }

    /**
     * Integration with velocity - GPU optimized
     * Used for physics integration step
     */
    __device__ __host__ void integrateTransform(const btCudaVector3& linearVelocity, const btCudaVector3& angularVelocity, btCudaScalar timeStep)
    {
        // Update position
        m_origin += linearVelocity * timeStep;
        
        // Update rotation using quaternion integration
        btCudaQuaternion dorn = btCudaQuaternion(angularVelocity[0], angularVelocity[1], angularVelocity[2], btCudaScalar(0.0)) * getRotation();
        btCudaQuaternion orn = getRotation() + dorn * (timeStep * btCudaScalar(0.5));
        orn.normalize();
        setRotation(orn);
    }

    /**
     * Calculate relative transform - GPU optimized
     */
    __device__ __host__ static btCudaTransform relativeTo(const btCudaTransform& base, const btCudaTransform& other)
    {
        return base.inverse() * other;
    }

    /**
     * Create look-at transform - GPU optimized
     */
    __device__ __host__ static btCudaTransform lookAt(const btCudaVector3& eye, const btCudaVector3& target, const btCudaVector3& up)
    {
        btCudaVector3 zaxis = (eye - target).normalized();
        btCudaVector3 xaxis = up.cross(zaxis).normalized();
        btCudaVector3 yaxis = zaxis.cross(xaxis);
        
        btCudaMatrix3x3 rotation;
        rotation.setValue(
            xaxis.getX(), yaxis.getX(), zaxis.getX(),
            xaxis.getY(), yaxis.getY(), zaxis.getY(),
            xaxis.getZ(), yaxis.getZ(), zaxis.getZ()
        );
        
        return btCudaTransform(rotation, eye);
    }
};

/**
 * Global utility functions
 */
__device__ __host__ inline btCudaTransform btCudaLerp(const btCudaTransform& t1, const btCudaTransform& t2, const btCudaScalar& t)
{
    return btCudaTransform::lerp(t1, t2, t);
}

#endif // BT_CUDA_TRANSFORM_CUH
