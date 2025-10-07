/*
CUDA Conversion of Bullet Physics btQuaternion
Copyright (c) 2003-2006 Gino van den Bergen / Erwin Coumans  https://bulletphysics.org
CUDA Conversion: 2025

This software is provided 'as-is', without any express or implied warranty.
*/

#ifndef BT_CUDA_QUATERNION_CUH
#define BT_CUDA_QUATERNION_CUH

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "btCudaVector3.cuh"
#include "btCudaScalar.cuh"

// Forward declaration to avoid circular dependency
class btCudaMatrix3x3;

/**
 * CUDA-accelerated Quaternion class
 * All operations optimized for GPU execution
 */
class btCudaQuaternion
{
private:
    btCudaScalar m_floats[4];  // x, y, z, w

public:
    /**
     * Constructors
     */
    __device__ __host__ btCudaQuaternion() {}

    __device__ __host__ btCudaQuaternion(const btCudaScalar& _x, const btCudaScalar& _y, const btCudaScalar& _z, const btCudaScalar& _w)
    {
        m_floats[0] = _x;
        m_floats[1] = _y;
        m_floats[2] = _z;
        m_floats[3] = _w;
    }

    __device__ __host__ btCudaQuaternion(const btCudaVector3& axis, const btCudaScalar& angle)
    {
        setRotation(axis, angle);
    }

    __device__ __host__ btCudaQuaternion(const btCudaScalar& yaw, const btCudaScalar& pitch, const btCudaScalar& roll)
    {
        setEuler(yaw, pitch, roll);
    }

    /**
     * Copy constructor
     */
    __device__ __host__ btCudaQuaternion(const btCudaQuaternion& q)
    {
        m_floats[0] = q.m_floats[0];
        m_floats[1] = q.m_floats[1];
        m_floats[2] = q.m_floats[2];
        m_floats[3] = q.m_floats[3];
    }

    /**
     * Element access
     */
    __device__ __host__ const btCudaScalar& x() const { return m_floats[0]; }
    __device__ __host__ const btCudaScalar& y() const { return m_floats[1]; }
    __device__ __host__ const btCudaScalar& z() const { return m_floats[2]; }
    __device__ __host__ const btCudaScalar& w() const { return m_floats[3]; }

    __device__ __host__ btCudaScalar& x() { return m_floats[0]; }
    __device__ __host__ btCudaScalar& y() { return m_floats[1]; }
    __device__ __host__ btCudaScalar& z() { return m_floats[2]; }
    __device__ __host__ btCudaScalar& w() { return m_floats[3]; }

    __device__ __host__ const btCudaScalar& operator[](int i) const { return m_floats[i]; }
    __device__ __host__ btCudaScalar& operator[](int i) { return m_floats[i]; }

    /**
     * Assignment operator
     */
    __device__ __host__ btCudaQuaternion& operator=(const btCudaQuaternion& q)
    {
        m_floats[0] = q.m_floats[0];
        m_floats[1] = q.m_floats[1];
        m_floats[2] = q.m_floats[2];
        m_floats[3] = q.m_floats[3];
        return *this;
    }

    /**
     * Set quaternion from axis-angle - GPU optimized
     */
    __device__ __host__ void setRotation(const btCudaVector3& axis, const btCudaScalar& angle)
    {
        btCudaScalar d = axis.length();
        if (d == btCudaScalar(0.0)) {
            setIdentity();
            return;
        }

        btCudaScalar s = btCudaSin(angle * btCudaScalar(0.5)) / d;
        setValue(axis.getX() * s, axis.getY() * s, axis.getZ() * s, btCudaCos(angle * btCudaScalar(0.5)));
    }

    /**
     * Set quaternion from Euler angles - GPU optimized
     */
    __device__ __host__ void setEuler(const btCudaScalar& yaw, const btCudaScalar& pitch, const btCudaScalar& roll)
    {
        btCudaScalar halfYaw = yaw * btCudaScalar(0.5);
        btCudaScalar halfPitch = pitch * btCudaScalar(0.5);
        btCudaScalar halfRoll = roll * btCudaScalar(0.5);
        btCudaScalar cosYaw = btCudaCos(halfYaw);
        btCudaScalar sinYaw = btCudaSin(halfYaw);
        btCudaScalar cosPitch = btCudaCos(halfPitch);
        btCudaScalar sinPitch = btCudaSin(halfPitch);
        btCudaScalar cosRoll = btCudaCos(halfRoll);
        btCudaScalar sinRoll = btCudaSin(halfRoll);
        setValue(cosRoll * sinPitch * cosYaw + sinRoll * cosPitch * sinYaw,
                cosRoll * cosPitch * sinYaw - sinRoll * sinPitch * cosYaw,
                sinRoll * cosPitch * cosYaw - cosRoll * sinPitch * sinYaw,
                cosRoll * cosPitch * cosYaw + sinRoll * sinPitch * sinYaw);
    }

    /**
     * Set quaternion values
     */
    __device__ __host__ void setValue(const btCudaScalar& _x, const btCudaScalar& _y, const btCudaScalar& _z, const btCudaScalar& _w)
    {
        m_floats[0] = _x;
        m_floats[1] = _y;
        m_floats[2] = _z;
        m_floats[3] = _w;
    }

    /**
     * Set as identity quaternion
     */
    __device__ __host__ void setIdentity()
    {
        setValue(btCudaScalar(0.0), btCudaScalar(0.0), btCudaScalar(0.0), btCudaScalar(1.0));
    }

    /**
     * Quaternion arithmetic - GPU optimized
     */
    __device__ __host__ btCudaQuaternion operator+(const btCudaQuaternion& q2) const
    {
        return btCudaQuaternion(m_floats[0] + q2.m_floats[0], m_floats[1] + q2.m_floats[1], m_floats[2] + q2.m_floats[2], m_floats[3] + q2.m_floats[3]);
    }

    __device__ __host__ btCudaQuaternion operator-(const btCudaQuaternion& q2) const
    {
        return btCudaQuaternion(m_floats[0] - q2.m_floats[0], m_floats[1] - q2.m_floats[1], m_floats[2] - q2.m_floats[2], m_floats[3] - q2.m_floats[3]);
    }

    __device__ __host__ btCudaQuaternion operator*(const btCudaScalar& s) const
    {
        return btCudaQuaternion(m_floats[0] * s, m_floats[1] * s, m_floats[2] * s, m_floats[3] * s);
    }

    /**
     * Quaternion multiplication (rotation composition) - GPU optimized
     */
    __device__ __host__ btCudaQuaternion operator*(const btCudaQuaternion& q) const
    {
        return btCudaQuaternion(
            m_floats[3] * q.m_floats[0] + m_floats[0] * q.m_floats[3] + m_floats[1] * q.m_floats[2] - m_floats[2] * q.m_floats[1],
            m_floats[3] * q.m_floats[1] + m_floats[1] * q.m_floats[3] + m_floats[2] * q.m_floats[0] - m_floats[0] * q.m_floats[2],
            m_floats[3] * q.m_floats[2] + m_floats[2] * q.m_floats[3] + m_floats[0] * q.m_floats[1] - m_floats[1] * q.m_floats[0],
            m_floats[3] * q.m_floats[3] - m_floats[0] * q.m_floats[0] - m_floats[1] * q.m_floats[1] - m_floats[2] * q.m_floats[2]
        );
    }

    /**
     * In-place operators
     */
    __device__ __host__ btCudaQuaternion& operator+=(const btCudaQuaternion& q)
    {
        m_floats[0] += q.m_floats[0];
        m_floats[1] += q.m_floats[1];
        m_floats[2] += q.m_floats[2];
        m_floats[3] += q.m_floats[3];
        return *this;
    }

    __device__ __host__ btCudaQuaternion& operator-=(const btCudaQuaternion& q)
    {
        m_floats[0] -= q.m_floats[0];
        m_floats[1] -= q.m_floats[1];
        m_floats[2] -= q.m_floats[2];
        m_floats[3] -= q.m_floats[3];
        return *this;
    }

    __device__ __host__ btCudaQuaternion& operator*=(const btCudaScalar& s)
    {
        m_floats[0] *= s;
        m_floats[1] *= s;
        m_floats[2] *= s;
        m_floats[3] *= s;
        return *this;
    }

    __device__ __host__ btCudaQuaternion& operator*=(const btCudaQuaternion& q)
    {
        setValue(
            m_floats[3] * q.m_floats[0] + m_floats[0] * q.m_floats[3] + m_floats[1] * q.m_floats[2] - m_floats[2] * q.m_floats[1],
            m_floats[3] * q.m_floats[1] + m_floats[1] * q.m_floats[3] + m_floats[2] * q.m_floats[0] - m_floats[0] * q.m_floats[2],
            m_floats[3] * q.m_floats[2] + m_floats[2] * q.m_floats[3] + m_floats[0] * q.m_floats[1] - m_floats[1] * q.m_floats[0],
            m_floats[3] * q.m_floats[3] - m_floats[0] * q.m_floats[0] - m_floats[1] * q.m_floats[1] - m_floats[2] * q.m_floats[2]
        );
        return *this;
    }

    /**
     * Dot product
     */
    __device__ __host__ btCudaScalar dot(const btCudaQuaternion& q) const
    {
        return m_floats[0] * q.m_floats[0] + m_floats[1] * q.m_floats[1] + m_floats[2] * q.m_floats[2] + m_floats[3] * q.m_floats[3];
    }

    /**
     * Length calculations - GPU optimized
     */
    __device__ __host__ btCudaScalar length2() const
    {
        return dot(*this);
    }

    __device__ __host__ btCudaScalar length() const
    {
        return btCudaSqrt(length2());
    }

    /**
     * Normalize quaternion - GPU optimized
     */
    __device__ __host__ btCudaQuaternion& normalize()
    {
        btCudaScalar len = length();
        if (len > CUDA_EPSILON) {
            *this *= btCudaScalar(1.0) / len;
        } else {
            setIdentity();
        }
        return *this;
    }

    __device__ __host__ btCudaQuaternion normalized() const
    {
        btCudaQuaternion result = *this;
        result.normalize();
        return result;
    }

    /**
     * Quaternion conjugate (inverse rotation)
     */
    __device__ __host__ btCudaQuaternion inverse() const
    {
        return btCudaQuaternion(-m_floats[0], -m_floats[1], -m_floats[2], m_floats[3]);
    }

    /**
     * Rotate vector by quaternion - GPU optimized
     */
    __device__ __host__ btCudaVector3 operator*(const btCudaVector3& w) const
    {
        // v + 2.0 * cross(q.xyz, cross(q.xyz, v) + q.w * v)
        btCudaVector3 q_xyz(m_floats[0], m_floats[1], m_floats[2]);
        btCudaVector3 t = q_xyz.cross(w) + m_floats[3] * w;
        return w + btCudaScalar(2.0) * q_xyz.cross(t);
    }

    /**
     * Spherical linear interpolation - GPU optimized
     */
    __device__ __host__ btCudaQuaternion slerp(const btCudaQuaternion& q, const btCudaScalar& t) const
    {
        btCudaScalar theta = dot(q);
        
        if (theta < btCudaScalar(0.0)) {
            theta = -theta;
        }
        
        if (theta > btCudaScalar(0.9995)) {
            // Use linear interpolation for quaternions that are very close
            return ((*this) * (btCudaScalar(1.0) - t) + q * t).normalized();
        }
        
        btCudaScalar angle = btCudaAcos(theta);
        btCudaScalar sinAngle = btCudaSin(angle);
        btCudaScalar ta = btCudaSin((btCudaScalar(1.0) - t) * angle) / sinAngle;
        btCudaScalar tb = btCudaSin(t * angle) / sinAngle;
        
        return (*this) * ta + q * tb;
    }

    /**
     * Get rotation axis and angle
     */
    __device__ __host__ void getAxisAngle(btCudaVector3& axis, btCudaScalar& angle) const
    {
        btCudaScalar sqrLength = m_floats[0] * m_floats[0] + m_floats[1] * m_floats[1] + m_floats[2] * m_floats[2];
        if (sqrLength > CUDA_EPSILON) {
            angle = btCudaScalar(2.0) * btCudaAcos(m_floats[3]);
            btCudaScalar invLength = btCudaScalar(1.0) / btCudaSqrt(sqrLength);
            axis.setValue(m_floats[0] * invLength, m_floats[1] * invLength, m_floats[2] * invLength);
        } else {
            angle = btCudaScalar(0.0);
            axis.setValue(btCudaScalar(1.0), btCudaScalar(0.0), btCudaScalar(0.0));
        }
    }

    /**
     * Get Euler angles from quaternion
     */
    __device__ __host__ void getEulerZYX(btCudaScalar& yaw, btCudaScalar& pitch, btCudaScalar& roll) const
    {
        btCudaScalar test = m_floats[0] * m_floats[1] + m_floats[2] * m_floats[3];
        if (test > btCudaScalar(0.499)) { // singularity at north pole
            yaw = btCudaScalar(2.0) * btCudaAtan2(m_floats[0], m_floats[3]);
            pitch = CUDA_HALF_PI;
            roll = btCudaScalar(0);
            return;
        }
        if (test < btCudaScalar(-0.499)) { // singularity at south pole
            yaw = btCudaScalar(-2.0) * btCudaAtan2(m_floats[0], m_floats[3]);
            pitch = -CUDA_HALF_PI;
            roll = btCudaScalar(0);
            return;
        }
        btCudaScalar sqx = m_floats[0] * m_floats[0];
        btCudaScalar sqy = m_floats[1] * m_floats[1];
        btCudaScalar sqz = m_floats[2] * m_floats[2];
        yaw = btCudaAtan2(btCudaScalar(2.0) * m_floats[1] * m_floats[3] - btCudaScalar(2.0) * m_floats[0] * m_floats[2], btCudaScalar(1.0) - btCudaScalar(2.0) * sqy - btCudaScalar(2.0) * sqz);
        pitch = btCudaAsin(btCudaScalar(2.0) * test);
        roll = btCudaAtan2(btCudaScalar(2.0) * m_floats[0] * m_floats[3] - btCudaScalar(2.0) * m_floats[1] * m_floats[2], btCudaScalar(1.0) - btCudaScalar(2.0) * sqx - btCudaScalar(2.0) * sqz);
    }

    /**
     * Static utility functions
     */
    __device__ __host__ static btCudaQuaternion getIdentity()
    {
        return btCudaQuaternion(btCudaScalar(0.0), btCudaScalar(0.0), btCudaScalar(0.0), btCudaScalar(1.0));
    }
};

/**
 * Global operators
 */
__device__ __host__ inline btCudaQuaternion operator*(const btCudaScalar& s, const btCudaQuaternion& q)
{
    return q * s;
}

__device__ __host__ inline btCudaScalar btCudaDot(const btCudaQuaternion& q1, const btCudaQuaternion& q2)
{
    return q1.dot(q2);
}

__device__ __host__ inline btCudaQuaternion btCudaSlerp(const btCudaQuaternion& q1, const btCudaQuaternion& q2, const btCudaScalar& t)
{
    return q1.slerp(q2, t);
}

#endif // BT_CUDA_QUATERNION_CUH
