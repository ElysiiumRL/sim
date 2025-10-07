/*
CUDA Conversion of Bullet Physics btVector3
Copyright (c) 2003-2006 Gino van den Bergen / Erwin Coumans  https://bulletphysics.org
CUDA Conversion: 2025

This software is provided 'as-is', without any express or implied warranty.
In no event will the authors be held liable for any damages arising from the use of this software.
Permission is granted to anyone to use this software for any purpose, 
including commercial applications, and to alter it and redistribute it freely, 
subject to the following restrictions:

1. The origin of this software must not be misrepresented; you must not claim that you wrote the original software.
2. Altered source versions must be plainly marked as such, and must not be misrepresented as being the original software.
3. This notice may not be removed or altered from any source distribution.
*/

#ifndef BT_CUDA_VECTOR3_CUH
#define BT_CUDA_VECTOR3_CUH

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "btCudaScalar.cuh"
#include "btCudaMinMax.cuh"

/**
 * CUDA-accelerated 3D Vector class
 * All operations marked with __device__ __host__ for GPU/CPU compatibility
 */
class btCudaVector3
{
public:
    union {
        struct { btCudaScalar m_floats[4]; };
        struct { btCudaScalar x, y, z, w; };
    };

    /**
     * Constructors - GPU/CPU compatible
     */
    __device__ __host__ btCudaVector3() {}

    __device__ __host__ btCudaVector3(const btCudaScalar& _x, const btCudaScalar& _y, const btCudaScalar& _z)
    {
        m_floats[0] = _x;
        m_floats[1] = _y;
        m_floats[2] = _z;
        m_floats[3] = btCudaScalar(0.0);
    }

    __device__ __host__ btCudaVector3(const btCudaScalar& _x, const btCudaScalar& _y, const btCudaScalar& _z, const btCudaScalar& _w)
    {
        m_floats[0] = _x;
        m_floats[1] = _y;
        m_floats[2] = _z;
        m_floats[3] = _w;
    }

    /**
     * Copy constructor
     */
    __device__ __host__ btCudaVector3(const btCudaVector3& other)
    {
        m_floats[0] = other.m_floats[0];
        m_floats[1] = other.m_floats[1];
        m_floats[2] = other.m_floats[2];
        m_floats[3] = other.m_floats[3];
    }

    /**
     * Element access
     */
    __device__ __host__ const btCudaScalar& getX() const { return m_floats[0]; }
    __device__ __host__ const btCudaScalar& getY() const { return m_floats[1]; }
    __device__ __host__ const btCudaScalar& getZ() const { return m_floats[2]; }
    __device__ __host__ const btCudaScalar& getW() const { return m_floats[3]; }

    __device__ __host__ btCudaScalar& getX() { return m_floats[0]; }
    __device__ __host__ btCudaScalar& getY() { return m_floats[1]; }
    __device__ __host__ btCudaScalar& getZ() { return m_floats[2]; }
    __device__ __host__ btCudaScalar& getW() { return m_floats[3]; }

    __device__ __host__ void setX(btCudaScalar _x) { m_floats[0] = _x; }
    __device__ __host__ void setY(btCudaScalar _y) { m_floats[1] = _y; }
    __device__ __host__ void setZ(btCudaScalar _z) { m_floats[2] = _z; }
    __device__ __host__ void setW(btCudaScalar _w) { m_floats[3] = _w; }

    /**
     * Array access operators
     */
    __device__ __host__ const btCudaScalar& operator[](int i) const { return m_floats[i]; }
    __device__ __host__ btCudaScalar& operator[](int i) { return m_floats[i]; }

    /**
     * Assignment operator
     */
    __device__ __host__ btCudaVector3& operator=(const btCudaVector3& v)
    {
        m_floats[0] = v.m_floats[0];
        m_floats[1] = v.m_floats[1];
        m_floats[2] = v.m_floats[2];
        m_floats[3] = v.m_floats[3];
        return *this;
    }

    /**
     * Vector arithmetic operations - GPU optimized
     */
    __device__ __host__ btCudaVector3 operator+(const btCudaVector3& v) const
    {
        return btCudaVector3(
            m_floats[0] + v.m_floats[0],
            m_floats[1] + v.m_floats[1],
            m_floats[2] + v.m_floats[2]
        );
    }

    __device__ __host__ btCudaVector3 operator-(const btCudaVector3& v) const
    {
        return btCudaVector3(
            m_floats[0] - v.m_floats[0],
            m_floats[1] - v.m_floats[1],
            m_floats[2] - v.m_floats[2]
        );
    }

    __device__ __host__ btCudaVector3 operator*(const btCudaScalar& s) const
    {
        return btCudaVector3(m_floats[0] * s, m_floats[1] * s, m_floats[2] * s);
    }

    __device__ __host__ btCudaVector3 operator/(const btCudaScalar& s) const
    {
        btCudaScalar inv = btCudaScalar(1.0) / s;
        return btCudaVector3(m_floats[0] * inv, m_floats[1] * inv, m_floats[2] * inv);
    }

    /**
     * In-place operators
     */
    __device__ __host__ btCudaVector3& operator+=(const btCudaVector3& v)
    {
        m_floats[0] += v.m_floats[0];
        m_floats[1] += v.m_floats[1];
        m_floats[2] += v.m_floats[2];
        return *this;
    }

    __device__ __host__ btCudaVector3& operator-=(const btCudaVector3& v)
    {
        m_floats[0] -= v.m_floats[0];
        m_floats[1] -= v.m_floats[1];
        m_floats[2] -= v.m_floats[2];
        return *this;
    }

    __device__ __host__ btCudaVector3& operator*=(const btCudaScalar& s)
    {
        m_floats[0] *= s;
        m_floats[1] *= s;
        m_floats[2] *= s;
        return *this;
    }

    __device__ __host__ btCudaVector3& operator/=(const btCudaScalar& s)
    {
        btCudaScalar inv = btCudaScalar(1.0) / s;
        m_floats[0] *= inv;
        m_floats[1] *= inv;
        m_floats[2] *= inv;
        return *this;
    }

    /**
     * Dot product - GPU optimized
     */
    __device__ __host__ btCudaScalar dot(const btCudaVector3& v) const
    {
        return m_floats[0] * v.m_floats[0] + m_floats[1] * v.m_floats[1] + m_floats[2] * v.m_floats[2];
    }

    /**
     * Cross product - GPU optimized
     */
    __device__ __host__ btCudaVector3 cross(const btCudaVector3& v) const
    {
        return btCudaVector3(
            m_floats[1] * v.m_floats[2] - m_floats[2] * v.m_floats[1],
            m_floats[2] * v.m_floats[0] - m_floats[0] * v.m_floats[2],
            m_floats[0] * v.m_floats[1] - m_floats[1] * v.m_floats[0]
        );
    }

    /**
     * Length calculations - GPU optimized using fast math
     */
    __device__ __host__ btCudaScalar length2() const
    {
        return dot(*this);
    }

    __device__ __host__ btCudaScalar length() const
    {
#ifdef __CUDA_ARCH__
        return sqrtf(length2());  // Use fast sqrt on GPU
#else
        return sqrt(length2());
#endif
    }

    /**
     * Normalization - GPU optimized
     */
    __device__ __host__ btCudaVector3 normalized() const
    {
        btCudaScalar len = length();
        if (len > CUDA_EPSILON) {
            return *this / len;
        }
        return btCudaVector3(1, 0, 0);  // Return unit vector if original is zero
    }

    __device__ __host__ btCudaVector3& normalize()
    {
        btCudaScalar len = length();
        if (len > CUDA_EPSILON) {
            *this /= len;
        } else {
            setValue(1, 0, 0);
        }
        return *this;
    }

    /**
     * Distance calculations
     */
    __device__ __host__ btCudaScalar distance2(const btCudaVector3& v) const
    {
        return (v - *this).length2();
    }

    __device__ __host__ btCudaScalar distance(const btCudaVector3& v) const
    {
        return (v - *this).length();
    }

    /**
     * Set values
     */
    __device__ __host__ void setValue(const btCudaScalar& _x, const btCudaScalar& _y, const btCudaScalar& _z)
    {
        m_floats[0] = _x;
        m_floats[1] = _y;
        m_floats[2] = _z;
        m_floats[3] = btCudaScalar(0.0);
    }

    /**
     * Zero vector
     */
    __device__ __host__ void setZero()
    {
        setValue(btCudaScalar(0.0), btCudaScalar(0.0), btCudaScalar(0.0));
    }

    /**
     * Check if zero
     */
    __device__ __host__ bool isZero() const
    {
        return length2() < CUDA_EPSILON * CUDA_EPSILON;
    }

    /**
     * Linear interpolation - GPU optimized
     */
    __device__ __host__ btCudaVector3 lerp(const btCudaVector3& v, const btCudaScalar& t) const
    {
        return btCudaVector3(
            m_floats[0] + (v.m_floats[0] - m_floats[0]) * t,
            m_floats[1] + (v.m_floats[1] - m_floats[1]) * t,
            m_floats[2] + (v.m_floats[2] - m_floats[2]) * t
        );
    }

    /**
     * Component-wise operations
     */
    __device__ __host__ btCudaVector3 absolute() const
    {
        return btCudaVector3(
            fabsf(m_floats[0]),
            fabsf(m_floats[1]),
            fabsf(m_floats[2])
        );
    }

    __device__ __host__ int minAxis() const
    {
        return m_floats[0] < m_floats[1] ? (m_floats[0] < m_floats[2] ? 0 : 2) : (m_floats[1] < m_floats[2] ? 1 : 2);
    }

    __device__ __host__ int maxAxis() const
    {
        return m_floats[0] < m_floats[1] ? (m_floats[1] < m_floats[2] ? 2 : 1) : (m_floats[0] < m_floats[2] ? 2 : 0);
    }

    /**
     * Static utility functions
     */
    __device__ __host__ static btCudaVector3 zero()
    {
        return btCudaVector3(0, 0, 0);
    }

    __device__ __host__ static btCudaVector3 unitX()
    {
        return btCudaVector3(1, 0, 0);
    }

    __device__ __host__ static btCudaVector3 unitY()
    {
        return btCudaVector3(0, 1, 0);
    }

    __device__ __host__ static btCudaVector3 unitZ()
    {
        return btCudaVector3(0, 0, 1);
    }
};

/**
 * Global operators
 */
__device__ __host__ inline btCudaVector3 operator*(const btCudaScalar& s, const btCudaVector3& v)
{
    return v * s;
}

__device__ __host__ inline btCudaScalar btCudaDot(const btCudaVector3& v1, const btCudaVector3& v2)
{
    return v1.dot(v2);
}

__device__ __host__ inline btCudaVector3 btCudaCross(const btCudaVector3& v1, const btCudaVector3& v2)
{
    return v1.cross(v2);
}

__device__ __host__ inline btCudaVector3 btCudaLerp(const btCudaVector3& v1, const btCudaVector3& v2, const btCudaScalar& t)
{
    return v1.lerp(v2, t);
}

#endif // BT_CUDA_VECTOR3_CUH
