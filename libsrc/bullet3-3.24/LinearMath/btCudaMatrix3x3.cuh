/*
CUDA Conversion of Bullet Physics btMatrix3x3
Copyright (c) 2003-2006 Gino van den Bergen / Erwin Coumans  https://bulletphysics.org
CUDA Conversion: 2025

This software is provided 'as-is', without any express or implied warranty.
*/

#ifndef BT_CUDA_MATRIX3X3_CUH
#define BT_CUDA_MATRIX3X3_CUH

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "btCudaVector3.cuh"
#include "btCudaQuaternion.cuh"
#include "btCudaScalar.cuh"

/**
 * CUDA-accelerated 3x3 Matrix class
 * All operations optimized for GPU execution
 */
class btCudaMatrix3x3
{
private:
    btCudaVector3 m_el[3];  // Three row vectors

public:
    /**
     * Constructors
     */
    __device__ __host__ btCudaMatrix3x3() {}

    __device__ __host__ explicit btCudaMatrix3x3(const btCudaQuaternion& q) { setRotation(q); }

    __device__ __host__ btCudaMatrix3x3(const btCudaScalar& xx, const btCudaScalar& xy, const btCudaScalar& xz,
                                       const btCudaScalar& yx, const btCudaScalar& yy, const btCudaScalar& yz,
                                       const btCudaScalar& zx, const btCudaScalar& zy, const btCudaScalar& zz)
    {
        setValue(xx, xy, xz, yx, yy, yz, zx, zy, zz);
    }

    __device__ __host__ btCudaMatrix3x3(const btCudaMatrix3x3& other)
    {
        m_el[0] = other.m_el[0];
        m_el[1] = other.m_el[1];
        m_el[2] = other.m_el[2];
    }

    /**
     * Assignment operator
     */
    __device__ __host__ btCudaMatrix3x3& operator=(const btCudaMatrix3x3& other)
    {
        m_el[0] = other.m_el[0];
        m_el[1] = other.m_el[1];
        m_el[2] = other.m_el[2];
        return *this;
    }

    /**
     * Element access
     */
    __device__ __host__ btCudaVector3& operator[](int i) { return m_el[i]; }
    __device__ __host__ const btCudaVector3& operator[](int i) const { return m_el[i]; }

    __device__ __host__ btCudaVector3& getRow(int i) { return m_el[i]; }
    __device__ __host__ const btCudaVector3& getRow(int i) const { return m_el[i]; }

    __device__ __host__ btCudaVector3 getColumn(int i) const
    {
        return btCudaVector3(m_el[0][i], m_el[1][i], m_el[2][i]);
    }

    /**
     * Set matrix values
     */
    __device__ __host__ void setValue(const btCudaScalar& xx, const btCudaScalar& xy, const btCudaScalar& xz,
                                     const btCudaScalar& yx, const btCudaScalar& yy, const btCudaScalar& yz,
                                     const btCudaScalar& zx, const btCudaScalar& zy, const btCudaScalar& zz)
    {
        m_el[0].setValue(xx, xy, xz);
        m_el[1].setValue(yx, yy, yz);
        m_el[2].setValue(zx, zy, zz);
    }

    /**
     * Set as rotation matrix from quaternion - GPU optimized
     */
    __device__ __host__ void setRotation(const btCudaQuaternion& q)
    {
        btCudaScalar d = q.length2();
        btCudaScalar s = btCudaScalar(2.0) / d;
        
        btCudaScalar xs = q.x() * s, ys = q.y() * s, zs = q.z() * s;
        btCudaScalar wx = q.w() * xs, wy = q.w() * ys, wz = q.w() * zs;
        btCudaScalar xx = q.x() * xs, xy = q.x() * ys, xz = q.x() * zs;
        btCudaScalar yy = q.y() * ys, yz = q.y() * zs, zz = q.z() * zs;
        
        setValue(
            btCudaScalar(1.0) - (yy + zz), xy - wz, xz + wy,
            xy + wz, btCudaScalar(1.0) - (xx + zz), yz - wx,
            xz - wy, yz + wx, btCudaScalar(1.0) - (xx + yy)
        );
    }

    /**
     * Set as identity matrix
     */
    __device__ __host__ void setIdentity()
    {
        setValue(btCudaScalar(1.0), btCudaScalar(0.0), btCudaScalar(0.0),
                btCudaScalar(0.0), btCudaScalar(1.0), btCudaScalar(0.0),
                btCudaScalar(0.0), btCudaScalar(0.0), btCudaScalar(1.0));
    }

    /**
     * Static identity matrix
     */
    __device__ __host__ static const btCudaMatrix3x3& getIdentity()
    {
        static const btCudaMatrix3x3 identityMatrix(
            btCudaScalar(1.0), btCudaScalar(0.0), btCudaScalar(0.0),
            btCudaScalar(0.0), btCudaScalar(1.0), btCudaScalar(0.0),
            btCudaScalar(0.0), btCudaScalar(0.0), btCudaScalar(1.0)
        );
        return identityMatrix;
    }

    /**
     * Matrix multiplication - GPU optimized
     */
    __device__ __host__ btCudaMatrix3x3 operator*(const btCudaMatrix3x3& m) const
    {
        return btCudaMatrix3x3(
            m.tdotx(m_el[0]), m.tdoty(m_el[0]), m.tdotz(m_el[0]),
            m.tdotx(m_el[1]), m.tdoty(m_el[1]), m.tdotz(m_el[1]),
            m.tdotx(m_el[2]), m.tdoty(m_el[2]), m.tdotz(m_el[2])
        );
    }

    /**
     * Matrix addition
     */
    __device__ __host__ btCudaMatrix3x3 operator+(const btCudaMatrix3x3& m) const
    {
        return btCudaMatrix3x3(
            m_el[0][0] + m.m_el[0][0], m_el[0][1] + m.m_el[0][1], m_el[0][2] + m.m_el[0][2],
            m_el[1][0] + m.m_el[1][0], m_el[1][1] + m.m_el[1][1], m_el[1][2] + m.m_el[1][2],
            m_el[2][0] + m.m_el[2][0], m_el[2][1] + m.m_el[2][1], m_el[2][2] + m.m_el[2][2]
        );
    }

    /**
     * Matrix subtraction
     */
    __device__ __host__ btCudaMatrix3x3 operator-(const btCudaMatrix3x3& m) const
    {
        return btCudaMatrix3x3(
            m_el[0][0] - m.m_el[0][0], m_el[0][1] - m.m_el[0][1], m_el[0][2] - m.m_el[0][2],
            m_el[1][0] - m.m_el[1][0], m_el[1][1] - m.m_el[1][1], m_el[1][2] - m.m_el[1][2],
            m_el[2][0] - m.m_el[2][0], m_el[2][1] - m.m_el[2][1], m_el[2][2] - m.m_el[2][2]
        );
    }

    /**
     * Scalar multiplication
     */
    __device__ __host__ btCudaMatrix3x3 operator*(const btCudaScalar& s) const
    {
        return btCudaMatrix3x3(
            m_el[0][0] * s, m_el[0][1] * s, m_el[0][2] * s,
            m_el[1][0] * s, m_el[1][1] * s, m_el[1][2] * s,
            m_el[2][0] * s, m_el[2][1] * s, m_el[2][2] * s
        );
    }

    /**
     * Matrix-vector multiplication - GPU optimized
     */
    __device__ __host__ btCudaVector3 operator*(const btCudaVector3& v) const
    {
        return btCudaVector3(
            m_el[0].dot(v),
            m_el[1].dot(v),
            m_el[2].dot(v)
        );
    }

    /**
     * Transpose - GPU optimized
     */
    __device__ __host__ btCudaMatrix3x3 transpose() const
    {
        return btCudaMatrix3x3(
            m_el[0][0], m_el[1][0], m_el[2][0],
            m_el[0][1], m_el[1][1], m_el[2][1],
            m_el[0][2], m_el[1][2], m_el[2][2]
        );
    }

    /**
     * Determinant calculation - GPU optimized
     */
    __device__ __host__ btCudaScalar determinant() const
    {
        return m_el[0][0] * (m_el[1][1] * m_el[2][2] - m_el[1][2] * m_el[2][1]) -
               m_el[0][1] * (m_el[1][0] * m_el[2][2] - m_el[1][2] * m_el[2][0]) +
               m_el[0][2] * (m_el[1][0] * m_el[2][1] - m_el[1][1] * m_el[2][0]);
    }

    /**
     * Matrix inverse - GPU optimized
     */
    __device__ __host__ btCudaMatrix3x3 inverse() const
    {
        btCudaScalar det = determinant();
        if (btCudaFabs(det) < CUDA_EPSILON) {
            return getIdentity();  // Return identity if not invertible
        }
        
        btCudaScalar invDet = btCudaScalar(1.0) / det;
        
        return btCudaMatrix3x3(
            (m_el[1][1] * m_el[2][2] - m_el[1][2] * m_el[2][1]) * invDet,
            -(m_el[0][1] * m_el[2][2] - m_el[0][2] * m_el[2][1]) * invDet,
            (m_el[0][1] * m_el[1][2] - m_el[0][2] * m_el[1][1]) * invDet,
            
            -(m_el[1][0] * m_el[2][2] - m_el[1][2] * m_el[2][0]) * invDet,
            (m_el[0][0] * m_el[2][2] - m_el[0][2] * m_el[2][0]) * invDet,
            -(m_el[0][0] * m_el[1][2] - m_el[0][2] * m_el[1][0]) * invDet,
            
            (m_el[1][0] * m_el[2][1] - m_el[1][1] * m_el[2][0]) * invDet,
            -(m_el[0][0] * m_el[2][1] - m_el[0][1] * m_el[2][0]) * invDet,
            (m_el[0][0] * m_el[1][1] - m_el[0][1] * m_el[1][0]) * invDet
        );
    }

    /**
     * Get quaternion from rotation matrix - GPU optimized
     */
    __device__ __host__ btCudaQuaternion getRotation() const
    {
        btCudaScalar trace = m_el[0][0] + m_el[1][1] + m_el[2][2];
        
        if (trace > btCudaScalar(0.0)) {
            btCudaScalar s = btCudaSqrt(trace + btCudaScalar(1.0));
            btCudaScalar w = s * btCudaScalar(0.5);
            s = btCudaScalar(0.5) / s;
            return btCudaQuaternion(
                (m_el[2][1] - m_el[1][2]) * s,
                (m_el[0][2] - m_el[2][0]) * s,
                (m_el[1][0] - m_el[0][1]) * s,
                w
            );
        } else {
            int i = m_el[0][0] < m_el[1][1] ? 
                   (m_el[1][1] < m_el[2][2] ? 2 : 1) : 
                   (m_el[0][0] < m_el[2][2] ? 2 : 0);
                   
            int j = (i + 1) % 3;
            int k = (i + 2) % 3;
            
            btCudaScalar s = btCudaSqrt(m_el[i][i] - m_el[j][j] - m_el[k][k] + btCudaScalar(1.0));
            btCudaScalar q[4];
            q[i] = s * btCudaScalar(0.5);
            s = btCudaScalar(0.5) / s;
            q[3] = (m_el[k][j] - m_el[j][k]) * s;
            q[j] = (m_el[j][i] + m_el[i][j]) * s;
            q[k] = (m_el[k][i] + m_el[i][k]) * s;
            
            return btCudaQuaternion(q[0], q[1], q[2], q[3]);
        }
    }

    /**
     * Utility functions for matrix multiplication
     */
    __device__ __host__ btCudaScalar tdotx(const btCudaVector3& v) const
    {
        return m_el[0][0] * v[0] + m_el[1][0] * v[1] + m_el[2][0] * v[2];
    }

    __device__ __host__ btCudaScalar tdoty(const btCudaVector3& v) const
    {
        return m_el[0][1] * v[0] + m_el[1][1] * v[1] + m_el[2][1] * v[2];
    }

    __device__ __host__ btCudaScalar tdotz(const btCudaVector3& v) const
    {
        return m_el[0][2] * v[0] + m_el[1][2] * v[1] + m_el[2][2] * v[2];
    }

    /**
     * Euler angle extraction - GPU optimized
     */
    __device__ __host__ void getEulerYPR(btCudaScalar& yaw, btCudaScalar& pitch, btCudaScalar& roll) const
    {
        // Extract yaw, pitch, roll from rotation matrix
        if (m_el[1][0] > btCudaScalar(0.998)) { // singularity at north pole
            yaw = btCudaAtan2(m_el[0][2], m_el[2][2]);
            pitch = CUDA_HALF_PI;
            roll = btCudaScalar(0);
            return;
        }
        if (m_el[1][0] < btCudaScalar(-0.998)) { // singularity at south pole
            yaw = btCudaAtan2(m_el[0][2], m_el[2][2]);
            pitch = -CUDA_HALF_PI;
            roll = btCudaScalar(0);
            return;
        }
        yaw = btCudaAtan2(-m_el[2][0], m_el[0][0]);
        pitch = btCudaAsin(m_el[1][0]);
        roll = btCudaAtan2(-m_el[1][2], m_el[1][1]);
    }
};

/**
 * Global operators
 */
__device__ __host__ inline btCudaMatrix3x3 operator*(const btCudaScalar& s, const btCudaMatrix3x3& m)
{
    return m * s;
}

#endif // BT_CUDA_MATRIX3X3_CUH
