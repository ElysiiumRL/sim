/*
CUDA Conversion of Bullet Physics btCollisionShape
Copyright (c) 2003-2006 Erwin Coumans  https://bulletphysics.org
CUDA Conversion: 2025

This software is provided 'as-is', without any express or implied warranty.
*/

#ifndef BT_CUDA_COLLISION_SHAPE_CUH
#define BT_CUDA_COLLISION_SHAPE_CUH

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "../../LinearMath/btCudaTransform.cuh"
#include "../../LinearMath/btCudaVector3.cuh"

// Shape types for CUDA collision shapes
enum btCudaShapeType {
    CUDA_BOX_SHAPE_PROXYTYPE = 0,
    CUDA_TRIANGLE_SHAPE_PROXYTYPE,
    CUDA_TETRAHEDRAL_SHAPE_PROXYTYPE,
    CUDA_CONVEX_TRIANGLEMESH_SHAPE_PROXYTYPE,
    CUDA_CONVEX_HULL_SHAPE_PROXYTYPE,
    CUDA_CONVEX_POINT_CLOUD_SHAPE_PROXYTYPE,
    CUDA_CUSTOM_POLYHEDRAL_SHAPE_TYPE,
    CUDA_IMPLICIT_CONVEX_SHAPES_START_HERE,
    CUDA_SPHERE_SHAPE_PROXYTYPE,
    CUDA_MULTI_SPHERE_SHAPE_PROXYTYPE,
    CUDA_CAPSULE_SHAPE_PROXYTYPE,
    CUDA_CONE_SHAPE_PROXYTYPE,
    CUDA_CONVEX_SHAPE_PROXYTYPE,
    CUDA_CYLINDER_SHAPE_PROXYTYPE,
    CUDA_UNIFORM_SCALING_SHAPE_PROXYTYPE,
    CUDA_MINKOWSKI_SUM_SHAPE_PROXYTYPE,
    CUDA_MINKOWSKI_DIFFERENCE_SHAPE_PROXYTYPE,
    CUDA_BOX_2D_SHAPE_PROXYTYPE,
    CUDA_CONVEX_2D_SHAPE_PROXYTYPE,
    CUDA_CUSTOM_CONVEX_SHAPE_TYPE,
    CUDA_CONCAVE_SHAPES_START_HERE,
    CUDA_TRIANGLE_MESH_SHAPE_PROXYTYPE,
    CUDA_SCALED_TRIANGLE_MESH_SHAPE_PROXYTYPE,
    CUDA_FAST_CONCAVE_MESH_PROXYTYPE,
    CUDA_TERRAIN_SHAPE_PROXYTYPE,
    CUDA_GIMPACT_SHAPE_PROXYTYPE,
    CUDA_MULTIMATERIAL_TRIANGLE_MESH_PROXYTYPE,
    CUDA_EMPTY_SHAPE_PROXYTYPE,
    CUDA_STATIC_PLANE_PROXYTYPE,
    CUDA_CUSTOM_CONCAVE_SHAPE_TYPE,
    CUDA_COMPOUND_SHAPE_PROXYTYPE,
    CUDA_SOFTBODY_SHAPE_PROXYTYPE,
    CUDA_HFFLUID_SHAPE_PROXYTYPE,
    CUDA_HFFLUID_BUOYANT_CONVEX_SHAPE_PROXYTYPE,
    CUDA_INVALID_SHAPE_PROXYTYPE,
    CUDA_MAX_BROADPHASE_COLLISION_TYPES
};

/**
 * Base CUDA Collision Shape class
 * All collision shapes inherit from this
 */
struct btCudaCollisionShape
{
    btCudaShapeType m_shapeType;
    btCudaScalar m_margin;
    int m_userIndex;
    
    /**
     * Initialize base shape
     */
    __device__ __host__ void init(btCudaShapeType shapeType)
    {
        m_shapeType = shapeType;
        m_margin = btCudaScalar(0.04);  // Default margin
        m_userIndex = -1;
    }
    
    /**
     * Virtual functions that need to be implemented by derived classes
     */
    __device__ virtual void getAabb(const btCudaTransform& t, btCudaVector3& aabbMin, btCudaVector3& aabbMax) const = 0;
    __device__ virtual btCudaVector3 localGetSupportingVertex(const btCudaVector3& vec) const = 0;
    __device__ virtual btCudaVector3 localGetSupportingVertexWithoutMargin(const btCudaVector3& vec) const = 0;
    __device__ virtual void batchedUnitVectorGetSupportingVertexWithoutMargin(const btCudaVector3* vectors, btCudaVector3* supportVerticesOut, int numVectors) const = 0;
    __device__ virtual btCudaScalar getMargin() const { return m_margin; }
    __device__ virtual void setMargin(btCudaScalar margin) { m_margin = margin; }
    __device__ virtual btCudaShapeType getShapeType() const { return m_shapeType; }
    __device__ virtual bool isPolyhedral() const { return false; }
    __device__ virtual bool isConvex2d() const { return false; }
    __device__ virtual bool isConvex() const { return false; }
    __device__ virtual bool isNonMoving() const { return false; }
    __device__ virtual bool isConcave() const { return false; }
    __device__ virtual bool isCompound() const { return false; }
    __device__ virtual bool isSoftBody() const { return false; }
    __device__ virtual bool isInfinite() const { return false; }
    
    /**
     * Get bounding sphere
     */
    __device__ virtual void getBoundingSphere(btCudaVector3& center, btCudaScalar& radius) const
    {
        btCudaTransform tr;
        tr.setIdentity();
        btCudaVector3 aabbMin, aabbMax;
        getAabb(tr, aabbMin, aabbMax);
        
        radius = (aabbMax - aabbMin).length() * btCudaScalar(0.5);
        center = (aabbMin + aabbMax) * btCudaScalar(0.5);
    }
    
    /**
     * Get angular motion disc (for collision detection optimization)
     */
    __device__ virtual btCudaScalar getAngularMotionDisc() const
    {
        btCudaVector3 center;
        btCudaScalar disc;
        getBoundingSphere(center, disc);
        disc += (center).length();
        return disc;
    }
    
    /**
     * Get contact breaking threshold
     */
    __device__ virtual btCudaScalar getContactBreakingThreshold(btCudaScalar defaultContactThreshold) const
    {
        return getAngularMotionDisc() * defaultContactThreshold;
    }
};

/**
 * CUDA Box Shape
 */
struct btCudaBoxShape : public btCudaCollisionShape
{
    btCudaVector3 m_implicitShapeDimensions;
    
    __device__ __host__ void init(const btCudaVector3& boxHalfExtents)
    {
        btCudaCollisionShape::init(CUDA_BOX_SHAPE_PROXYTYPE);
        setSafeMargin(boxHalfExtents);
        
        btCudaVector3 margin(getMargin(), getMargin(), getMargin());
        m_implicitShapeDimensions = (boxHalfExtents * btCudaScalar(0.5)) - margin;
    }
    
    __device__ void setSafeMargin(const btCudaVector3& halfExtents, btCudaScalar minDimension = btCudaScalar(0.01))
    {
        btCudaScalar minDim = btCudaMin(btCudaMin(halfExtents[0], halfExtents[1]), halfExtents[2]);
        setMargin(btCudaMin(minDim, minDimension));
    }
    
    __device__ virtual void getAabb(const btCudaTransform& t, btCudaVector3& aabbMin, btCudaVector3& aabbMax) const override
    {
        btCudaVector3 halfExtents = getHalfExtentsWithMargin();
        btCudaMatrix3x3 abs_b = t.getBasis().absolute();
        btCudaVector3 center = t.getOrigin();
        btCudaVector3 extent = btCudaVector3(abs_b[0].dot(halfExtents), abs_b[1].dot(halfExtents), abs_b[2].dot(halfExtents));
        
        aabbMin = center - extent;
        aabbMax = center + extent;
    }
    
    __device__ virtual btCudaVector3 localGetSupportingVertex(const btCudaVector3& vec) const override
    {
        btCudaVector3 halfExtents = getHalfExtentsWithoutMargin();
        btCudaVector3 margin(getMargin(), getMargin(), getMargin());
        halfExtents += margin;
        
        return btCudaVector3(
            vec.getX() < btCudaScalar(0.0) ? -halfExtents.getX() : halfExtents.getX(),
            vec.getY() < btCudaScalar(0.0) ? -halfExtents.getY() : halfExtents.getY(),
            vec.getZ() < btCudaScalar(0.0) ? -halfExtents.getZ() : halfExtents.getZ()
        );
    }
    
    __device__ virtual btCudaVector3 localGetSupportingVertexWithoutMargin(const btCudaVector3& vec) const override
    {
        btCudaVector3 halfExtents = getHalfExtentsWithoutMargin();
        
        return btCudaVector3(
            vec.getX() < btCudaScalar(0.0) ? -halfExtents.getX() : halfExtents.getX(),
            vec.getY() < btCudaScalar(0.0) ? -halfExtents.getY() : halfExtents.getY(),
            vec.getZ() < btCudaScalar(0.0) ? -halfExtents.getZ() : halfExtents.getZ()
        );
    }
    
    __device__ virtual void batchedUnitVectorGetSupportingVertexWithoutMargin(const btCudaVector3* vectors, btCudaVector3* supportVerticesOut, int numVectors) const override
    {
        btCudaVector3 halfExtents = getHalfExtentsWithoutMargin();
        
        for (int i = 0; i < numVectors; i++) {
            const btCudaVector3& vec = vectors[i];
            supportVerticesOut[i] = btCudaVector3(
                vec.getX() < btCudaScalar(0.0) ? -halfExtents.getX() : halfExtents.getX(),
                vec.getY() < btCudaScalar(0.0) ? -halfExtents.getY() : halfExtents.getY(),
                vec.getZ() < btCudaScalar(0.0) ? -halfExtents.getZ() : halfExtents.getZ()
            );
        }
    }
    
    __device__ btCudaVector3 getHalfExtentsWithMargin() const
    {
        btCudaVector3 halfExtents = getHalfExtentsWithoutMargin();
        btCudaVector3 margin(getMargin(), getMargin(), getMargin());
        halfExtents += margin;
        return halfExtents;
    }
    
    __device__ const btCudaVector3& getHalfExtentsWithoutMargin() const
    {
        return m_implicitShapeDimensions;
    }
    
    __device__ virtual bool isPolyhedral() const override { return true; }
    __device__ virtual bool isConvex() const override { return true; }
};

/**
 * CUDA Sphere Shape
 */
struct btCudaSphereShape : public btCudaCollisionShape
{
    btCudaScalar m_radius;
    
    __device__ __host__ void init(btCudaScalar radius)
    {
        btCudaCollisionShape::init(CUDA_SPHERE_SHAPE_PROXYTYPE);
        m_radius = radius;
        setMargin(radius);
    }
    
    __device__ virtual void getAabb(const btCudaTransform& t, btCudaVector3& aabbMin, btCudaVector3& aabbMax) const override
    {
        btCudaVector3 center = t.getOrigin();
        btCudaVector3 extent(getMargin(), getMargin(), getMargin());
        
        aabbMin = center - extent;
        aabbMax = center + extent;
    }
    
    __device__ virtual btCudaVector3 localGetSupportingVertex(const btCudaVector3& vec) const override
    {
        btCudaVector3 supportVertex = localGetSupportingVertexWithoutMargin(vec);
        btCudaVector3 vecnorm = vec;
        if (vecnorm.length2() < (CUDA_EPSILON * CUDA_EPSILON)) {
            vecnorm.setValue(btCudaScalar(-1.0), btCudaScalar(-1.0), btCudaScalar(-1.0));
        }
        vecnorm.normalize();
        supportVertex += getMargin() * vecnorm;
        return supportVertex;
    }
    
    __device__ virtual btCudaVector3 localGetSupportingVertexWithoutMargin(const btCudaVector3& vec) const override
    {
        return btCudaVector3(btCudaScalar(0.0), btCudaScalar(0.0), btCudaScalar(0.0));
    }
    
    __device__ virtual void batchedUnitVectorGetSupportingVertexWithoutMargin(const btCudaVector3* vectors, btCudaVector3* supportVerticesOut, int numVectors) const override
    {
        for (int i = 0; i < numVectors; i++) {
            supportVerticesOut[i] = btCudaVector3(btCudaScalar(0.0), btCudaScalar(0.0), btCudaScalar(0.0));
        }
    }
    
    __device__ btCudaScalar getRadius() const { return m_radius; }
    __device__ void setUnscaledRadius(btCudaScalar radius) { m_radius = radius; setMargin(radius); }
    
    __device__ virtual bool isConvex() const override { return true; }
    __device__ virtual void setMargin(btCudaScalar margin) override { m_margin = margin; }
};

#endif // BT_CUDA_COLLISION_SHAPE_CUH
