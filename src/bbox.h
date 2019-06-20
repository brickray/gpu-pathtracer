#ifndef H_BBOX_H
#define H_BBOX_H

#include "common.h"
#include "ray.h"

class BBox{
public:
	float3 fmin, fmax;

public:
	__host__ __device__ BBox(){
		fmin = make_float3(INFINITY, INFINITY, INFINITY);
		fmax = make_float3(-INFINITY, -INFINITY, -INFINITY);
	}

	__host__ __device__ BBox(float3& a, float3& b){
		fmin = a;
		fmax = b;
	}

	__host__ __device__ void Reset(){
		fmin = make_float3(INFINITY, INFINITY, INFINITY);
		fmax = make_float3(-INFINITY, -INFINITY, -INFINITY);
	}

	//expand from 2d case
	__host__ __device__ void Expand(BBox& b){
		fmin.x = fminf(b.fmin.x, fmin.x);
		fmin.y = fminf(b.fmin.y, fmin.y);
		fmin.z = fminf(b.fmin.z, fmin.z);

		fmax.x = fmaxf(b.fmax.x, fmax.x);
		fmax.y = fmaxf(b.fmax.y, fmax.y);
		fmax.z = fmaxf(b.fmax.z, fmax.z);
	}

	__host__ __device__ void Expand(float3& v){
		fmin.x = fminf(fmin.x, v.x);
		fmin.y = fminf(fmin.y, v.y);
		fmin.z = fminf(fmin.z, v.z);

		fmax.x = fmaxf(fmax.x, v.x);
		fmax.y = fmaxf(fmax.y, v.y);
		fmax.z = fmaxf(fmax.z, v.z);
	}

	__host__ __device__ float3 Centric() const{
		return (fmin + fmax)*0.5f;
	}

	__host__ __device__ float3 Diagonal() const{
		return fmax - fmin;
	}

	__host__ __device__ float3 Offset(float3 p) const{
		float3 diag = Diagonal();
		float3 delta = p - fmin;
		return delta / diag;
	}

	__host__ __device__ float SurfaceArea() const{
		float3 delta = fmax - fmin;
		return 2.f*(delta.x*delta.y + delta.y*delta.z + delta.z*delta.x);
	}

	__host__ __device__ int GetMaxExtent() const{
		float3 diag = Diagonal();
		if (diag.x > diag.y && diag.x > diag.z)
			return 0;
		else if (diag.y > diag.z)
			return 1;
		else
			return 2;
	}

	__host__ __device__ bool Intersect(Ray& r) const{
		//相交条件,光线进入平面的最大t值小于离开平面的最小t值
		float3 inv_dir = { 1.f / r.d.x, 1.f / r.d.y, 1.f / r.d.z };
		float t1 = (fmin.x - r.o.x)*inv_dir.x;
		float t2 = (fmax.x - r.o.x)*inv_dir.x;
		float t3 = (fmin.y - r.o.y)*inv_dir.y;
		float t4 = (fmax.y - r.o.y)*inv_dir.y;
		float t5 = (fmin.z - r.o.z)*inv_dir.z;
		float t6 = (fmax.z - r.o.z)*inv_dir.z;

		//若相交tmin tmax即为交点
		float tmin = fmaxf(fmaxf(fminf(t1, t2), fminf(t3, t4)), fminf(t5, t6));
		float tmax = fminf(fminf(fmaxf(t1, t2), fmaxf(t3, t4)), fmaxf(t5, t6));
		if (tmax <= 0.00001f) return false;//包围盒在射线后方
		if (tmin > tmax)
			return false;
		if (tmin > r.tmax)
			return false;
		return true;
	}

	__host__ __device__ void boundingSphere(float3& center, float& radius) const{
		center = Centric();
		radius = sqrtf(dot(fmax - center, fmax - center));
	}
};

#endif