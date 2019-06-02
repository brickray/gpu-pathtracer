#ifndef H_SPHERE_H
#define H_SPHERE_H

#include "common.h"
#include "bbox.h"
#include "intersection.h"

class Sphere{
public:
	float3 origin;
	float radius;
	int matIdx;

public:
	__host__ __device__ BBox GetBBox(){
		return BBox(float3(origin - make_float3(radius, radius, radius)),
			float3(origin + make_float3(radius, radius, radius)));
	}

	__host__ __device__ float GetSurfaceArea(){
		return FOURPI*radius*radius;
	}

	__host__ __device__ bool Intersect(Ray& ray, Intersection* isect){
		float3 op = ray.o - origin;

		//float A = dot(r.d, r.d); //r.d is normalized
		float B = dot(op, ray.d);
		float C = dot(op, op) - radius*radius;

		float delta = B*B - C;
		if (delta < 0.f)
			return false;

		float sqrDelta = sqrtf(delta);
		float t1 = -B - sqrDelta;
		float t2 = -B + sqrDelta;

		//shape behind ray
		if (t1 < 0.f && t2 < 0.f)
			return false;
		//- +  t1 = + t2 = -
		//+ + 
		if (t1 < 0.f || t2 < 0.f){
			float tt1 = t1, tt2 = t2;
			t1 = tt1 < 0.f ? tt2 : tt1;
			t2 = tt1 < 0.f ? tt1 : tt2;
		}
		else{
			float temp;
			if (t1>t2){
				temp = t2;
				t2 = t1;
				t1 = temp;
			}
		}

		if (t1 > ray.tmax)
			return false;

		//intersect
		if (t1 > ray.tmin)
			ray.tmax = t1;
		else if (t2 > 0.f)
			ray.tmax = t2;
		else
			return false;

		//set intersection
		if (isect){
			isect->pos = ray(ray.tmax);
			isect->nor = normalize(isect->pos - origin);
			float3 normal = isect->nor;
			//calc uv
			float costheta = dot(normal, make_float3(0.f, 1.f, 0.f));
			float theta = acosf(costheta);
			float v = acosf(costheta) * ONE_OVER_PI;
			float cosphi = dot(make_float3(1.f, 0.f, 0.f), make_float3(normal.x, 0.f, normal.z));
			float phi = acosf(cosphi);
			phi = normal.z > 0.f ? TWOPI - phi : phi;
			float u = phi * ONE_OVER_TWO_PI;
			isect->dpdu = normalize(make_float3(-TWOPI * isect->pos.y, TWOPI * isect->pos.x, 0));
			isect->uv = make_float2(u, v);
			isect->matIdx = matIdx;
			isect->lightIdx = -1;
		}

		return true;
	}
};

#endif