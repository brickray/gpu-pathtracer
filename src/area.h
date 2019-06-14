#ifndef H_AREA_H
#define H_AREA_H

#include "common.h"
#include "mesh.h"

class Area{
public:
	float3 radiance;
	Triangle triangle;
	int medium;

public:
	__host__ __device__ void SampleLight(float3& pos, float2& u, float3& rad, Ray& ray, float3& nor, float& pdf, float epsilon = 0.01) const{
		float3 dir;
		triangle.SampleShape(pos, u, dir, nor, pdf);
		rad = pdf != 0.f ? radiance : make_float3(0.f, 0.f, 0.f);
		ray = Ray(pos, normalize(dir), nullptr, epsilon, sqrtf(dot(dir, dir) - epsilon));
	}

	__host__ __device__ void SampleLight(float4& u, Ray& ray, float3& nor, float3& rad, float& pdfA, float& pdfW, float epsilon = 0.01){
		float3 pos, dir;
		triangle.SampleShape(u, pos, dir, nor, pdfA, pdfW);
		rad = radiance;
		ray = Ray(pos, dir, nullptr, epsilon);
	}

	__host__ __device__ void Pdf(Ray& ray, float3& nor, float& pdfA, float& pdfW) const{
		float sa = triangle.GetSurfaceArea();
		pdfA = 1.f / sa;
		pdfW = fabs(dot(ray.d, nor))*ONE_OVER_PI;
	}

	__host__ __device__ float3 GetPower() const{
		return radiance * triangle.GetSurfaceArea() * PI;
	}

	__host__ __device__ float3 Le(float3& nor, float3& dir) const{
		if (dot(nor, dir) > 0.f) return radiance;
		else return make_float3(0.f, 0.f, 0.f);
	}
};

#endif