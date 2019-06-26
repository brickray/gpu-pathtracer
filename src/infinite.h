#ifndef H_INFINITE_H
#define H_INFINITE_H

#include "common.h"

class Infinite{
public:
	float3* data;
	int width, height;
	float3 center;
	float radius;
	float3 u, v, w;
	bool isvalid = false;

public:
	//TODO importance
	__host__ __device__ void SampleLight(float3& pos, float2& uniform, float3& rad, Ray& ray, float3& nor, float& pdf, float epsilon = 0.01) const{
		float pdfW;
		float3 dir = UniformSphere(uniform.x, uniform.y, pdfW);
		//dir = dir.x*u + dir.y*v + dir.z*w;
		//float3 p = pos + dir*radius;
		float costheta = dot(dir, v);
		float theta = acos(costheta);
		float3 d = normalize(dir - costheta*v);
		float cosphi = dot(d, u);
		float phi = acos(cosphi);
		float c = dot(d, w);
		phi = c > 0 ? TWOPI - phi : phi;
		float uu = phi / TWOPI;
		float vv = theta / PI;

		nor = -dir;
		ray = Ray(pos, dir, nullptr, epsilon, 2.f*radius - epsilon);
		pdf = pdfW;
		rad = getTexelBilinear(make_float2(1.f - uu, vv));
	}

	__host__ __device__ void Pdf(Ray& ray, float3& nor, float& pdfA, float& pdfW) const{
		pdfW = ONE_OVER_FOUR_PI;
		pdfA = 1.f / (PI*radius*radius);
	}

	__host__ __device__ float3 GetPower() const{
		return FOURPI*radius*radius*data[0];
	}

	__host__ __device__ float3 Le(float3& dir) const{
		float costheta = dot(dir, v);
		float theta = acos(costheta);
		float3 d = normalize(dir - costheta*v);
		float cosphi = dot(d, u);
		float phi = acos(cosphi);
		float c = dot(d, w);
		phi = c > 0 ? TWOPI - phi : phi;
		float uu = phi / TWOPI;
		float vv = theta / PI;

		return getTexelBilinear(make_float2(1.f - uu, vv));
	}

	__host__ __device__ void Init(BBox& bbox){
		bbox.boundingSphere(center, radius);
	}

private:
	__host__ __device__ float3 getTexelBilinear(float2 uv) const{
		float xx = width * uv.x;
		float yy = height * uv.y;
		int x = floor(xx);
		int y = floor(yy);
		float dx = fabs(xx - x);
		float dy = fabs(yy - y);
		float3 c00 = getTexel(make_int2(x, y));
		float3 c10 = getTexel(make_int2(x + 1, y));
		float3 c01 = getTexel(make_int2(x, y + 1));
		float3 c11 = getTexel(make_int2(x + 1, y + 1));
		return (1 - dy)*((1 - dx)*c00 + dx*c10)
			+ dy*((1 - dx)*c01 + dx*c11);
	}

	__host__ __device__ float3 getTexel(int2 uv) const{
		int x = uv.x, y = uv.y;
		float rx = x - (x / width)*width;
		float ry = y - (y / height)*height;
		x = (rx < 0) ? rx + width : rx;
		y = (ry < 0) ? ry + height : ry;
		if (x < 0) x = 0;
		if (x > width - 1) x = width - 1;
		if (y < 0) y = 0;
		if (y > height - 1) y = height - 1;

		float3 c = data[y*width + x];
		return c;
	}
};

#endif