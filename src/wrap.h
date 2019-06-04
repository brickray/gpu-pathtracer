#ifndef H_WRAP_H
#define H_WRAP_H

#include "common.h"

__host__ __device__ inline void MakeCoordinate(float3& n, float3& u, float3& w){
	if (std::abs(n.x) > std::abs(n.y)) {
		float invLen = 1.0f / std::sqrt(n.x * n.x + n.z * n.z);
		w = make_float3(n.z * invLen, 0.0f, -n.x * invLen);
	}
	else {
		float invLen = 1.0f / std::sqrt(n.y * n.y + n.z * n.z);
		w = make_float3(0.0f, n.z * invLen, -n.y * invLen);
	}
	u = cross(w, n);
}

__host__ __device__ inline float3 ToWorld(float3& dir, float3& u, float3& v, float3& w){
	return dir.x*u + dir.y*v + dir.z*w;
}

__host__ __device__ inline float3 ToLocal(float3& dir, float3& u, float3& v, float3& w){
	return make_float3(dot(dir, u), dot(dir, v), dot(dir, w));
}

__host__ __device__ inline float3 UniformSphere(float u1, float u2, float& pdf){
	float costheta = 1.f - 2.f*u1;
	float sintheta = sqrtf(1.f - costheta*costheta);
	float phi = TWOPI*u2;
	float cosphi = cosf(phi);
	float sinphi = sinf(phi);

	pdf = ONE_OVER_FOUR_PI;

	return make_float3(sintheta*cosphi, costheta, sintheta*sinphi);
}

__host__ __device__ inline float3 UniformHemiSphere(float u1, float u2, float3& n, float& pdf){
	float costheta = u1;
	float sintheta = sqrtf(1.f - costheta*costheta);
	float phi = TWOPI * u2;
	float cosphi = cosf(phi);
	float sinphi = sinf(phi);

	pdf = ONE_OVER_TWO_PI;

	float3 dir = make_float3(sintheta*cosphi, costheta, sintheta*sinphi);
	return dir;
}

__host__ __device__ inline float3 CosineHemiSphere(float u1, float u2, float3& n, float& pdf){
	float sintheta = sqrtf(u1);
	float costheta = sqrtf(1.f - u1);
	float phi = TWOPI*u2;
	float cosphi = cosf(phi);
	float sinphi = sinf(phi);

	pdf = costheta*ONE_OVER_PI;

	float3 dir = make_float3(sintheta*cosphi, costheta, sintheta*sinphi);
	return dir;
}


__host__ __device__ inline float3 UniformCone(float u1, float u2, float costhetamax, float3& n, float& pdf){
	float costheta = 1.f - u1*(1.f - costhetamax);
	float sintheta = sqrtf(1.f - costheta*costheta);
	float phi = TWOPI*u2;
	float cosphi = cosf(phi);
	float sinphi = sinf(phi);

	pdf = 1.f / (TWOPI *(1.f - costhetamax));

	float3 dir = make_float3(sintheta*cosphi, costheta, sintheta*sinphi);
	return dir;
}

__host__ __device__ inline float2 UniformDisk(float u1, float u2, float& pdf){
	float r = sqrtf(u1);
	float phi = TWOPI*u2;

	pdf = ONE_OVER_PI;

	return make_float2(r*cosf(phi), r*sinf(phi));
}

__host__ __device__ inline float2 ConcentricDisk(float u1, float u2, float& pdf){
	// Map uniform random numbers to $[-1,1]^2$
	float2 uOffset = 2.f * make_float2(u1, u2) - make_float2(1, 1);

	// Handle degeneracy at the origin
	if (uOffset.x == 0 && uOffset.y == 0) return make_float2(0, 0);

	// Apply concentric mapping to point
	float theta, r;
	if (std::abs(uOffset.x) > std::abs(uOffset.y)) {
		r = uOffset.x;
		theta = PI * 0.25f * (uOffset.y / uOffset.x);
	}
	else {
		r = uOffset.y;
		theta = PI * 0.5f - PI * 0.25f * (uOffset.x / uOffset.y);
	}

	pdf = ONE_OVER_PI;

	return r * make_float2(std::cos(theta), std::sin(theta));
}

__host__ __device__ inline float2 UniformTriangle(float u1, float u2){
	float su1 = sqrtf(u1);
	float u = 1.f - su1;
	float v = u2 * su1;
	return make_float2(u, v);
}

__host__ __device__ inline float EquiAngular(float u, float D, float thetaA, float thetaB){
	return D * tanf(u*(thetaB - thetaA) + thetaA);
}

__host__ __device__ inline float EquiAngularPdf(float t, float D, float thetaA, float thetaB){
	return D / ((thetaB - thetaA)*(t*t + D*D));
}

__host__ __device__ inline float2 GaussianDiskInfinity(float u1, float u2, float falloff){
	float r = sqrtf(log(u1) / -falloff);
	float theta = TWOPI*u2;

	return{ r*cos(theta), r*sin(theta) };
}

__host__ __device__ inline float GaussianDiskInfinityPdf(float x, float y, float falloff){
	return ONE_OVER_PI * falloff*exp(-falloff*(x*x + y*y));
}

__host__ __device__ inline float GaussianDiskInfinityPdf(float3& center, float3& sample, float3& n, float falloff){
	float3 d = sample - center;
	float3 projected = d - n*dot(d, n);
	return ONE_OVER_PI*falloff*exp(-falloff*dot(projected, projected));
}

__host__ __device__ inline float2 GaussianDisk(float u1, float u2, float falloff, float rmax){
	float r = sqrtf(log(1.0f - u1 * (1.0f - exp(-falloff * rmax * rmax))) /
		-falloff);
	float theta = TWOPI * u2;
	return{ r * cos(theta), r * sin(theta) };
}

__host__ __device__ inline float GaussianDiskPdf(float x, float y, float falloff, float rmax){
	return GaussianDiskInfinityPdf(x, y, falloff) /
		(1.0f - exp(-falloff * rmax * rmax));
}

__host__ __device__ inline float GaussianDiskPdf(float3& center, float3& sample, float3& n, float falloff, float rmax){
	return GaussianDiskInfinityPdf(center, sample, n, falloff) / (1.f - exp(-falloff*rmax*rmax));
}

__host__ __device__ inline float Exponential(float u, float falloff){
	return -log(u) / falloff;
}

__host__ __device__ inline float ExponentialPdf(float x, float falloff){
	return falloff*exp(-falloff*x);
}


#endif