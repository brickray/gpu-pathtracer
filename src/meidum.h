#ifndef H_MEDIUM_H
#define H_MEDIUM_H

#include "common.h"
#include "ray.h"
#include "wrap.h"

//just homogeneous now
class Medium{
public:
	float3 sigmaA, sigmaS, sigmaT;
	float g;

public:
	__device__ float3 Tr(const Ray& ray, curandState& rng) const{
		float3 c = sigmaT*(-ray.tmax);
		return Exp(c);
	}

	__device__ float3 Sample(const Ray& ray, float u, float& t, bool& sampled) const{
		float sigma = dot(sigmaT, { 0.212671f, 0.715160f, 0.072169f });
		float dist = Exponential(u, sigma);
		bool sampledMedium = dist < ray.tmax;
		sampled = sampledMedium;
		t = dist;

		return sampledMedium ? (sigmaS / sigmaT) : make_float3(1.f, 1.f, 1.f);
	}
};

#endif