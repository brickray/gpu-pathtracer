#ifndef H_MEDIUM_H
#define H_MEDIUM_H

#include "common.h"
#include "ray.h"
#include "wrap.h"

class Homogeneous{
public:
	float3 sigmaA, sigmaS, sigmaT;
	float g;

public:
	__device__ float3 Tr(const Ray& ray, curandState& rng) const{
		float3 c = sigmaT*(-ray.tmax);
		return Exp(c);
	}

	__device__ float3 Sample(const Ray& ray, curandState& rng, float& t, bool& sampled) const{
		float sigma = dot(sigmaT, { 0.212671f, 0.715160f, 0.072169f });
		float dist = Exponential(curand_uniform(&rng), sigma);
		bool sampledMedium = dist < ray.tmax;
		sampled = sampledMedium;
		t = dist;

		return sampledMedium ? (sigmaS / sigmaT) : make_float3(1.f, 1.f, 1.f);
	}

	__device__ void SamplePhase(float2 u, float3& dir, float& phase, float& pdf) const{
		if (g == 0){
			phase = ONE_OVER_FOUR_PI;
			dir = UniformSphere(u.x, u.y, pdf);
			return;
		}

		//hg
		float costheta;
		if (fabs(g) < 1e-3)
			costheta = 1.f - 2.f*u.x;
		else{
			float sqrtTerm = (1.f - g*g) / (1.f - g + 2.f*g*u.x);
			costheta = (1.f + g*g - sqrtTerm*sqrtTerm) / (2.f*g);
		}

		float sintheta = sqrt(1.f - costheta*costheta);
		float phi = TWOPI*u.y;
		float sinphi = sin(phi), cosphi = cos(phi);
		dir = make_float3(sintheta*cosphi, costheta, sintheta*sinphi);
		float cubicTerm = (1.f + g*g - 2.f*g*costheta);
		phase = ONE_OVER_FOUR_PI*(1.f - g*g) / sqrt(cubicTerm*cubicTerm*cubicTerm);
		pdf = phase;
	}
};

class Heterogeneous{
public:
	float3 sigmaA, sigmaS, sigmaT;
	float g;
	int nx, ny, nz;
	float* density;
	float invMaxDensity;
	float3 p0, p1;

public:
	__device__ float3 Tr(const Ray& ray, curandState& rng) const{
		float sigma = dot(sigmaT, { 0.212671f, 0.715160f, 0.072169f });
		Ray r = ray;
		float3 d = p1 - p0;
		float tr = 1.f;
		float dist = 0.f;
		while (true){
			dist += -log(curand_uniform(&rng)) * invMaxDensity / sigma;
			if (dist > r.tmax) break;
			float3 p = r(dist);
			p = (p - p0) / d;
			tr *= 1.f - getDensity(p)*invMaxDensity;
		}

		return{ tr, tr, tr };
	}

	__device__ float3 Sample(const Ray& ray, curandState& rng, float& t, bool& sampled) const{
		float sigma = dot(sigmaT, { 0.212671f, 0.715160f, 0.072169f });
		Ray r = ray;
		float3 d = p1 - p0;
		float dist = 0.f;
		while (true){
			dist += -log(curand_uniform(&rng)) * invMaxDensity / sigma;
			if (dist > r.tmax) break;
			float3 p = r(dist);
			p = (p - p0) / d;
			if (getDensity(p)*invMaxDensity > curand_uniform(&rng)){
				t = dist;
				sampled = true;
				return sigmaS / sigmaT;
			}
		}

		t = dist;
		sampled = false;
		return make_float3(1.f, 1.f, 1.f);
	}

private:
	__device__ float getDensity(float3& p) const{
		float3 ps = make_float3(p.x*nx, p.y*ny, p.z*nz);
		float3 psi = make_float3(floor(ps.x), floor(ps.y), floor(ps.z));
		float3 delta = ps - psi;
		float d00 = lerp(d(psi), d(psi + make_float3(1, 0, 0)), delta.x);
		float d10 = lerp(d(psi + make_float3(0, 1, 0)), d(psi + make_float3(1, 1, 0)), delta.x);
		float d01 = lerp(d(psi + make_float3(0, 0, 1)), d(psi + make_float3(1, 0, 1)), delta.x);
		float d11 = lerp(d(psi + make_float3(0, 1, 1)), d(psi + make_float3(1, 1, 1)), delta.x);
		float d0 = lerp(d00, d10, delta.y);
		float d1 = lerp(d01, d11, delta.y);
		return lerp(d0, d1, delta.z);
	}

	__device__ float d(float3& p) const{
		int x = p.x, y = p.y, z = p.z;
		if (x<0 || x>nx-1 || y<0 || y>ny-1 || z<0 || z>nz-1) return 0.f;
		return density[z*ny*nx + y*nx + x];
	}
};

enum MediumType{
	MT_HOMOGENEOUS = 0,
	MT_HETEROGENEOUS,
};

class Medium{
public:
	MediumType type;

	union{
		Homogeneous homogeneous;
		Heterogeneous heterogeneous;
	};
};

static void ReadDensityFromFile(const char* file, int nx, int ny, int nz, float* d){
	FILE* fp = nullptr;
	fp = fopen(file, "r");
	for (int i = 0; i < nx*ny*nz; ++i){
		float c;
		fscanf(fp, "%f\n", &c);
		d[i] = c;
	}
}

#endif