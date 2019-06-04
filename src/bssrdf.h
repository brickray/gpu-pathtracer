#ifndef H_BSSRDF_H
#define H_BSSRDF_H

#include "common.h"
#include "wrap.h"

class Bssrdf{
public:
	float3 absorb, scatterPrime;
	float eta, g, A;
public:
	Bssrdf(float3 absorb, float3 scatterPrime, float eta, float g)
		:absorb(absorb),
		scatterPrime(scatterPrime),
		eta(eta),
		g(g){
		float fdr = Fdr(eta);
		A = (1.f + fdr) / (1.f - fdr);
	}

	__host__ __device__ float Fdr(float eta){
		//see Donner. C 2006 Chapter 5
		//the internal Fresnel reflectivity
		//approximated with a simple polynomial expansion
		if (eta < 1.f)
			return -0.4399f + 0.7099f / eta - 0.3199f / (eta*eta) +
			0.0636f / (eta*eta*eta);
		else
			return -1.4399f / (eta*eta) + 0.7099f / eta + 0.6911f + 0.0636f*eta;
	}

	//Rd(r)=α′/4π[zr(σtr*dr+1)e−σtr*dr/dr3+zv(σtr*dv+1)e−σtr*dv/dv3]
	__host__ __device__ float3 Rd(float d2){
		//see Donner. C 2006 Chapter 5 for the full derivation
		//of the following disffusion dipole approximation equation
		float3 sigmaA = absorb;
		float3 sigmaSPrime = scatterPrime;
		float3 sigmaTPrime = sigmaA + sigmaSPrime;
		float3 sigmaTr = Sqrt(3.f*sigmaA*sigmaTPrime);
		float3 one = { 1.f, 1.f, 1.f };
		//distance beneath surface where we put the positive dipole light
		float3 zr = one / sigmaTPrime;
		//distance above surface where we put the negative dipole light
		float3 zv = zr + 4.f / 3.f*A*zr;
		//distance from x to the dipole light
		//Pythagorean theorem（勾股定理）
		float3 dr = Sqrt(zr*zr + make_float3(d2, d2, d2));
		float3 dv = Sqrt(zv*zv + make_float3(d2, d2, d2));

		float3 alphaPrime = sigmaSPrime / sigmaTPrime;
		float3 sTrDr = sigmaTr*dr;
		float3 sTrDv = sigmaTr*dv;
		float3 rd = 0.25f*ONE_OVER_PI*alphaPrime*(
			(zr*(one + sTrDr)*Exp(-sTrDr) / (dr*dr*dr)) +
			(zv*(one + sTrDv)*Exp(-sTrDv) / (dv*dv*dv)));
		return Clamp(rd, 0.f, INFINITY);
	}

	__host__ __device__ void SampleProbeRay(float3 pos, float3 nor, float2 u, float sigmaTr, float rMax, Ray& probeRay, float& pdf){
		float2 sample = GaussianDisk(u.x, u.y, sigmaTr, rMax);
		float halfChordLength = sqrtf(rMax*rMax - dot(sample, sample));
		float3 uu, ww;
		MakeCoordinate(nor, uu, ww);
		float3 p = make_float3(sample.x, -halfChordLength, sample.y);
		p = ToWorld(p, uu, nor, ww);
		p += pos;
		probeRay.o = p;
		probeRay.d = nor;
		probeRay.tmax = 2.f*halfChordLength;
		pdf = GaussianDiskPdf(sample.x, sample.y, sigmaTr, rMax);
		return;
	}

	__host__ __device__ float3 GetSigmaTr(){
		float3 sigmaA = absorb;
		float3 sigmaSPrime = scatterPrime;
		float3 sigmaTPrime = sigmaA + sigmaSPrime;
		return Sqrt(3.f*sigmaA*sigmaTPrime);
	}

	__host__ __device__ float3 GetSigmaS(){
		return scatterPrime / (1.f - g);
	}

	__host__ __device__ float3 GetSigmaT(){
		return GetSigmaS() + absorb;
	}

	__host__ __device__ float GetPhase(){
		return ONE_OVER_FOUR_PI;
	}

	__host__ __device__ float RdIntegral(float alphap, float A){
		float sqrtTerm = sqrtf(3.f*(1.f - alphap));
		return alphap / 2.f *(1.f + expf(-4.f / 3.f*A*sqrtTerm))*expf(-sqrtTerm);
	}

	//from pbrt-v2
	__host__ __device__ void ConvertFromDiffuse(float3 kd, float meanPathLength, float eta){
		float rgb[3] = { kd.x, kd.y, kd.z };
		float sigmaSPrime[3];
		float sigmaA[3];
		for (int i = 0; i < 3; ++i){
			float alphaLow = 0.f, alphaHigh = 1.f;
			float kd0 = RdIntegral(alphaLow, A);
			float kd1 = RdIntegral(alphaHigh, A);
			for (int j = 0; j < 16; ++j){
				float alphaMid = (alphaLow + alphaHigh)*0.5f;
				float kdt = RdIntegral(alphaMid, A);
				if (kdt < rgb[i]) {
					alphaLow = alphaMid;
					kd0 = kdt;
				}
				else{
					alphaHigh = alphaMid;
					kd1 = kdt;
				}
			}

			float alphap = (alphaLow + alphaHigh)*0.5f;
			float sigmaTr = 1.f / meanPathLength;
			float sigmaPrimeT = sigmaTr / sqrtf(3.f*(1.f - alphap));
			sigmaSPrime[i] = alphap*sigmaPrimeT;
			sigmaA[i] = sigmaPrimeT - sigmaSPrime[i];
		}

		scatterPrime = make_float3(sigmaSPrime[0], sigmaSPrime[1], sigmaSPrime[2]);
		absorb = make_float3(sigmaA[0], sigmaA[1], sigmaA[2]);
	}
};

//signle scatter
static float BeamDiffusionSS(float sigmaS, float sigmaA, float g, float eta, float r){
	float sigmaT = sigmaA + sigmaS, albedo = sigmaS / sigmaT;

}

#endif