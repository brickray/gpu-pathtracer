#ifndef H_INTERSECTION_H
#define H_INTERSECTION_H

#include "common.h"

struct Intersection{
	float3 pos; //hit point
	float3 nor; //normal of hit point
	float2 uv; //tex coord of hit point
	float3 dpdu; //tangent 
	int matIdx; //index of bsdf
	int bssrdf; //index of bssrdf
	int lightIdx;
	int mediumInside, mediumOutside;

	__host__ __device__ Intersection(){
		lightIdx = -1;
	}
};

#endif