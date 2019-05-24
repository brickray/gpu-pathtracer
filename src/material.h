#ifndef H_MATERIAL_H
#define H_MATERIAL_H

#include "common.h"

using namespace std;

enum MaterialType{
	MT_LAMBERTIAN = 0,
	MT_MIRROR,
	MT_DIELECTRIC,
	MT_ROUGHCONDUCTOR,
	MT_SUBSTRATE,
};

class Material{
public:
	MaterialType type;
	float roughness;
	float insideIOR, outsideIOR;
	float3 k, eta; //metal
	float3 diffuse, specular;
};

__host__ __device__ inline bool IsDiffuse(MaterialType type){
	return type != MT_MIRROR && type != MT_DIELECTRIC;
}

#endif