#ifndef H_MESH_H
#define H_MESH_H

#include "common.h"
#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>

#include "bbox.h"
#include "wrap.h"
#include "intersection.h"

struct Vertex{
	float3 v;
	float3 n;
	float2 uv;
	float3 t;
};

class Triangle{
public:
	Vertex v1, v2, v3;
	int matIdx;
	int bssrdfIdx;
	int lightIdx;
	int mediumInside, mediumOutside;

public:
	__host__ __device__ BBox GetBBox() const{
		float3 f1 = v1.v, f2 = v2.v, f3 = v3.v;
		BBox bbox;
		bbox.Expand(f1);
		bbox.Expand(f2);
		bbox.Expand(f3);

		return bbox;
	}

	__host__ __device__ float GetSurfaceArea() const{
		float3 e1 = v2.v - v1.v;
		float3 e2 = v3.v - v1.v;
		return length(cross(e1, e2)) * 0.5f;
	}

	__host__ __device__ bool Intersect(Ray& ray, Intersection* isect) const{
		float3 e1 = v2.v - v1.v;
		float3 e2 = v3.v - v1.v;
		float3 s1 = cross(ray.d, e2);
		float divisor = dot(s1, e1);
		if (fabs(divisor) < 1e-8f)
			return false;
		float invDivisor = 1.0 / divisor;
		float3 s = ray.o - v1.v;
		float b1 = dot(s, s1)*invDivisor;
		if (b1 < 0.0 || b1 > 1.0)
			return false;

		float3 s2 = cross(s, e1);
		float b2 = dot(ray.d, s2)*invDivisor;
		if (b2 < 0.0 || b1 + b2 > 1.0)
			return false;

		float tt = dot(e2, s2) * invDivisor;
		if (tt < ray.tmin || tt > ray.tmax)
			return false;

		ray.tmax = tt;
		if (isect){
			isect->pos = ray(tt);
			//不能默认文件里的法线已经归一化，这里需要手动归一化一下
			//coffee场景里就因为这个问题导致渲染出现奇怪条纹，查了2天才查出来。。
			isect->nor = normalize(v1.n * (1.f - b1 - b2) + v2.n*b1 + v3.n*b2);
			isect->uv = v1.uv*(1.f - b1 - b2) + v2.uv*b1 + v3.uv*b2;
			isect->matIdx = matIdx;
			isect->lightIdx = lightIdx;
			isect->dpdu = normalize(v1.t * (1.f - b1 - b2) + v2.t*b1 + v3.t*b2);
			isect->bssrdf = bssrdfIdx;
			isect->mediumInside = mediumInside;
			isect->mediumOutside = mediumOutside;
		}

		return true;
	}

	__host__ __device__ void SampleShape(float3& pos, float2& u, float3& dir, float3& nor, float& pdf) const{
		float2 uv = UniformTriangle(u.x, u.y);
		float3 p = uv.x * v1.v + uv.y * v2.v + (1 - uv.x - uv.y)*v3.v;
		float3 normal = normalize(uv.x*v1.n + uv.y*v2.n + (1 - uv.x - uv.y)*v3.n);
		dir = p - pos;
		nor = normal;
		pdf = 1.f / (GetSurfaceArea() * fabsf(dot(normal, normalize(dir)))) * dot(dir, dir);
		if (dot(normal, dir) >= 0.f)
			pdf = 0.f;
	}

	__host__ __device__ void SampleShape(float4& u, float3& pos, float3& dir, float3& nor, float& pdfA, float& pdfW){
		float2 uv = UniformTriangle(u.x, u.y);
		pos = uv.x * v1.v + uv.y * v2.v + (1 - uv.x - uv.y)*v3.v;
	    nor = normalize(uv.x*v1.n + uv.y*v2.n + (1 - uv.x - uv.y)*v3.n);
		dir = CosineHemiSphere(u.z, u.w, nor, pdfW);
		float3 uu, ww;
		MakeCoordinate(nor, uu, ww);
		dir = ToWorld(dir, uu, nor, ww);
		pdfA = 1.f / GetSurfaceArea();
	}
};

class Scene;
class Mesh{
public:
	vector<Vertex> vertices;
	vector<Triangle> triangles;
	int matIdx;
	int bssrdfIdx;

public:
	void LoadObjFromFile(std::string filename, unsigned int flags, mat4& trs);

private:
	void processNode(aiNode* node, const aiScene* scene, mat4& trs);
	void processMesh(aiMesh* aimesh, const aiScene* scene, mat4& trs);
	float3 genTangent(int idx1, int idx2, int idx3);
};


#endif