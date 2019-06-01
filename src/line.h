#ifndef H_LINE_H
#define H_LINE_H

#include "common.h"
#include "bbox.h"
#include "intersection.h"

class Line{
public:
	float3 p0, p1;
	float width0, width1;
	int matIdx;

public:
	__host__ __device__ BBox GetBBox() const{
		BBox bbox;

		float maxWidth = width0 > width1 ? width0 : width1;
		float3 width = make_float3(maxWidth, maxWidth, maxWidth);
		bbox.Expand(p0 - width);
		bbox.Expand(p0 + width);
		bbox.Expand(p1 - width);
		bbox.Expand(p1 + width);
		return bbox;
	}

	__host__ __device__ float GetSurfaceArea() const{
		float len = length(p1 - p0);
		float width = 0.5f*(width0 + width1);
		return len*width;
	}

	__host__ __device__ bool Intersect(Ray& ray, Intersection* isect) const{
		// setup intersection params
		auto u = ray.d;
		auto v = p1 - p0;
		auto w = ray.o - p0;

		// compute values to solve a linear system
		auto a = dot(u, u);
		auto b = dot(u, v);
		auto c = dot(v, v);
		auto d = dot(u, w);
		auto e = dot(v, w);
		auto det = a * c - b * b;

		// check determinant and exit if lines are parallel
		// (could use EPSILONS if desired)
		if (det == 0) return false;

		// compute Parameters on both ray and segment
		auto t = (b * e - c * d) / det;
		auto s = (a * e - b * d) / det;

		// exit if not within bounds
		if (t < ray.tmin || t > ray.tmax) return false;

		// clamp segment param to segment corners
		s = clamp(s, (float)0, (float)1);

		// compute segment-segment distance on the closest points
		auto pr = ray.o + ray.d * t;
		auto pl = p0 + (p1 - p0) * s;
		auto prl = pr - pl;

		// check with the line radius at the same point
		auto d2 = dot(prl, prl);
		auto r = width0 * (1 - s) + width1 * s;
		if (d2 > r * r) return{};

		// intersection occurred: set params and exit
		ray.tmax = t;

		if (isect){
			isect->pos = ray(t);
			isect->nor = -ray.d;
			isect->uv = { s, sqrt(d2) / r };
			float3 dpdu, dpdv;
			MakeCoordinate(isect->nor, dpdu, dpdv);
			isect->dpdu = dpdu;
			isect->matIdx = matIdx;
			isect->lightIdx = -1;
		}

		return true;
	}
};

#endif