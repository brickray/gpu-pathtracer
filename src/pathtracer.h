#ifndef H_PATHTRACER_H
#define H_PATHTRACER_H

#include "common.h"

class Camera;
class Scene;
class BVH;

struct HDRMap{
	bool isvalid;
	int width, height;
	float4* image;
};

void Render(Scene& scene, unsigned width, unsigned height, Camera* camera, unsigned iter, bool reset, float3* output);
void BeginRender(Scene& scene, BVH& bvh, Camera cam, unsigned width, unsigned height, float ep, int max_depth, HDRMap& hdrmap);
void EndRender();

#endif