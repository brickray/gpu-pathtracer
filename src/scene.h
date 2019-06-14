#ifndef H_SCENE_H
#define H_SCENE_H

#include "common.h"
#include "bvh.h"
#include "primitive.h"
#include "material.h"
#include "texture.h"
#include "area.h"
#include "camera.h"
#include "bssrdf.h"
#include "medium.h"
#include "infinite.h"

enum IntegratorType{
	IT_AO = 0,
	IT_PT,
	IT_VPT,
	IT_LT,
	IT_BDPT,
};

class Scene{
public:
	vector<Primitive> primitives;
	vector<Material> materials;
	vector<Bssrdf> bssrdfs;
	vector<Medium> mediums;
	vector<Area> lights;
	vector<Texture> textures;
	vector<float> lightDistribution;
	Camera* camera;
	Infinite infinite;
	BVH bvh;
	struct{
		IntegratorType type;
		union{
			float maxDist;
			int maxDepth;
		};
	} integrator;

public:
	void Init(Camera* cam, string file){
		camera = cam;

		clock_t now = clock();
		bvh.LoadOrBuildBVH(primitives, file);
		printf("Build bvh using %.3fms\n", float(clock() - now));
		printf("Bvh total nodes:%d\n", bvh.total_nodes);
		printf("Scene Bounds [%.3f, %.3f, %.3f]-[%.3f, %.3f, %.3f]\n",
			bvh.root_box.fmin.x, bvh.root_box.fmin.y, bvh.root_box.fmin.z,
			bvh.root_box.fmax.x, bvh.root_box.fmax.y, bvh.root_box.fmax.z);

		//infinite light prepare
		if (infinite.isvalid)
			infinite.Init(bvh.root_box);
		//init light distribution
		float3 luma = { 0.212671f, 0.715160f, 0.072169f };
		float sum = 0.f;
		lightDistribution.push_back(0.f);
		for (int i = 0; i < lights.size(); ++i){
			Area light = lights[i];
			float3 power = light.GetPower();
			float p = dot(luma, power);
			sum += p;
			lightDistribution.push_back(sum);
		}
		if (infinite.isvalid){
			float3 power = infinite.GetPower();
			sum += dot(luma, power);
			lightDistribution.push_back(sum);
		}

		for (int i = 0; i < lightDistribution.size(); ++i)
			lightDistribution[i] /= sum;
	}
};

#endif