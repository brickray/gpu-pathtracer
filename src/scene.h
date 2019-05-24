#ifndef H_SCENE_H
#define H_SCENE_H

#include <vector>
#include "mesh.h"
#include "material.h"
#include "area.h"
#include "camera.h"

class Scene{
public:
	vector<Triangle> triangles;
	vector<Material> materials;
	vector<Area> lights;
	vector<float> lightDistribution;
	Camera camera;

public:
	void Init(){
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

		for (int i = 0; i < lightDistribution.size(); ++i)
			lightDistribution[i] /= sum;
	}
};

#endif