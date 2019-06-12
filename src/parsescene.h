#ifndef H_PARSESCENE_H
#define H_PARSESCENE_H

#include "common.h"
#include "camera.h"
#include "scene.h"

struct GlobalConfig{
	//default size[512,512]
	int width, height; //canvas size 

	//default parameter
	//position [0,0,0]
	//u [1,0,0]
	//v [0,1,0]
	//w [0,0,1]
	//fov [60]
	//apertureRadius [0]
	//focalDistance [0]
	Camera camera;
	float camera_move_speed;
	//default 0.01
	float epsilon;
};

bool LoadScene(const char* filename, GlobalConfig& config, Scene& scene);

#endif