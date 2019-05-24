#ifndef H_TEXTURE_H
#define H_TEXTURE_H

#include <cuda_runtime.h>
#include <string>
#include <map>

using namespace std;

class Texture{
public:
	float3* data;
	int width, height;

public:
	Texture(){}
	Texture(int w, int h);
	~Texture();
};

class TextureManager{
public:
	map<string, Texture*> texture;

public:
	void LoadTexture(const char* filename);
	void DeleteTexture(const char* filename);
};

#endif