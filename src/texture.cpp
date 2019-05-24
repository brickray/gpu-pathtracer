#include "texture.h"

#include <math.h>
#include <stb\stb_image.h>

Texture::Texture(int w, int h){
	width = w;
	height = h;
	data = new float3[width*height];
}

Texture::~Texture(){
	delete[] data;
}

void TextureManager::LoadTexture(const char* filename){
	if (texture[filename])
		return;

	int width, height, component;
	unsigned char* tex = NULL;// stbi_load(filename, &width, &height, &component, 0);
	if (tex){
		Texture* t = new Texture(width, height);
		for (int i = 0; i < width*height; ++i){
			float r = tex[component*i] * 255;
			float g = tex[component*i + 1] * 255;
			float b = tex[component*i + 2] * 255;

			//convert from srgb space to linear space
			t->data[i] = make_float3(powf(r, 2.2f), powf(g, 2.2f), powf(b, 2.2f));
		}

		texture[filename] = t;
		//stbi_image_free(tex);
	}
	else{
		fprintf(stderr, "Can not load file\n");
		//stbi_image_free(tex);
	}
}

void TextureManager::DeleteTexture(const char* filename){
	map<string, Texture*>::iterator it = texture.find(filename);
	if (it != texture.end()){
		delete (it->second);
		texture.erase(it);
	}
}