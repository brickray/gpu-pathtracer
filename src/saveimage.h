#ifndef H_SAVEIMAGE_H
#define H_SAVEIMAGE_H 

#include <cuda_runtime.h>
#include <fstream>
#include <stb\stb_image.h>
#include "cutil_math.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb\stb_image_write.h>

void save_bmp(const char* filename, unsigned width, unsigned height, float3* input){
	std::ofstream out(filename, std::ios::binary);
	if (!out.is_open()){
		return;
	}

	char* szBuf = new char[width*height * 3];
	char* rgb = szBuf;
	for (size_t i = 0; i < height; ++i)
	for (size_t j = 0; j < width; ++j)
	{
		unsigned pixel = i*width + j;
		rgb[0] = (unsigned char)(clamp(input[pixel].x, 0.f, 1.f) * 255.f);
		rgb[1] = (unsigned char)(clamp(input[pixel].y, 0.f, 1.f) * 255.f);
		rgb[2] = (unsigned char)(clamp(input[pixel].z, 0.f, 1.f) * 255.f);

		rgb += 3;
	}

	typedef struct tagBITMAPFILEHEADER
	{
		unsigned int bfSize; //file size;
		unsigned short bfReserved1;
		unsigned short bfReserved2;
		unsigned int bfOffbits; // the offset of the real data (rgb color)
	}BITMAPFILEHEADER;

	typedef struct tagBITMAPINFOHEADER
	{
		unsigned int biSize; //the size of struct, always 40
		int biWidth; //width of image;
		int biHeight;// height of image;
		unsigned short biPlanes; //must be 1
		unsigned short biBitCount; //number of bits in color, may 1, 2, 4, 8, 16, 24, 32
		unsigned int biCompression; //method of compression, 0 stands for non-compression
		unsigned int biSizeImage; //
		int biXPelsPerMeter; //resolution of X direction
		int biYPelsPerMeter; //resolution of Y direction
		unsigned int biClrUsed;
		unsigned int biClrImportant;
	}BITMAPINFOHEADER;

	BITMAPINFOHEADER infoheader;
	memset(&infoheader, 0, sizeof(infoheader));
	infoheader.biWidth = width;
	infoheader.biHeight = height;
	infoheader.biSize = sizeof(infoheader);
	infoheader.biPlanes = 1;
	infoheader.biBitCount = 24;
	infoheader.biCompression = 0;

	BITMAPFILEHEADER fileheader;
	memset(&fileheader, 0, sizeof(fileheader));
	fileheader.bfOffbits = 54;
	fileheader.bfSize = fileheader.bfOffbits + width * height * 3;

	unsigned short type = 0x4D42; // for memory align
	out.write((char*)&type, sizeof(unsigned short));
	out.write((char*)&fileheader, sizeof(fileheader));
	out.write((char*)&infoheader, sizeof(infoheader));
	out.write(szBuf, width * height * 3);

	out.close();

	delete[] szBuf;

	return;
}

void save_png(const char* filename, unsigned width, unsigned height, float3* input){
	unsigned char* transform = new unsigned char[width*height * 3];
	for (int i = 0; i < height; ++i){
		for (int j = 0; j < width; ++j){
			unsigned pixel = i*width + j;
			unsigned inverse = (height - i - 1)*width + j;
			transform[3 * pixel] = unsigned(clamp(input[inverse].x, 0.f, 1.f) * 255.f);
			transform[3 * pixel + 1] = unsigned(clamp(input[inverse].y, 0.f, 1.f) * 255.f);
			transform[3 * pixel + 2] = unsigned(clamp(input[inverse].z, 0.f, 1.f) * 255.f);
		}
	}

	stbi_write_png(filename, width, height, 3, transform, 0);
	fprintf(stderr, "Save .png file successfully\n");

	delete[] transform;
}

#endif