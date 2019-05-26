#ifndef H_COMMON_H
#define H_COMMON_H

#include <stdio.h>
#include <vector>
#include <string>
#include <map>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>
#include "cutil_math.h"
#include <thrust\device_vector.h>
#include <glm\glm.hpp>
#include <glm\gtc\matrix_transform.hpp>

using namespace std;
using namespace glm;

#define PI                  3.14159265358f
#define TWOPI               6.28318530716f
#define FOURPI              12.56637061432f 
#define ONE_OVER_PI         0.3183098861847f
#define ONE_OVER_TWO_PI     0.1591549430923f
#define ONE_OVER_FOUR_PI    0.0795774715461f

static void HandleError(cudaError_t err,
	const char *file,
	int line) {
	if (err != cudaSuccess) {
		printf("%s in %s at line %d\n", cudaGetErrorString(err),
			file, line);

		__debugbreak();
	}
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

__host__ __device__ inline float RadiansToDegrees(float radians) {
	float degrees = radians * 180.0 / PI;
	return degrees;
}

__host__ __device__ inline float DegreesToRadians(float degrees) {
	float radians = degrees / 180.0 * PI;
	return radians;
}

__host__ __device__ inline void Swap(float& a, float& b){
	float temp;
	temp = a;
	a = b;
	b = temp;
}

__host__ __device__ inline float3 Clamp(float3 value, float minValue, float maxValue){
	if (value.x < minValue) value.x = minValue;
	if (value.x > maxValue) value.x = maxValue;
	if (value.y < minValue) value.y = minValue;
	if (value.y > maxValue) value.y = maxValue;
	if (value.z < minValue) value.z = minValue;
	if (value.z > maxValue) value.z = maxValue;

	return value;
}

__host__ __device__ inline bool IsBlack(float3& c){
	return c.x == 0 && c.y == 0 && c.z == 0;
}

__host__ __device__ inline float3 Exp(const float3& c){
	float r = expf(c.x);
	float g = expf(c.y);
	float b = expf(c.z);
	return{ r, g, b };
}

__host__ __device__ inline float3 Sqrt(const float3& c){
	float r = sqrt(c.x);
	float g = sqrt(c.y);
	float b = sqrt(c.z);
	return{ r, g, b };
}

inline float3 VecToFloat3(vec3& v){
	return make_float3(v.x, v.y, v.z);
}

inline vec3 Float3ToVec(float3& v){
	return vec3(v.x, v.y, v.z);
}

#endif