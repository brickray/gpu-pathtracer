#ifndef H_CAMERA_H
#define H_CAMERA_H

#include "common.h"
#include "ray.h"

class Camera{
public:
	float3 position;
	float3 u, v, w;
	float2 resolution;
	float distance;
	float fov;
	float apertureRadius;
	float focalDistance;

	bool filmic; //true when using filmic tonemap. false gamma correction

private:
	float width, height;
	float2 pixel2screen;
	float ratio;
public:
	__host__ __device__ Camera(){

	}
	__host__ __device__ Camera(float3 pos, float3 uu, float3 vv, float3 ww,
		float2 res, float dis, float angle, float radius, float focal)
	:position(pos),u(uu),v(vv),w(ww)
	,resolution(res),distance(dis),fov(angle)
	,apertureRadius(radius),focalDistance(focal){
		float half_fov = fov*.5f;
		height = tanf(DegreesToRadians(half_fov))*distance;
		width = height*resolution.x / resolution.y;

		pixel2screen.x = 2.f*width / resolution.x;
		pixel2screen.y = 2.f*height / resolution.y;
		ratio = focalDistance / distance;
	}

	__host__ __device__ Ray GeneratePrimaryRay(float x, float y, float2 xy){
		float xx = x*pixel2screen.x - width;
		float yy = y*pixel2screen.y - height;

		float3 dir, orig = position;

		if (apertureRadius > 0.00001f){
			float2 aperture_xy = xy*apertureRadius;
			float focal_x = ratio*xx;
			float focal_y = ratio*yy;
			float3 aperture = make_float3(aperture_xy, 0);
			float3 focal = make_float3(focal_x, focal_y, -focalDistance);

			dir = focal - aperture; 
			dir = dir.x*u + dir.y*v + dir.z*w;
			orig += (aperture.x*u + aperture.y*v);
		}
		else{
			dir = xx*u + yy*v + -distance*w;
		}

		dir = normalize(dir);

		Ray ray;
		ray.o = orig;
		ray.d = dir;
		return ray;
	}

	__host__ __device__ void Lookat(const float3& eye_pos, const float3& dest, const float3& up){
		position = eye_pos;
		w = normalize(eye_pos - dest);
		u = normalize(cross(up, w));
		v = normalize(cross(w, u));
	}
};


#endif