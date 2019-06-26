#ifndef H_CAMERA_H
#define H_CAMERA_H

#include "common.h"
#include "ray.h"
#include "wrap.h"

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
	bool environment; //environment camera?
	int medium;

private:
	float width, height;
	float2 pixel2screen;
	float ratio;
	float area;
public:
	__host__ __device__ Camera(){

	}
	__host__ __device__ Camera(float3 pos, float3 uu, float3 vv, float3 ww,
		float2 res, float dis, float angle, float radius, float focal, bool filmic, int medium)
		:position(pos), u(uu), v(vv), w(ww)
		, resolution(res), distance(dis), fov(angle)
		, apertureRadius(radius), focalDistance(focal)
		, filmic(filmic)
		, medium(medium){
		float half_fov = fov*.5f;
		height = tanf(DegreesToRadians(half_fov))*distance;
		width = height*resolution.x / resolution.y;
		area = 4.f*width*height;

		pixel2screen.x = 2.f*width / resolution.x;
		pixel2screen.y = 2.f*height / resolution.y;
		ratio = focalDistance / distance;
	}

	__host__ __device__ Ray GeneratePrimaryRay(float x, float y, float2 xy){
		if (environment){
			float3 orig = position;
			float theta = PI*(1.f - y / resolution.y);
			float phi = TWOPI*(1.f - x / resolution.x);
			float3 dir = make_float3(sin(theta)*cos(phi), cos(theta), sin(theta)*sin(phi));
			dir = dir.x*u + dir.y*v - dir.z*w;
			return Ray(orig, dir);
		}

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

	__host__ __device__ void SampleCamera(float3& pos, Ray& ray, float& we, float& pdf, int& raster, float epsilon){
		float3 dir = position - pos;
		float3 ndir = normalize(dir);
		ray = Ray(pos, ndir, nullptr, epsilon, length(dir) - epsilon);
		float3 negative_dir = -ndir;
		//to camera space
		float3 cndir = ToLocal(negative_dir, u, v, w);
		if (cndir.z >= 0.f){
			pdf = 0;
			return;
		}
		float costheta = dot(cndir, make_float3(0, 0, -1.f));
		float scale = -distance / cndir.z;
		cndir *= scale;
		float2 plane = make_float2(cndir.x, cndir.y);
		plane /= make_float2(width, height);
		if (plane.x > 1.f || plane.x < -1.f ||
			plane.y > 1.f || plane.y < -1.f){
			pdf = 0;
			return;
		}

		plane = plane*0.5f + make_float2(0.5f, 0.5f);
		int x = floor(plane.x * (resolution.x - 1) + 0.5f);
		int y = floor(plane.y * (resolution.y - 1) + 0.5f);
		raster = y*resolution.x + x;
		pdf = dot(dir, dir) / costheta;
		we = distance*distance / (area*costheta*costheta*costheta*costheta);
	}

	//dir must from camera pos to dest pos
	__host__ __device__ void PdfCamera(float3& dir, float& pdfA, float& pdfW){
		pdfA = 1.f;
		float costheta = dot(dir, -w);
		pdfW = distance*distance / (area*costheta*costheta*costheta);
	}

	__host__ __device__ void Lookat(const float3& eye_pos, const float3& dest, const float3& up){
		position = eye_pos;
		w = normalize(eye_pos - dest);
		u = normalize(cross(up, w));
		v = normalize(cross(w, u));
	}
};


#endif