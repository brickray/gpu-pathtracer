#include "pathtracer.h"
#include "camera.h"
#include "scene.h"
#include "bvh.h"
#include "device_launch_parameters.h"

Camera* dev_camera;
int maxDepth;
float3* dev_image, *dev_color;
LinearBVHNode* dev_bvh_nodes;
Primitive* dev_primitives;
Material* dev_materials;
Area* dev_lights;
float* dev_light_distribution;
float4* hdr_map;
uchar4** dev_textures;
int* texture_size;//0 1为第一张图的长宽， 2 3为第二张图的长宽，以此类推
texture<float4, 1, cudaReadModeElementType> hdr_texture;

__device__ Camera* kernel_camera;
__device__ int kernel_hdr_width, kernel_hdr_height;
__device__ float3* kernel_acc_image, *kernel_color;
__device__ LinearBVHNode* kernel_linear;
__device__ Primitive* kernel_primitives;
__device__ Material* kernel_materials;
__device__ Area* kernel_lights;
__device__ uchar4** kernel_textures;
__device__ int* kernel_texture_size;
__device__ float* kernel_light_distribution;
__device__ int kernel_light_distribution_size;
__device__ bool kernel_hdr_isvalid;
//不同场景需要不同的epsilon，不知道怎么样优雅的实现
__device__ float kernel_epsilon;

__device__ inline unsigned int WangHash(unsigned int seed)
{
	seed = (seed ^ 61) ^ (seed >> 16);
	seed = seed + (seed << 3);
	seed = seed ^ (seed >> 4);
	seed = seed * 0x27d4eb2d;
	seed = seed ^ (seed >> 15);

	return seed;
}

__device__ inline float DielectricFresnel(float cosi, float cost, const float& etai, const float& etat){
	float Rparl = (etat * cosi - etai * cost) / (etat * cosi + etai * cost);
	float Rperp = (etai * cosi - etat * cost) / (etai * cosi + etat * cost);

	return (Rparl * Rparl + Rperp * Rperp) * 0.5f;
}

__device__ inline float3 ConductFresnel(float cosi, const float3& eta, const float3& k){
	float3 tmp = (eta * eta + k * k) * cosi * cosi;
	float3 Rparl2 = (tmp - eta * cosi * 2.f + 1.f) /
		(tmp + eta * cosi * 2.f + 1.f);
	float3 tmp_f = (eta * eta + k * k);
	float3 Rperp2 = (tmp_f - eta * cosi * 2.f + cosi * cosi) /
		(tmp_f + eta * cosi * 2.f + cosi * cosi);
	return (Rparl2 + Rperp2) * 0.5f;
}

__device__ inline float GGX_D(float3& wh, float3& normal, float3 dpdu, float alphaU, float alphaV){
	float costheta = dot(wh, normal);
	if (costheta <= 0.f) return 0.f;
	costheta = clamp(costheta, 0.f, 1.f);
	float costheta2 = costheta*costheta;
	float sintheta2 = 1.f - costheta2;
	float costheta4 = costheta2*costheta2;
	float tantheta2 = sintheta2 / costheta2;

	float3 uu = dpdu;
	float3 dir = normalize(wh - costheta*normal);
	float cosphi = dot(dir, uu);
	float cosphi2 = cosphi*cosphi;
	float sinphi2 = 1.f - cosphi2;
	float sqrD = 1.f + tantheta2*(cosphi2 / (alphaU*alphaU) + sinphi2 / (alphaV*alphaV));
	return 1.f / (PI*alphaU*alphaV*costheta4*sqrD*sqrD);
}

__device__ inline float SmithG(float3& w, float3& normal, float3& wh, float3 dpdu, float alphaU, float alphaV){
	float wdn = dot(w, normal);
	if (wdn * dot(w, wh) < 0.f)	return 0.f;
	float sintheta = sqrtf(clamp(1.f - wdn*wdn, 0.f, 1.f));
	float tantheta = sintheta / wdn;
	if (isinf(tantheta)) return 0.f;

	float3 uu = dpdu;
	float3 dir = normalize(w - wdn*normal);
	float cosphi = dot(dir, uu);
	float cosphi2 = cosphi*cosphi;
	float sinphi2 = 1.f - cosphi2;
	float alpha2 = cosphi2 * (alphaU*alphaU) + sinphi2 * (alphaV*alphaV);
	float sqrD = alpha2*tantheta*tantheta;
	return 2.f / (1.f + sqrtf(1 + sqrD));
}

__device__ inline float GGX_G(float3& wo, float3& wi, float3& normal, float3& wh, float3 dpdu, float alphaU, float alphaV){
	return SmithG(wo, normal, wh, dpdu, alphaU, alphaV)*SmithG(wi, normal, wh, dpdu, alphaU, alphaV);
}

__device__ inline float3 SampleGGX(float alphaU, float alphaV, float u1, float u2){
	if (alphaU == alphaV){
		float costheta = sqrtf((1.f - u1) / (u1*(alphaU*alphaV - 1.f) + 1.f));
		float sintheta = sqrtf(1.f - costheta*costheta);
		float phi = 2 * PI*u2;
		float cosphi = cosf(phi);
		float sinphi = sinf(phi);

		return{
			sintheta*cosphi,
			costheta,
			sintheta*sinphi
		};
	}
	else{
		float phi;
		if (u2 <= 0.25) phi = atan(alphaV / alphaU*tan(TWOPI*u2));
		else if (u2 >= 0.75f) phi = atan(alphaV / alphaU*tan(TWOPI*u2)) + TWOPI;
		else phi = atan(alphaV / alphaU*tan(TWOPI*u2)) + PI;
		float sinphi = sin(phi), cosphi = cos(phi);
		float sinphi2 = sinphi * sinphi;
		float cosphi2 = 1.0f - sinphi2;
		float inverseA = 1.0f / (cosphi2 / (alphaU*alphaU) + sinphi2 / (alphaV*alphaV));
		float theta = atan(sqrt(inverseA * u1 / (1.0f - u1)));
		float sintheta = sin(theta), costheta = cos(theta);
		return{
			sintheta*cosphi,
			costheta,
			sintheta*sinphi
		};
	}
}

__device__ inline float3 SchlickFresnel(float3 specular, float costheta){
	float3 rs = specular;
	float c = 1.f - costheta;
	return rs + c*c*c*c*c *(make_float3(1.f, 1.f, 1.f) - rs);
}

__device__ inline float PowerHeuristic(int nf, float fPdf, int ng, float gPdf) {
	float f = nf * fPdf, g = ng * gPdf;
	return (f * f) / (f * f + g * g);
}

//当光源多的时候可以使用二分法加速
__device__ int LookUpLightDistribution(float u, float& pdf){
	for (int i = 0; i < kernel_light_distribution_size; ++i){
		float s = kernel_light_distribution[i];
		float e = kernel_light_distribution[i + 1];
		if (u >= s && u <= e){
			pdf = e - s;
			return i;
		}
	}
}

__device__ inline float PdfFromLightDistribution(int idx){
	return kernel_light_distribution[idx + 1] - kernel_light_distribution[idx];
}

__device__ inline void GammaCorrection(float3& in){
	float one_over_gamma = 1.f / 2.2f;
	float exposure = 1.41421356f;

	//pow(x,y) 的内部实现是expf(y*log(x)) 所以x需要大于0
	in = fmaxf(in, make_float3(1e-5, 1e-5, 1e-5));

	in.x = __powf(in.x*exposure, one_over_gamma);
	in.y = __powf(in.y*exposure, one_over_gamma);
	in.z = __powf(in.z*exposure, one_over_gamma);
}

__device__ inline void FilmicTonemapping(float3& in){
	float3 c = in - make_float3(0.004f, 0.004f, 0.004f);
	c = (c*(6.2f*c + 0.5f)) / (c*(6.2f*c + 1.7f) + 0.06f);
	c = Clamp(c, 0.f, 1.f);
	in = c;
}

__device__ inline float Luminance(const float3& c){
	return dot(c, { 0.212671f, 0.715160f, 0.072169f });
}

__device__ inline bool SameHemiSphere(float3& in, float3& out, float3& nor){
	return dot(in, nor)*dot(out, nor) > 0 ? true : false;
}

__device__ bool Intersect(Ray& ray, Intersection* isect){
	int stack[64];
	int* stack_top = stack;
	int* stack_bottom = stack;

	bool ret = false;
	int node_idx = 0;
	do{
		LinearBVHNode node = kernel_linear[node_idx];
		bool intersect = node.bbox.Intersect(ray);
		if (intersect){
			if (!node.is_leaf){
				*stack_top++ = node.second_child_offset;
				*stack_top++ = node_idx + 1;
			}
			else{
				for (int i = node.start; i <= node.end; ++i){
					Primitive prim = kernel_primitives[i];

					if (prim.type == GT_TRIANGLE){
						if (prim.triangle.Intersect(ray, isect))
							ret = true;
					}
					else{
						if (prim.line.Intersect(ray, isect))
							ret = true;
					}
				}
			}
		}

		if (stack_top == stack_bottom)
			break;
		node_idx = *--stack_top;
	} while (true);

	return ret;
}

__device__ bool IntersectP(Ray& ray){
	int stack[64];
	int* stack_top = stack;
	int* stack_bottom = stack;

	int node_idx = 0;
	do{
		LinearBVHNode node = kernel_linear[node_idx];
		bool intersect = node.bbox.Intersect(ray);
		if (intersect){
			if (!node.is_leaf){
				*stack_top++ = node.second_child_offset;
				*stack_top++ = node_idx + 1;
			}
			else{
				for (int i = node.start; i <= node.end; ++i){
					Primitive prim = kernel_primitives[i];
					if (prim.type == GT_TRIANGLE){
						if (prim.triangle.Intersect(ray, nullptr))
							return true;
					}
					else{
						if (prim.line.Intersect(ray, nullptr))
							return true;
					}
				}
			}
		}

		if (stack_top == stack_bottom)
			break;
		node_idx = *--stack_top;
	} while (true);

	return false;
}

__device__ inline float4 getTexel(Material material, int w, int h, int2 uv){
	float inv = 1.f / 255.f;

	int x = uv.x, y = uv.y;
	float rx = x - (x / w)*w;
	float ry = y - (y / h)*h;
	x = (rx < 0) ? rx + w : rx;
	y = (ry < 0) ? ry + h : ry;
	if (x < 0) x = 0;
	if (x > w - 1) x = w - 1;
	if (y < 0) y = 0;
	if (y > h - 1) y = h - 1;

	uchar4 c = kernel_textures[material.textureIdx][y*w + x];
	return make_float4(c.x*inv, c.y*inv, c.z*inv, c.w * inv);
}

__device__ inline float4 GetTexel(Material material, float2 uv){
	if (material.textureIdx == -1)
		return make_float4(material.diffuse, 1.f);

	int w = kernel_texture_size[material.textureIdx * 2];
	int h = kernel_texture_size[material.textureIdx * 2 + 1];
	float xx = w * uv.x;
	float yy = h * uv.y;
	int x = floor(xx);
	int y = floor(yy);
	float dx = fabs(xx - x);
	float dy = fabs(yy - y);
	float4 c00 = getTexel(material, w, h, make_int2(x, y));
	float4 c10 = getTexel(material, w, h, make_int2(x + 1, y));
	float4 c01 = getTexel(material, w, h, make_int2(x, y + 1));
	float4 c11 = getTexel(material, w, h, make_int2(x + 1, y + 1));
	return (1 - dy)*((1 - dx)*c00 + dx*c10)
		+ dy*((1 - dx)*c01 + dx*c11);
}

__device__ void SampleBSDF(Material material, float3 in, float3 nor, float2 uv, float3 dpdu, float3 u, float3& out, float3& fr, float& pdf){
	switch(material.type){
	case MT_LAMBERTIAN:{
		float3 n = nor;
		if (dot(nor, in) < 0)
			n = -n;

		out = CosineHemiSphere(u.x, u.y, n, pdf);
		float3 uu = dpdu, ww;
		ww = cross(uu, n);
		out = ToWorld(out, uu, n, ww);
		fr = make_float3(GetTexel(material, uv)) * ONE_OVER_PI;
		break;
	}
	
	case MT_MIRROR:
		out = 2.f*dot(in, nor)*nor - in;
		fr = material.specular / fabs(dot(out, nor));
		pdf = 1.f;
		break;

	case MT_DIELECTRIC:{
		float3 wi = -in;
		float3 normal = nor;

		float ei = material.outsideIOR, et = material.insideIOR;
		float cosi = dot(wi, normal);
		bool enter = cosi < 0;
		if (!enter){
			float t = ei;
			ei = et;
			et = t;
		}

		float eta = ei / et, cost;
		float sint2 = eta*eta*(1.f - cosi*cosi);
		cost = sqrtf(1.f - sint2 < 0.f ? 0.f : 1.f - sint2);
		float3 rdir = 2.f * dot(-wi, normal) * normal + wi;
		float3 tdir = normalize((wi - normal*cosi)*eta + (enter ? -cost : cost)*normal);
		if (sint2 > 1.f){//total reflection
			out = rdir;
			fr = material.specular / fabs(dot(out, normal));
			pdf = 1.f;
			return;
		}

		float fresnel = DielectricFresnel(fabs(cost), fabs(cosi), et, ei);
		if (u.x > fresnel){//refract
			out = tdir;
			fr = material.specular*eta*eta / fabs(dot(out, normal)) * (1.f - fresnel);
			pdf = 1.f - fresnel;
		}
		else{//reflect
			out = rdir;
			fr = material.specular / fabs(dot(out, normal)) * fresnel;
			pdf = fresnel;
		}
		break;
	}

	case MT_ROUGHCONDUCTOR:{
		float3 n = nor;
		if (dot(nor, in) < 0)
			n = -n;

		float3 wh = SampleGGX(material.alphaU, material.alphaV, u.x, u.y);
		float3 uu = dpdu, ww;
		ww = cross(uu, n);
		wh = ToWorld(wh, uu, n, ww);
		out = 2.f*dot(in, wh)*wh - in;
		if (!SameHemiSphere(in, out, nor)){
			fr = { 0, 0, 0 };
			pdf = 0.f;
			return;
		}

		float cosi = dot(out, wh);
		float3 F = ConductFresnel(fabs(cosi), material.eta, material.k);
		float D = GGX_D(wh, n, dpdu, material.alphaU, material.alphaV);
		float G = GGX_G(in, out, n, wh, dpdu, material.alphaU, material.alphaV);

		fr = material.specular * F * D * G /
			(4.f * fabs(dot(in, n))*fabs(dot(out, n)));
		pdf = D * fabs(dot(wh, n)) / (4.f * fabs(dot(in, wh)));
		break;
	}

	case MT_SUBSTRATE:{
		float3 n = nor;
		if (dot(nor, in) < 0)
			n = -n;
		if (u.x < 0.5){
			float ux = u.x * 2.f;
			out = CosineHemiSphere(ux, u.y, n, pdf);
			float3 uu = dpdu, ww;
			ww = cross(uu, n);
			out = ToWorld(out, uu, n, ww);
		}
		else{
			float ux = (u.x - 0.5f) * 2.f;
			float3 wh = SampleGGX(material.alphaU, material.alphaV, ux, u.y);
			float3 uu = dpdu, ww;
			ww = cross(uu, n);
			wh = ToWorld(wh, uu, n, ww);
			out = 2.f * dot(wh, in) * wh - in;
		}
		if (!SameHemiSphere(in, out, n)){
			fr = { 0.f, 0.f, 0.f };
			pdf = 0.f;
			return;
		}
		float c0 = fabs(dot(in, n));
		float c1 = fabs(dot(out, n));
		float3 Rd = make_float3(GetTexel(material, uv));
		float3 Rs = material.specular;
		float cons0 = 1 - 0.5f * c0;
		float cons1 = 1 - 0.5f * c1;
		/*if (u.x < 0.5f){
			float3 diffuse = (28.f / (23.f * PI)) * Rd * (make_float3(1.f, 1.f, 1.f) - Rs) *
				(1 - cons0*cons0*cons0*cons0*cons0) *
				(1 - cons1*cons1*cons1*cons1*cons1);
			fr = diffuse;
			pdf = fabs(dot(out, n)) * ONE_OVER_PI*0.5f;
		}
		else{
			float3 wh = normalize(in + out);
			float D = GGX_D(wh, n, dpdu, material.alphaU, material.alphaV);
			float3 specular = D /
				(4.f * fabs(dot(out, wh))*Max(c0, c1))*
				SchlickFresnel(Rs, dot(out, wh));

			fr =  specular;
			pdf = 0.5f * (D * fabs(dot(wh, n)) / (4.f * dot(in, wh)));
		}*/
		float3 diffuse = (28.f / (23.f * PI)) * Rd * (make_float3(1.f, 1.f, 1.f) - Rs) *
			(1 - cons0*cons0*cons0*cons0*cons0) *
			(1 - cons1*cons1*cons1*cons1*cons1);
		float3 wh = normalize(in + out);
		float D = GGX_D(wh, n, dpdu, material.alphaU, material.alphaV);
		float3 specular = D /
			(4.f * fabs(dot(out, wh))*Max(c0, c1))*
			SchlickFresnel(Rs, dot(out, wh));

		fr = diffuse + specular;
		pdf = 0.5f * (fabs(dot(out, n)) * ONE_OVER_PI + D * fabs(dot(wh, n)) / (4.f * dot(in, wh)));

		break;
	}

	case MT_ROUGHDIELECTRIC:{
		float3 wi = -in;
		float3 n = nor;
		float3 wh = SampleGGX(material.alphaU, material.alphaV, u.x, u.y);
		float3 uu = dpdu, ww;
		ww = cross(uu, n);
		wh = ToWorld(wh, uu, n, ww);

		float ei = material.outsideIOR, et = material.insideIOR;
		float cosi = dot(wi, n);
		bool enter = cosi < 0;
		if (!enter){
			float t = ei;
			ei = et;
			et = t;
		}

		float D = GGX_D(wh, n, dpdu, material.alphaU, material.alphaV);
		float eta = ei / et, cost;
		cosi = dot(wi, wh);
		float sint2 = eta*eta*(1.f - cosi*cosi);
		cost = sqrtf(1.f - sint2 < 0.f ? 0.f : 1.f - sint2);
		float3 rdir = 2.f * dot(-wi, wh) * wh + wi;
		float3 tdir = normalize((wi - wh*cosi)*eta + (enter ? -cost : cost)*wh);
		if (sint2 > 1.f){//total reflection
			out = rdir;
			float G = GGX_G(in, out, n, wh, dpdu, material.alphaU, material.alphaV);
			fr = material.specular * D * G / (4.f * fabs(dot(in, n)) * fabs(dot(out, n)));
			pdf = D*fabs(dot(wh, n)) / (4.f*fabs(dot(wh, in)));
			return;
		}

		float fresnel = DielectricFresnel(fabs(cost), fabs(cosi), et, ei);
		if (u.z > fresnel){//refract
			out = tdir;
			float G = GGX_G(in, out, n, wh, dpdu, material.alphaU, material.alphaV);
			float c = et*dot(out, wh) + ei*dot(in, wh);
			fr = material.specular*et*et * D * G* (1.f - fresnel) * fabs(dot(in,wh)) * fabs(dot(out,wh)) / 
				(fabs(dot(out, n)) * fabs(dot(in, n)) * c*c);
			pdf = (1.f - fresnel) * D*fabs(dot(wh, n))* et*et*fabs(dot(out, wh)) / (c*c);
		}
		else{//reflect
			out = rdir;
			float G = GGX_G(in, out, n, wh, dpdu, material.alphaU, material.alphaV);
			fr = material.specular * fresnel * D * G / (4.f * fabs(dot(in, n)) * fabs(dot(out, n)));
			pdf = D*fabs(dot(wh, n)) / (4.f*fabs(dot(wh, in))) * fresnel;
		}
		break;
	}
	}
}

//__device__ void Fr(Material material, float3 in, float3 out, float3 nor, float2 uv, float3 dpdu, float u, float3& fr, float& pdf){
__device__ void Fr(Material material, float3 in, float3 out, float3 nor, float2 uv, float3 dpdu, float3& fr, float& pdf){
	switch (material.type){
	case MT_LAMBERTIAN:
		if (!SameHemiSphere(in, out, nor)){
			fr = make_float3(0.f, 0.f, 0.f);
			pdf = 0.f;
			return;
		}

		fr = make_float3(GetTexel(material, uv)) * ONE_OVER_PI;
		pdf = fabs(dot(out, nor)) * ONE_OVER_PI;
		break;

	case MT_MIRROR:
		fr = make_float3(0.f, 0.f, 0.f);
		pdf = 0.f;
		break;

	case MT_DIELECTRIC:
		fr = make_float3(0.f, 0.f, 0.f);
		pdf = 0.f;
		break;

	case MT_ROUGHCONDUCTOR:{
		if (!SameHemiSphere(in, out, nor)){
			fr = { 0, 0, 0 };
			pdf = 0;
			return;
		}
		float3 n = nor;
		if (dot(nor, in) < 0)
			n = -n;

		float3 wh = normalize(in + out);
		float cosi = dot(out, wh);
		float D = GGX_D(wh, n, dpdu, material.alphaU, material.alphaV);
		float G = GGX_G(in, out, n, wh, dpdu, material.alphaU, material.alphaV);
		float3 F = ConductFresnel(fabs(cosi), material.eta, material.k);
		fr = material.specular * F*D*G /
			(4.f*fabs(dot(in, n))*fabs(dot(out, n)));
		pdf = D * fabs(dot(wh, n)) / (4.f * fabs(dot(in, wh)));
		break;
	}

	case MT_SUBSTRATE:{
		if (!SameHemiSphere(in, out, nor)){
			fr = { 0, 0, 0 };
			pdf = 0;
			return;
		}

		float3 n = nor;
		if (dot(nor, in) < 0)
			n = -n;

		float c0 = fabs(dot(in, n));
		float c1 = fabs(dot(out, n));
		float3 Rd = make_float3(GetTexel(material, uv));
		float3 Rs = material.specular;
		float cons0 = 1 - 0.5f * c0;
		float cons1 = 1 - 0.5f * c1;
		float3 wh = normalize(in + out);
		float D = GGX_D(wh, n, dpdu, material.alphaU, material.alphaV);
		/*if (D < 1e-4 || u < 0.5f){
			float3 diffuse = (28.f / (23.f * PI)) * Rd * (make_float3(1.f, 1.f, 1.f) - Rs) *
				(1 - cons0*cons0*cons0*cons0*cons0) *
				(1 - cons1*cons1*cons1*cons1*cons1);
			fr = diffuse;
			pdf = 0.5f*fabs(dot(out, n)) * ONE_OVER_PI;
		}
		else{
			float3 specular = D /
				(4.f * fabs(dot(out, wh))*Max(c0, c1))*
				SchlickFresnel(Rs, dot(out, wh));

			fr =  specular;
			pdf = 0.5f * (D * fabs(dot(wh, n)) / (4.f * dot(in, wh)));
		}*/
		float3 diffuse = (28.f / (23.f * PI)) * Rd * (make_float3(1.f, 1.f, 1.f) - Rs) *
			(1 - cons0*cons0*cons0*cons0*cons0) *
			(1 - cons1*cons1*cons1*cons1*cons1);
		float3 specular = D /
			(4.f * fabs(dot(out, wh))*Max(c0, c1))*
			SchlickFresnel(Rs, dot(out, wh));
		fr = diffuse + specular;
		pdf = 0.5f*(fabs(dot(out, n)) * ONE_OVER_PI + D * fabs(dot(wh, n)) / (4.f * dot(in, wh)));
		break;
	}
					  
	case MT_ROUGHDIELECTRIC:{
		float3 wi = -in;
		float3 n = nor;
		bool reflect = dot(in, n)*dot(out, n)>0;

		float ei = material.outsideIOR, et = material.insideIOR;
		float cosi = dot(wi, n);
		bool enter = cosi < 0;
		if (!enter){
			float t = ei;
			ei = et;
			et = t;
		}

		float3 wh = normalize(-(ei*in + et*out));
		float eta = ei / et, cost;
		cosi = dot(wi, wh);
		float sint2 = eta*eta*(1.f - cosi*cosi);
		cost = sqrtf(1.f - sint2 < 0.f ? 0.f : 1.f - sint2);
		float fresnel = DielectricFresnel(fabs(cost), fabs(cosi), et, ei);
		float D = GGX_D(wh, n, dpdu, material.alphaU, material.alphaV);
		if (!reflect){//refract
			float G = GGX_G(in, out, n, wh, dpdu, material.alphaU, material.alphaV);
			float c = et*dot(out, wh) + ei*dot(in, wh);
			fr = material.specular*et*et * D * G* (1.f - fresnel) * fabs(dot(in, wh)) * fabs(dot(out, wh)) /
				(fabs(dot(out, n)) * fabs(dot(in, n)) * c*c);
			pdf = (1.f - fresnel) * D*fabs(dot(wh, n))* et*et*fabs(dot(out, wh)) / (c*c);
		}
		else{
			float G = GGX_G(in, out, n, wh, dpdu, material.alphaU, material.alphaV);
			fr = material.specular * fresnel * D * G / (4.f * fabs(dot(in, n)) * fabs(dot(out, n)));
			pdf = fresnel * D*fabs(dot(wh, n)) / (4.f*fabs(dot(wh, in)));

		}
		break;
	}
	}
}

__global__ void Tracing(int iter, int maxDepth){
	unsigned x = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned y = blockIdx.y*blockDim.y + threadIdx.y;
	unsigned pixel = x + y*blockDim.x*gridDim.x;

	//init seed
	int threadIndex = (blockIdx.x + blockIdx.y * gridDim.x) * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;

	curandState cudaRNG;
	curand_init(WangHash(iter) + threadIndex, 0, 0, &cudaRNG);

	//start
	float offsetx = curand_uniform(&cudaRNG) - 0.5f;
	float offsety = curand_uniform(&cudaRNG) - 0.5f;
	float unuse;
	float2 aperture = UniformDisk(curand_uniform(&cudaRNG), curand_uniform(&cudaRNG), unuse);//for dof
	Ray ray = kernel_camera->GeneratePrimaryRay(x + offsetx, y + offsety, aperture);
	ray.tmin = kernel_epsilon;

	float3 Li = make_float3(0.f, 0.f, 0.f);
	float3 beta = make_float3(1.f, 1.f, 1.f);
	Ray r = ray;
	Intersection isect;
	bool specular = false;
	for (int bounces = 0; bounces < maxDepth; ++bounces){
		if (!Intersect(r, &isect)){
			//infinity light
			break;
		}

		float3 pos = isect.pos;
		float3 nor = isect.nor;
		float2 uv = isect.uv;
		float3 dpdu = isect.dpdu;
		Material material = kernel_materials[isect.matIdx];
		if (bounces == 0 || specular){
			if (isect.lightIdx != -1){
				Li += beta*kernel_lights[isect.lightIdx].Le(nor, -r.d);
				break;
			}
		}

		if (IsDiffuse(material.type)){
			float3 Ld = make_float3(0.f, 0.f, 0.f);
			float u = curand_uniform(&cudaRNG);
			float choicePdf;
			int idx = LookUpLightDistribution(u, choicePdf);
			Area light = kernel_lights[idx];
			float2 u1 = make_float2(curand_uniform(&cudaRNG), curand_uniform(&cudaRNG));
			float3 radiance, lightNor;
			Ray shadowRay;
			float lightPdf;
			light.SampleLight(pos, u1, radiance, shadowRay, lightNor, lightPdf, kernel_epsilon);

			bool invisible = IntersectP(shadowRay);
			if (!IsBlack(radiance) && !invisible){
				float3 fr;
				float samplePdf;

				//Fr(material, -r.d, shadowRay.d, nor, uv, dpdu, curand_uniform(&cudaRNG), fr, samplePdf);
				Fr(material, -r.d, shadowRay.d, nor, uv, dpdu, fr, samplePdf);
				float weight = PowerHeuristic(1, lightPdf * choicePdf, 1, samplePdf);
				Ld += weight*fr*radiance*fabs(dot(nor, shadowRay.d)) / (lightPdf*choicePdf);
			} 

			float3 uniform = make_float3(curand_uniform(&cudaRNG), curand_uniform(&cudaRNG), curand_uniform(&cudaRNG));
			float3 out, fr;
			float pdf;
			SampleBSDF(material, -r.d, nor, uv, dpdu, uniform, out, fr, pdf);
			if (!(IsBlack(fr) || pdf == 0)){
				Intersection lightIsect;
				Ray lightRay(pos, out, kernel_epsilon);
				if (Intersect(lightRay, &lightIsect)){
					float3 p = lightIsect.pos;
					float3 n = lightIsect.nor;
					float3 radiance = { 0.f, 0.f, 0.f };
					if (lightIsect.lightIdx != -1)
						radiance = kernel_lights[lightIsect.lightIdx].Le(n, -lightRay.d);
					if (!IsBlack(radiance)){
						float pdfA, pdfW;
						kernel_lights[lightIsect.lightIdx].Pdf(Ray(p, -out, kernel_epsilon), n, pdfA, pdfW);
						float choicePdf = PdfFromLightDistribution(lightIsect.lightIdx);
						float lenSquare = dot(p - pos, p - pos);
						float costheta = fabs(dot(n, lightRay.d));
						float lPdf = pdfA * lenSquare / (costheta);
						float weight = PowerHeuristic(1, pdf, 1, lPdf * choicePdf);
						Ld += weight * fr * radiance * fabs(dot(out, nor)) / pdf;
					}
				}
			}

			Li += beta*Ld;
		}

		float3 u = make_float3(curand_uniform(&cudaRNG), curand_uniform(&cudaRNG), curand_uniform(&cudaRNG));
		float3 out, fr;
		float pdf;

		SampleBSDF(material, -r.d, nor, uv, dpdu, u, out, fr, pdf);
		if (IsBlack(fr))
			break;

		beta *= fr*fabs(dot(nor, out)) / pdf;
		specular = !IsDiffuse(material.type);

		if (bounces > 3){
			float illumate = clamp(1.f - Luminance(beta), 0.f, 1.f);
			if (curand_uniform(&cudaRNG) < illumate)
				break;

			beta /= (1 - illumate);
		}

		r = Ray(pos, out, kernel_epsilon);
	}

	if (!(isnan(Li.x) || isnan(Li.y) || isnan(Li.z)))
		kernel_color[pixel] = Li;
}

__global__ void Output(int iter, float3* output, bool reset, bool filmic){
	unsigned x = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned y = blockIdx.y*blockDim.y + threadIdx.y;
	unsigned pixel = x + y*blockDim.x*gridDim.x;

	if (reset){
		kernel_acc_image[pixel] = { 0, 0, 0 };
	}
	float3 color = kernel_color[pixel];
	kernel_acc_image[pixel] += color;

	color = kernel_acc_image[pixel] / iter;
	if (filmic)
		FilmicTonemapping(color);
	else
		GammaCorrection(color);
	output[pixel] = color;
}

__global__ void InitRender(
	Camera* camera,
	LinearBVHNode* bvh_nodes,
	Primitive* primitives,
	Material* materials,
	Area* lights,
	uchar4** texs,
	float* light_distribution,
	int ld_size,
	int* tex_size,
	float3* image, 
	float3* color,
	float ep,
	int hdr_w, 
	int hdr_h, 
	bool isvalid){
	kernel_camera = camera;
	kernel_linear = bvh_nodes;
	kernel_primitives = primitives;
	kernel_materials = materials;
	kernel_lights = lights;
	kernel_textures = texs;
	kernel_light_distribution = light_distribution;
	kernel_light_distribution_size = ld_size;
	kernel_texture_size = tex_size;
	kernel_acc_image = image;
	kernel_color = color;
	kernel_epsilon = ep;
	kernel_hdr_width = hdr_w;
	kernel_hdr_height = hdr_h;
	kernel_hdr_isvalid = isvalid;
}

void BeginRender(
	Scene& scene,
	BVH& bvh,
	Camera cam,
	unsigned width,
	unsigned height,
	float ep,
	int max_depth, 
	HDRMap& hdrmap){
	int mesh_memory_use = 0;
	int material_memory_use = 0;
	int bvh_memory_use = 0;
	int light_memory_use = 0;
	int texture_memory_use = 0;
	maxDepth = max_depth;
	int num_primitives = bvh.prims.size();
	HANDLE_ERROR(cudaMalloc(&dev_camera, sizeof(Camera)));
	HANDLE_ERROR(cudaMemcpy(dev_camera, &cam, sizeof(Camera), cudaMemcpyHostToDevice));

	if (num_primitives){
		HANDLE_ERROR(cudaMalloc(&dev_primitives, num_primitives*sizeof(Primitive)));
		HANDLE_ERROR(cudaMemcpy(dev_primitives, &bvh.prims[0], num_primitives*sizeof(Primitive), cudaMemcpyHostToDevice));
		mesh_memory_use += num_primitives*sizeof(Primitive);
	}
	if (bvh.total_nodes > 0){
		HANDLE_ERROR(cudaMalloc(&dev_bvh_nodes, bvh.total_nodes*sizeof(LinearBVHNode)));
		HANDLE_ERROR(cudaMemcpy(dev_bvh_nodes, bvh.linear_root, bvh.total_nodes*sizeof(LinearBVHNode), cudaMemcpyHostToDevice));
		bvh_memory_use += bvh.total_nodes*sizeof(LinearBVHNode);
	}

	//copy material
	int num_materials = scene.materials.size();
	HANDLE_ERROR(cudaMalloc(&dev_materials, num_materials*sizeof(Material)));
	HANDLE_ERROR(cudaMemcpy(dev_materials, &scene.materials[0], num_materials*sizeof(Material), cudaMemcpyHostToDevice));
	material_memory_use += num_materials*sizeof(Material);

	int num_lights = scene.lights.size();
	HANDLE_ERROR(cudaMalloc(&dev_lights, num_lights*sizeof(Area)));
	HANDLE_ERROR(cudaMemcpy(dev_lights, &scene.lights[0], num_lights*sizeof(Area), cudaMemcpyHostToDevice));
	light_memory_use+= num_lights*sizeof(Area);

	//copy textures
	if (scene.textures.size()){
		HANDLE_ERROR(cudaMalloc(&texture_size, scene.textures.size() * 2 * sizeof(int)));
		vector<int> texSize;
		HANDLE_ERROR(cudaMalloc(&dev_textures, scene.textures.size()*sizeof(uchar4*)));
		for (int i = 0; i < scene.textures.size(); ++i){
			Texture tex = scene.textures[i];
			texSize.push_back(tex.width);
			texSize.push_back(tex.height);
			uchar4* t;
			HANDLE_ERROR(cudaMalloc(&t, tex.width*tex.height*sizeof(uchar4)));
			texture_memory_use += tex.width*tex.height*sizeof(uchar4);
			HANDLE_ERROR(cudaMemcpy(t, &tex.data[0], tex.width*tex.height*sizeof(uchar4), cudaMemcpyHostToDevice));
			HANDLE_ERROR(cudaMemcpy(&dev_textures[i], &t, sizeof(uchar4*), cudaMemcpyHostToDevice));
		}
		HANDLE_ERROR(cudaMemcpy(texture_size, &texSize[0], scene.textures.size() * 2 * sizeof(int), cudaMemcpyHostToDevice));
	}

	int num_pixel = width*height;
	HANDLE_ERROR(cudaMalloc(&dev_image, num_pixel*sizeof(float3)));
	texture_memory_use += num_pixel*sizeof(float3);
	HANDLE_ERROR(cudaMalloc(&dev_color, num_pixel*sizeof(float3)));
	texture_memory_use += num_pixel*sizeof(float3);
	if (hdrmap.isvalid){
		HANDLE_ERROR(cudaMalloc(&hdr_map, hdrmap.width*hdrmap.height*sizeof(float4)));
		texture_memory_use += num_pixel*sizeof(float4);
		HANDLE_ERROR(cudaMemcpy(hdr_map, hdrmap.image, hdrmap.width*hdrmap.height*sizeof(float4), cudaMemcpyHostToDevice));
		hdr_texture.filterMode = cudaFilterModeLinear;
		cudaChannelFormatDesc channel4desc = cudaCreateChannelDesc<float4>();
		HANDLE_ERROR(cudaBindTexture(NULL, &hdr_texture, hdr_map, &channel4desc, hdrmap.width*hdrmap.height*sizeof(float4)));
	}

	int ld_size = scene.lightDistribution.size();
	HANDLE_ERROR(cudaMalloc(&dev_light_distribution, ld_size*sizeof(float)));
	HANDLE_ERROR(cudaMemcpy(dev_light_distribution, &scene.lightDistribution[0], ld_size*sizeof(float), cudaMemcpyHostToDevice));
	texture_memory_use += ld_size*sizeof(float);
	
	InitRender << <1, 1 >> >(dev_camera, dev_bvh_nodes,
		dev_primitives, dev_materials, dev_lights, dev_textures, dev_light_distribution, ld_size,
		texture_size, dev_image, dev_color, ep, hdrmap.width, hdrmap.height, hdrmap.isvalid);

	HANDLE_ERROR(cudaDeviceSynchronize());

	fprintf(stderr, "\n\nMesh video memory use:[%.3fM]\n", (float)mesh_memory_use / (1024 * 1024));
	fprintf(stderr, "Bvh video memory use:[%.3fM]\n", (float)bvh_memory_use / (1024 * 1024));
	fprintf(stderr, "Material video memory use:[%.3fM]\n", (float)material_memory_use / (1024 * 1024));
	fprintf(stderr, "Light video memory use:[%.3fM]\n", (float)light_memory_use / (1024 * 1024));
	fprintf(stderr, "Texture video memory use:[%.2fM]\n", (float)texture_memory_use / (1024 * 1024));
	fprintf(stderr, "Total video memory use:[%.3fM]\n", (float)(mesh_memory_use + bvh_memory_use + material_memory_use + light_memory_use + texture_memory_use) / (1024 * 1024));
}

void EndRender(){
	HANDLE_ERROR(cudaFree(dev_primitives));
	HANDLE_ERROR(cudaFree(dev_bvh_nodes));

	HANDLE_ERROR(cudaFree(dev_image));
	HANDLE_ERROR(cudaFree(dev_color));
	HANDLE_ERROR(cudaFree(hdr_map));
	HANDLE_ERROR(cudaUnbindTexture(hdr_texture));
}

void Render(Scene& scene, unsigned width, unsigned height, Camera* camera, unsigned iter, bool reset, float3* output){
	HANDLE_ERROR(cudaMemcpy(dev_camera, camera, sizeof(Camera), cudaMemcpyHostToDevice));
	int block_x = 32, block_y = 4;
	dim3 block(block_x, block_y);
	dim3 grid(width / block.x, height / block.y);

	Tracing << <grid, block >> >(iter, maxDepth);

	grid.x = width / block.x;
	grid.y = height / block.y;
	Output << <grid, block >> >(iter, output, reset, camera->filmic);
}
