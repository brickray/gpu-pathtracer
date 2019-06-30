#include "pathtracer.h"
#include "camera.h"
#include "scene.h"
#include "bvh.h"
#include "device_launch_parameters.h"
#include <thrust/random.h>

Camera* dev_camera;
float3* dev_image, *dev_color;
LinearBVHNode* dev_bvh_nodes;
Primitive* dev_primitives;
Material* dev_materials;
Bssrdf* dev_bssrdfs;
Medium* dev_mediums;
Area* dev_lights;
Infinite* dev_infinite;
float* dev_light_distribution;
uchar4** dev_textures;
int* texture_size;//0 1为第一张图的长宽， 2 3为第二张图的长宽，以此类推

__device__ Camera* kernel_camera;
__device__ int kernel_hdr_width, kernel_hdr_height;
__device__ float3* kernel_acc_image, *kernel_color;
__device__ LinearBVHNode* kernel_linear;
__device__ Primitive* kernel_primitives;
__device__ Material* kernel_materials;
__device__ Bssrdf* kernel_bssrdfs;
__device__ Medium* kernel_mediums;
__device__ Area* kernel_lights;
__device__ Infinite* kernel_infinite;
__device__ uchar4** kernel_textures;
__device__ int* kernel_texture_size;
__device__ float* kernel_light_distribution;
__device__ int kernel_light_size;
__device__ int kernel_light_distribution_size;
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

__host__ __device__ inline float DielectricFresnel(float cosi, float cost, const float& etai, const float& etat){
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

__host__ __device__ inline float3 Reflect(float3 in, float3 nor){
	return 2.f*dot(in, nor)*nor - in;
}

__host__ __device__ inline float3 Refract(float3 in, float3 nor, float etai, float etat){
	float cosi = dot(in, nor);
	bool enter = cosi > 0;
	if (!enter){
		float t = etai;
		etai = etat;
		etat = t;
	}

	float eta = etai / etat;
	float sini2 = 1.f - cosi*cosi;
	float sint2 = sini2*eta*eta;
	float cost = sqrtf(1.f - sint2);
	return normalize((nor*cosi-in)*eta + (enter ? -cost : cost)*nor);
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
	c = fmaxf(make_float3(0, 0, 0), c);
	c = (c*(6.2f*c + 0.5f)) / (c*(6.2f*c + 1.7f) + 0.06f);
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
					else if(prim.type == GT_LINES){
						if (prim.line.Intersect(ray, isect))
							ret = true;
					}
					else if (prim.type == GT_SPHERE){
						if (prim.sphere.Intersect(ray, isect))
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
					else if (prim.type == GT_LINES){
						if (prim.line.Intersect(ray, nullptr))
							return true;
					}
					else if (prim.type == GT_SPHERE){
						if (prim.sphere.Intersect(ray, nullptr))
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

__device__ float3 Tr(Ray& ray, thrust::uniform_real_distribution<float>& uniform, thrust::default_random_engine& rng){
	float3 tr = make_float3(1, 1, 1);
	float tmax = ray.tmax;
	while (true){
		Intersection isect;
		bool invisible = Intersect(ray, &isect);
		if (invisible && isect.matIdx != -1)
			return{ 0, 0, 0 };

		if (ray.medium){
			if (ray.medium->type == MT_HOMOGENEOUS)
				tr *= ray.medium->homogeneous.Tr(ray, uniform, rng);
			else
				tr *= ray.medium->heterogeneous.Tr(ray, uniform, rng);
		}

		if (!invisible) break;
		Medium* m = dot(ray.d, isect.nor) > 0 ? (isect.mediumOutside == -1 ? nullptr : &kernel_mediums[isect.mediumOutside])
			: (isect.mediumInside == -1 ? nullptr : &kernel_mediums[isect.mediumInside]);
		tmax -= ray.tmax;
		ray = Ray(ray(ray.tmax), ray.d, m, kernel_epsilon, tmax);
	}

	return tr;
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

//**************************bssrdf*****************
__device__ float3 SingleScatter(Intersection* isect, float3 in, thrust::uniform_real_distribution<float>& uniform, thrust::default_random_engine& rng){
	float3 pos = isect->pos;
	float3 nor = isect->nor;
	float coso = fabs(dot(in, nor));
	Bssrdf bssrdf = kernel_bssrdfs[isect->bssrdf];
	float eta = bssrdf.eta;
	float sino2 = 1.f - coso*coso;
	float cosi = sqrtf(1.f - sino2 / (eta*eta));
	float fresnel = 1.f - DielectricFresnel(coso, cosi, 1.f, eta);
	float sigmaTr = Luminance(bssrdf.GetSigmaTr());
	float3 sigmaS = bssrdf.GetSigmaS();
	float3 sigmaT = bssrdf.GetSigmaT();
	float3 rdir = Reflect(in, nor);
	float3 tdir = Refract(in, nor, 1.f, eta);
	float3 L = { 0, 0, 0 };
	Intersection rIsect;
	if (Intersect(Ray(pos,rdir,nullptr,kernel_epsilon), &rIsect)){
		if (rIsect.lightIdx != -1){
			L += (1.f - fresnel)*kernel_lights[rIsect.lightIdx].Le(rIsect.nor, -rdir);
		}
	}
	Intersection tIsect;
	Intersect(Ray(pos, tdir, nullptr, kernel_hdr_height), &tIsect);
	float len = length(tIsect.pos - pos);
	int samples = 1;
	for (int i = 0; i < samples; ++i){
		float d = Exponential(uniform(rng), sigmaTr);
		if (d > len) continue;
		float3 pSample = pos + tdir*d;
		float pdf = ExponentialPdf(d, sigmaTr);
		float choicePdf;
		float u = uniform(rng);
		int idx = LookUpLightDistribution(u, choicePdf);
		Area light = kernel_lights[idx];
		float lightPdf;
		Ray shadowRay;
		float3 radiance, lightNor;
		float2 u1 = make_float2(uniform(rng), uniform(rng));
		light.SampleLight(pSample, u1, radiance, shadowRay, lightNor, lightPdf, kernel_epsilon);
		if (IsBlack(radiance))
			continue;

		float tmax = shadowRay.tmax;
		Intersection wiIsect;
		if (Intersect(shadowRay, &wiIsect)){
			if (wiIsect.bssrdf == isect->bssrdf){
				float3 wiPos = wiIsect.pos;
				float3 wiNor = wiIsect.nor;
				shadowRay.tmin += shadowRay.tmax;
				shadowRay.tmax = tmax;
				if (!IntersectP(shadowRay)){
					float p = bssrdf.GetPhase();
					float cosi = fabs(dot(wiNor, shadowRay.d));
					float sini2 = 1.f - cosi*cosi;
					float coso = sqrtf(1.f - sini2 / (eta*eta));
					float fresnelI = 1.f - DielectricFresnel(cosi, coso, 1.f, eta);
					float G = fabs(dot(wiNor, tdir)) / cosi;
					float3 sigmaTC = sigmaT*(1.f + G);
					float di = length(wiPos - pSample);
					float et = 1.f / eta;
					float diPrime = di*fabs(dot(shadowRay.d, wiNor)) /
						sqrt(1.f - et*et*(1.f - cosi*cosi));
					L += (fresnel*fresnelI*p*sigmaS / sigmaTC)*
						Exp(-diPrime*sigmaT)*
						Exp(-d*sigmaT)*radiance / (lightPdf*choicePdf*pdf);
				}
			}
		}
	}

	L /= samples;
	return L;
}

__device__ float3 MultipleScatter(Intersection* isect, float3 in, thrust::uniform_real_distribution<float>& uniform, thrust::default_random_engine& rng){
	float3 pos = isect->pos;
	float3 nor = isect->nor;
	float coso = fabs(dot(in, nor));
	Bssrdf bssrdf = kernel_bssrdfs[isect->bssrdf];
	float eta = bssrdf.eta;
	float sino2 = 1.f - coso*coso;
	float cosi = sqrtf(1.f - sino2 / (eta*eta));
	float fresnel = 1.f - DielectricFresnel(coso, cosi, 1.f, eta);
	float sigmaTr = Luminance(bssrdf.GetSigmaTr());
	float skipRatio = 0.01f;
	float rMax = sqrt(log(skipRatio) / -sigmaTr);
	float3 L = { 0, 0, 0 };
	int samples = 1;
	for (int i = 0; i < samples; ++i){
		Ray probeRay;
		float pdf;
		float2 u = make_float2(uniform(rng), uniform(rng));
		bssrdf.SampleProbeRay(pos, nor, u, sigmaTr, rMax, probeRay, pdf);
		probeRay.tmin = kernel_epsilon;

		Intersection probeIsect;
		if (Intersect(probeRay, &probeIsect)){
			if (isect->bssrdf == probeIsect.bssrdf){
				float3 probePos = probeIsect.pos;
				float3 probeNor = probeIsect.nor;
				float3 rd = bssrdf.Rd(dot(probePos - pos, probePos - pos));
				float choicePdf;
				float u = uniform(rng);
				int idx = LookUpLightDistribution(u, choicePdf);
				Area light = kernel_lights[idx];
				float lightPdf;
				float2 u1 = make_float2(uniform(rng), uniform(rng));
				float3 radiance, lightNor;
				Ray shadowRay;
				light.SampleLight(probePos, u1, radiance, shadowRay, lightNor, lightPdf, kernel_epsilon);
				if (!IsBlack(radiance) && !IntersectP(shadowRay)){
					float cosi = fabs(dot(shadowRay.d, probeNor));
					float sini2 = 1.f - cosi*cosi;
					float cost = sqrtf(1.f - sini2 / (eta*eta));
					float3 irradiance = radiance*cosi / (lightPdf*choicePdf);
					float fresnelI = 1.f - DielectricFresnel(cosi, cost, 1.f, eta);
					pdf *= fabs(dot(probeRay.d, probeNor));
					L += (ONE_OVER_PI*fresnel*fresnelI*rd*irradiance) / pdf;
				}
			}
		}

		L /= samples;
		return L;
	}
}
//**************************bssrdf end*************

//**************************BSDF Sampling**************************
__device__ void SampleBSDF(Material material, float3 in, float3 nor, float2 uv, float3 dpdu, float3 u, float3& out, float3& fr, float& pdf, TransportMode mode = TransportMode::Radiance){
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
		out = Reflect(in, nor);
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
		float3 rdir = Reflect(-wi, normal);
		float3 tdir = Refract(in, nor, material.outsideIOR, material.insideIOR);
		if (sint2 > 1.f){//total reflection
			out = rdir;
			fr = material.specular / fabs(dot(out, normal));
			pdf = 1.f;
			return;
		}

		float fresnel = DielectricFresnel(fabs(cost), fabs(cosi), et, ei);
		if (u.x > fresnel){//refract
			out = tdir;
			fr = material.specular / fabs(dot(out, normal)) * (1.f - fresnel);
			if (mode == TransportMode::Radiance)
				fr *= eta*eta;
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
		out = Reflect(in, wh);
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
			out = Reflect(in, wh);
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
		float3 rdir = Reflect(-wi, wh);
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
			fr = material.specular*ei*ei * D * G* (1.f - fresnel) * fabs(dot(in, wh)) * fabs(dot(out, wh)) /
				(fabs(dot(out, n)) * fabs(dot(in, n)) * c*c);
			if (mode == TransportMode::Radiance)
				fr *= (1.f / (eta*eta));
			
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
__device__ void Fr(Material material, float3 in, float3 out, float3 nor, float2 uv, float3 dpdu, float3& fr, float& pdf, TransportMode mode = TransportMode::Radiance){
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
			fr = material.specular*ei*ei * D * G* (1.f - fresnel) * fabs(dot(in, wh)) * fabs(dot(out, wh)) /
				(fabs(dot(out, n)) * fabs(dot(in, n)) * c*c);
			if (mode == TransportMode::Radiance)
				fr *= (1.f / (eta*eta));
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
//**************************BSDF End*******************************

//**************************AO Integrator**************************
__global__ void Ao(int iter, float maxDist){
	unsigned x = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned y = blockIdx.y*blockDim.y + threadIdx.y;
	unsigned pixel = x + y*blockDim.x*gridDim.x;

	//init seed
	int threadIndex = (blockIdx.x + blockIdx.y * gridDim.x) * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;

	thrust::default_random_engine rng(WangHash(pixel) + WangHash(iter));
	thrust::uniform_real_distribution<float> uniform(0.0f, 1.0f);

	//start
	float offsetx = uniform(rng) - 0.5f;
	float offsety = uniform(rng) - 0.5f;
	float unuse;
	float2 aperture = UniformDisk(uniform(rng), uniform(rng), unuse);//for dof
	Ray ray = kernel_camera->GeneratePrimaryRay(x + offsetx, y + offsety, aperture);
	ray.tmin = kernel_epsilon;

	float3 L = { 0.f, 0.f, 0.f };
	Intersection isect;
	bool intersect = Intersect(ray, &isect);
	if (!intersect){
		kernel_color[pixel] = { 0, 0, 0 };
		return;
	}

	float3 pos = isect.pos;
	float3 nor = isect.nor;
	float pdf = 0.f;
	if (dot(-ray.d, nor) < 0.f)
		nor = -nor;
	float3 dir = CosineHemiSphere(uniform(rng), uniform(rng), nor, pdf);
	float3 uu = isect.dpdu, ww;
	ww = cross(uu, nor);
	dir = ToWorld(dir, uu, nor, ww);
	float cosine = dot(dir, nor);
	Ray r(pos, dir, nullptr, kernel_epsilon, maxDist);
	intersect = IntersectP(r);
	if (!intersect){
		float v = cosine*ONE_OVER_PI / pdf;
		L += make_float3(v, v, v);
	}

	if (!IsNan(L))
		kernel_color[pixel] = L;
}
//**************************AO End*********************************

//**************************Path Integrator************************
__global__ void Path(int iter, int maxDepth){
	unsigned x = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned y = blockIdx.y*blockDim.y + threadIdx.y;
	unsigned pixel = x + y*blockDim.x*gridDim.x;

	//init seed
	int threadIndex = (blockIdx.x + blockIdx.y * gridDim.x) * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;

	thrust::default_random_engine rng(WangHash(pixel) + WangHash(iter));
	thrust::uniform_real_distribution<float> uniform(0.0f, 1.0f);

	//start
	float offsetx = uniform(rng) - 0.5f;
	float offsety = uniform(rng) - 0.5f;
	float unuse;
	float2 aperture = UniformDisk(uniform(rng), uniform(rng), unuse);//for dof
	Ray ray = kernel_camera->GeneratePrimaryRay(x + offsetx, y + offsety, aperture);
	ray.tmin = kernel_epsilon;
	
	float3 Li = make_float3(0.f, 0.f, 0.f);
	float3 beta = make_float3(1.f, 1.f, 1.f);
	Ray r = ray;
	Intersection isect;
	bool specular = false;
	for (int bounces = 0; bounces < maxDepth; ++bounces){
		if (!Intersect(r, &isect)){
			if ((bounces == 0 || specular) && kernel_infinite->isvalid)
				Li += beta*kernel_infinite->Le(r.d);
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

		//direct light with multiple importance sampling
		if (!IsDelta(material.type)){
			float3 Ld = make_float3(0.f, 0.f, 0.f);
			bool inf = false;
			float u = uniform(rng);
			float choicePdf;
			int idx = LookUpLightDistribution(u, choicePdf);
			if (idx == kernel_light_size) inf = true;
			float2 u1 = make_float2(uniform(rng), uniform(rng));
			float3 radiance, lightNor;
			Ray shadowRay;
			float lightPdf;
			if (!inf)
				kernel_lights[idx].SampleLight(pos, u1, radiance, shadowRay, lightNor, lightPdf, kernel_epsilon);
			else
				kernel_infinite->SampleLight(pos, u1, radiance, shadowRay, lightNor, lightPdf, kernel_epsilon);
			shadowRay.medium = r.medium;

			if (!IsBlack(radiance) && !IntersectP(shadowRay)){
				float3 fr;
				float samplePdf;

				//Fr(material, -r.d, shadowRay.d, nor, uv, dpdu, uniform(rng), fr, samplePdf);
				Fr(material, -r.d, shadowRay.d, nor, uv, dpdu, fr, samplePdf);

				float weight = PowerHeuristic(1, lightPdf * choicePdf, 1, samplePdf);
				Ld += weight*fr*radiance*fabs(dot(nor, shadowRay.d)) / (lightPdf*choicePdf);
			}

			float3 us = make_float3(uniform(rng), uniform(rng), uniform(rng));
			float3 out, fr;
			float pdf;
			SampleBSDF(material, -r.d, nor, uv, dpdu, us, out, fr, pdf);
			if (!(IsBlack(fr) || pdf == 0)){
				Intersection lightIsect;
				Ray lightRay(pos, out, r.medium, kernel_epsilon);
				if (Intersect(lightRay, &lightIsect)){
					float3 p = lightIsect.pos;
					float3 n = lightIsect.nor;
					float3 radiance = { 0.f, 0.f, 0.f };
					if (lightIsect.lightIdx != -1)
						radiance = kernel_lights[lightIsect.lightIdx].Le(n, -lightRay.d);
					if (!IsBlack(radiance)){
						float pdfA, pdfW;
						kernel_lights[lightIsect.lightIdx].Pdf(Ray(p, -out, r.medium, kernel_epsilon), n, pdfA, pdfW);
						float choicePdf = PdfFromLightDistribution(lightIsect.lightIdx);
						float lenSquare = dot(p - pos, p - pos);
						float costheta = fabs(dot(n, lightRay.d));
						float lPdf = pdfA * lenSquare / (costheta);
						float weight = PowerHeuristic(1, pdf, 1, lPdf * choicePdf);
						
						Ld += weight * fr * radiance * fabs(dot(out, nor)) / pdf;
					}
				}
				else{
					//infinite
					if (kernel_infinite->isvalid){
						float3 radiance = { 0.f, 0.f, 0.f };
						radiance = kernel_infinite->Le(lightRay.d);
						float choicePdf = PdfFromLightDistribution(kernel_light_size);
						float lightPdf, pdfA;
						float3 lightNor;
						kernel_infinite->Pdf(lightRay, lightNor, pdfA, lightPdf);
						float weight = PowerHeuristic(1, pdf, 1, lightPdf*choicePdf);
						
						Ld += weight * fr * radiance * fabs(dot(out, nor)) / pdf;
					}
				}
			}

			Li += beta*Ld;
		}

		float3 u = make_float3(uniform(rng), uniform(rng), uniform(rng));
		float3 out, fr;
		float pdf;

		SampleBSDF(material, -r.d, nor, uv, dpdu, u, out, fr, pdf);
		if (IsBlack(fr))
			break;

		beta *= fr*fabs(dot(nor, out)) / pdf;
		specular = IsDelta(material.type);

		r = Ray(pos, out, nullptr, kernel_epsilon);

		if (bounces > 3){
			float illumate = clamp(1.f - Luminance(beta), 0.f, 1.f);
			if (uniform(rng) < illumate)
				break;

			beta /= (1 - illumate);
		}
	}

	if (!IsInf(Li) && !IsNan(Li))
			kernel_color[pixel] = Li;
}
//**************************Path End*******************************

//**************************VolPath Integrator*********************
__global__ void Volpath(int iter, int maxDepth){
	unsigned x = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned y = blockIdx.y*blockDim.y + threadIdx.y;
	unsigned pixel = x + y*blockDim.x*gridDim.x;

	//init seed
	int threadIndex = (blockIdx.x + blockIdx.y * gridDim.x) * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;

	thrust::default_random_engine rng(WangHash(pixel) + WangHash(iter));
	thrust::uniform_real_distribution<float> uniform(0.0f, 1.0f);

	//start
	float offsetx = uniform(rng) - 0.5f;
	float offsety = uniform(rng) - 0.5f;
	float unuse;
	float2 aperture = UniformDisk(uniform(rng), uniform(rng), unuse);//for dof
	Ray ray = kernel_camera->GeneratePrimaryRay(x + offsetx, y + offsety, aperture);
	ray.tmin = kernel_epsilon;
	ray.medium = kernel_camera->medium == -1 ? nullptr : &kernel_mediums[kernel_camera->medium];

	float3 Li = make_float3(0.f, 0.f, 0.f);
	float3 beta = make_float3(1.f, 1.f, 1.f);
	Ray r = ray;
	Intersection isect;
	bool specular = false;
	for (int bounces = 0; bounces < maxDepth; ++bounces){
		if (!Intersect(r, &isect)){
			if ((bounces == 0 || specular) && kernel_infinite->isvalid)
				Li += beta*kernel_infinite->Le(r.d);
			break;
		}

		float3 pos = isect.pos;
		float3 nor = isect.nor;
		float2 uv = isect.uv;
		float3 dpdu = isect.dpdu;

		float sampledDist;
		bool sampledMedium = false;
		if (r.medium){
			if (r.medium->type == MT_HOMOGENEOUS)
				beta *= r.medium->homogeneous.Sample(r, uniform, rng, sampledDist, sampledMedium);
			else
				beta *= r.medium->heterogeneous.Sample(r, uniform, rng, sampledDist, sampledMedium);
		}
		if (IsBlack(beta)) break;
		if (sampledMedium){
			//TODO:多重重要性采样
			bool inf = false;
			float u = uniform(rng);
			float choicePdf;
			int idx = LookUpLightDistribution(u, choicePdf);
			if (idx == kernel_light_size) inf = true;
			float3 samplePos = r(sampledDist);
			float2 u1 = make_float2(uniform(rng), uniform(rng));
			float3 radiance, lightNor;
			Ray shadowRay;
			float lightPdf;
			if (!inf)
				kernel_lights[idx].SampleLight(samplePos, u1, radiance, shadowRay, lightNor, lightPdf, kernel_epsilon);
			else
				kernel_infinite->SampleLight(samplePos, u1, radiance, shadowRay, lightNor, lightPdf, kernel_epsilon);
			shadowRay.medium = r.medium;
			float3 tr = Tr(shadowRay, uniform, rng);
			float phase, unuse;
			r.medium->Phase(-r.d, shadowRay.d, phase, unuse);

			if (!IsBlack(radiance))
				Li += tr*beta*phase*radiance / (lightPdf * choicePdf);

			float pdf;
			float2 phaseU = make_float2(uniform(rng), uniform(rng));
			float3 dir;
			r.medium->SamplePhase(phaseU, dir, phase, pdf);
			r = Ray(samplePos, dir, r.medium, kernel_epsilon);
			specular = false;
		}
		else{
			if (bounces == 0 || specular){
				if (isect.lightIdx != -1){
					float3 tr = { 1.f, 1.f, 1.f };
					if (r.medium){
						if (r.medium->type == MT_HOMOGENEOUS)
							tr = r.medium->homogeneous.Tr(r, uniform, rng);
						else
							tr = r.medium->heterogeneous.Tr(r, uniform, rng);
					}
					Li += tr*beta*kernel_lights[isect.lightIdx].Le(nor, -r.d);
					break;
				}
			}

			if (isect.matIdx == -1){
				bounces--;
				Medium* m = dot(r.d, isect.nor) > 0 ? (isect.mediumOutside == -1 ? nullptr : &kernel_mediums[isect.mediumOutside])
					: (isect.mediumInside == -1 ? nullptr : &kernel_mediums[isect.mediumInside]);
				r = Ray(pos, r.d, m, kernel_epsilon);

				continue;
			}

			Material material = kernel_materials[isect.matIdx];
			//direct light with multiple importance sampling
			if (!IsDelta(material.type)){
				float3 Ld = make_float3(0.f, 0.f, 0.f);
				bool inf = false;
				float u = uniform(rng);
				float choicePdf;
				int idx = LookUpLightDistribution(u, choicePdf);
				if (idx == kernel_light_size) inf = true;
				float2 u1 = make_float2(uniform(rng), uniform(rng));
				float3 radiance, lightNor;
				Ray shadowRay;
				float lightPdf;
				if (!inf)
					kernel_lights[idx].SampleLight(pos, u1, radiance, shadowRay, lightNor, lightPdf, kernel_epsilon);
				else
					kernel_infinite->SampleLight(pos, u1, radiance, shadowRay, lightNor, lightPdf, kernel_epsilon);
				shadowRay.medium = r.medium;

				if (!IsBlack(radiance)){
					float3 fr;
					float samplePdf;

					//Fr(material, -r.d, shadowRay.d, nor, uv, dpdu, uniform(rng), fr, samplePdf);
					Fr(material, -r.d, shadowRay.d, nor, uv, dpdu, fr, samplePdf);
					float3 tr = Tr(shadowRay, uniform, rng);

					float weight = PowerHeuristic(1, lightPdf * choicePdf, 1, samplePdf);
					Ld += weight*tr*fr*radiance*fabs(dot(nor, shadowRay.d)) / (lightPdf*choicePdf);
				}

				float3 us = make_float3(uniform(rng), uniform(rng), uniform(rng));
				float3 out, fr;
				float pdf;
				SampleBSDF(material, -r.d, nor, uv, dpdu, us, out, fr, pdf);
				if (!(IsBlack(fr) || pdf == 0)){
					Intersection lightIsect;
					Ray lightRay(pos, out, r.medium, kernel_epsilon);
					if (Intersect(lightRay, &lightIsect)){
						float3 p = lightIsect.pos;
						float3 n = lightIsect.nor;
						float3 radiance = { 0.f, 0.f, 0.f };
						if (lightIsect.lightIdx != -1)
							radiance = kernel_lights[lightIsect.lightIdx].Le(n, -lightRay.d);
						if (!IsBlack(radiance)){
							float pdfA, pdfW;
							kernel_lights[lightIsect.lightIdx].Pdf(Ray(p, -out, r.medium, kernel_epsilon), n, pdfA, pdfW);
							float choicePdf = PdfFromLightDistribution(lightIsect.lightIdx);
							float lenSquare = dot(p - pos, p - pos);
							float costheta = fabs(dot(n, lightRay.d));
							float lPdf = pdfA * lenSquare / (costheta);
							float weight = PowerHeuristic(1, pdf, 1, lPdf * choicePdf);
							float3 tr = { 1.f, 1.f, 1.f };
							if (lightRay.medium){
								if (lightRay.medium->type == MT_HOMOGENEOUS)
									tr = lightRay.medium->homogeneous.Tr(lightRay, uniform, rng);
								else
									tr = lightRay.medium->heterogeneous.Tr(lightRay, uniform, rng);
							}
							Ld += weight * tr * fr * radiance * fabs(dot(out, nor)) / pdf;
						}
					}
					else{
						//infinite
						if (kernel_infinite->isvalid){
							float3 radiance = { 0.f, 0.f, 0.f };
							radiance = kernel_infinite->Le(lightRay.d);
							float choicePdf = PdfFromLightDistribution(kernel_light_size);
							float lightPdf, pdfA;
							float3 lightNor;
							kernel_infinite->Pdf(lightRay, lightNor, pdfA, lightPdf);
							float weight = PowerHeuristic(1, pdf, 1, lightPdf*choicePdf);
							float3 tr = { 1.f, 1.f, 1.f };
							if (lightRay.medium){
								if (lightRay.medium->type == MT_HOMOGENEOUS)
									tr = lightRay.medium->homogeneous.Tr(lightRay, uniform, rng);
								else
									tr = lightRay.medium->heterogeneous.Tr(lightRay, uniform, rng);
							}
							Ld += weight * tr * fr * radiance * fabs(dot(out, nor)) / pdf;
						}
					}
				}

				Li += beta*Ld;
			}

			float3 u = make_float3(uniform(rng), uniform(rng), uniform(rng));
			float3 out, fr;
			float pdf;

			SampleBSDF(material, -r.d, nor, uv, dpdu, u, out, fr, pdf);
			if (IsBlack(fr))
				break;

			beta *= fr*fabs(dot(nor, out)) / pdf;
			specular = IsDelta(material.type);

			Medium* m = dot(out, nor) > 0 ? (isect.mediumOutside == -1 ? nullptr : &kernel_mediums[isect.mediumOutside])
				: (isect.mediumInside == -1 ? nullptr : &kernel_mediums[isect.mediumInside]);
			m = dot(-r.d, nor)*dot(out, nor) > 0 ? r.medium : m;

			r = Ray(pos, out, m, kernel_epsilon);
		}

		if (bounces > 3){
			float illumate = clamp(1.f - Luminance(beta), 0.f, 1.f);
			if (uniform(rng) < illumate)
				break;

			beta /= (1 - illumate);
		}
	}

	if (!IsInf(Li) && !IsNan(Li))
			kernel_color[pixel] = Li;
}
//**************************VolPath End****************************

//**************************Lighttracing Integrator****************
__global__ void LightTracingInit(){
	unsigned x = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned y = blockIdx.y*blockDim.y + threadIdx.y;
	unsigned pixel = x + y*blockDim.x*gridDim.x;

	kernel_color[pixel] = { 0.f, 0.f, 0.f };
}

__global__ void LightTracing(int iter, int maxDepth){
	unsigned x = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned y = blockIdx.y*blockDim.y + threadIdx.y;
	unsigned pixel = x + y*blockDim.x*gridDim.x;

	//init seed
	int threadIndex = (blockIdx.x + blockIdx.y * gridDim.x) * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;
	thrust::default_random_engine rng(WangHash(pixel) + WangHash(iter));
	thrust::uniform_real_distribution<float> uniform(0.0f, 1.0f);

	float3 beta = { 1.f, 1.f, 1.f };
	float choicePdf;
	int lightIdx = LookUpLightDistribution(uniform(rng), choicePdf);
	Area light = kernel_lights[lightIdx];
	float4 u = make_float4(uniform(rng), uniform(rng), uniform(rng), uniform(rng));
	Ray ray;
	float3 nor, radiance;
	float pdfA, pdfW;
	light.SampleLight(u, ray, nor, radiance, pdfA, pdfW, kernel_epsilon);
	ray.medium = light.medium == -1 ? nullptr : &kernel_mediums[light.medium];

	beta *= radiance*fabs(dot(ray.d, nor)) / (pdfA*pdfW*choicePdf);

	Ray shadowRay;
	float we, cameraPdf;
	int raster;
	kernel_camera->SampleCamera(ray.o, shadowRay, we, cameraPdf, raster, kernel_epsilon);
	shadowRay.medium = ray.medium;
	if (cameraPdf != 0.f){
		float3 tr = Tr(shadowRay, uniform, rng);
		if (!IsBlack(tr))
			kernel_color[raster] += tr*radiance;
	}

	Intersection isect;
	for (int bounces = 0; bounces < maxDepth; ++bounces){
		if (!Intersect(ray, &isect)){
			break;
		}

		float3 pos = isect.pos;
		float3 nor = isect.nor;
		float2 uv = isect.uv;
		float sampledDist;
		bool sampledMedium = false;
		if (ray.medium){
			if (ray.medium->type == MT_HOMOGENEOUS)
				beta *= ray.medium->homogeneous.Sample(ray, uniform, rng, sampledDist, sampledMedium);
			else
				beta *= ray.medium->heterogeneous.Sample(ray, uniform, rng, sampledDist, sampledMedium);
		}
		if (IsBlack(beta)) break;
		if (sampledMedium){
			float3 samplePos = ray(sampledDist);
			Ray shadowRay;
			float we, cameraPdf;
			int raster;
			kernel_camera->SampleCamera(samplePos, shadowRay, we, cameraPdf, raster, kernel_epsilon);
			shadowRay.medium = ray.medium;
			float3 tr = Tr(shadowRay, uniform, rng);
			float phase, unuse;
			ray.medium->Phase(-ray.d, shadowRay.d, phase, unuse);

			float3 L = beta*we*tr*phase / cameraPdf;
			if (!IsInf(L) && !IsNan(L)){
				//kernel_color[raster] += L;
				atomicAdd(&kernel_color[raster].x, L.x);
				atomicAdd(&kernel_color[raster].y, L.y);
				atomicAdd(&kernel_color[raster].z, L.z);
			}

			float pdf;
			float2 phaseU = make_float2(uniform(rng), uniform(rng));
			float3 dir;
			ray.medium->SamplePhase(phaseU, dir, phase, pdf);
			ray = Ray(samplePos, dir, ray.medium, kernel_epsilon);
		}
		else{
			if (isect.matIdx == -1){
				bounces--;
				Medium* m = dot(ray.d, isect.nor) > 0 ? (isect.mediumOutside == -1 ? nullptr : &kernel_mediums[isect.mediumOutside])
					: (isect.mediumInside == -1 ? nullptr : &kernel_mediums[isect.mediumInside]);
				ray = Ray(pos, ray.d, m, kernel_epsilon);

				continue;
			}

			Material mat = kernel_materials[isect.matIdx];

			//direct
			if (!IsDelta(mat.type)){
				Ray shadowRay;
				float we, cameraPdf;
				int raster;
				kernel_camera->SampleCamera(pos, shadowRay, we, cameraPdf, raster, kernel_epsilon);
				shadowRay.medium = ray.medium;

				if (cameraPdf != 0.f){
					float3 tr = Tr(shadowRay, uniform, rng);
					float3 fr;
					float unuse;
					Fr(mat, -ray.d, shadowRay.d, nor, uv, isect.dpdu, fr, unuse);

					float3 L = tr*beta*fr*we*fabs(dot(shadowRay.d, nor)) / cameraPdf;
					if (!IsInf(L) && !IsNan(L)){
						//kernel_color[raster] += L;
						atomicAdd(&kernel_color[raster].x, L.x);
						atomicAdd(&kernel_color[raster].y, L.y);
						atomicAdd(&kernel_color[raster].z, L.z);
					}
				}
			}

			float3 u = make_float3(uniform(rng), uniform(rng), uniform(rng));
			float3 out, fr;
			float pdf;
			SampleBSDF(mat, -ray.d, nor, uv, isect.dpdu, u, out, fr, pdf, TransportMode::Importance);
			if (IsBlack(fr))
				break;
			beta *= fr*fabs(dot(out, nor)) / pdf;
			Medium* m = dot(out, nor) > 0 ? (isect.mediumOutside == -1 ? nullptr : &kernel_mediums[isect.mediumOutside])
				: (isect.mediumInside == -1 ? nullptr : &kernel_mediums[isect.mediumInside]);
			m = dot(-ray.d, nor)*dot(out, nor) > 0 ? ray.medium : m;

			ray = Ray(pos, out, m, kernel_epsilon);
		}

		if (bounces > 3){
			float illumate = clamp(1.f - Luminance(beta), 0.f, 1.f);
			if (uniform(rng) < illumate)
				break;

			beta /= (1 - illumate);
		}
	}
}
//**************************Lighttracing End***********************

//**************************Bdpt Integrator************************
#define BDPT_MAX_DEPTH 65

struct BdptVertex{
	float3 beta;
	Intersection isect;
	Medium* medium = nullptr;
	bool delta;
	float fwd;
	float rev;
};

//convert pdf from area to omega
__device__ float ConvertPdf(float pdf, Intersection& prev, Intersection& cur){
	float3 dir = prev.pos - cur.pos;
	float square = dot(dir, dir);
	dir = normalize(dir);
	float ret = pdf / square;
	if (!IsBlack(cur.nor))
	    ret *= fabs(dot(dir, cur.nor));
	return ret;
}

__device__ int GenerateCameraPath(int x, int y, BdptVertex* path, thrust::uniform_real_distribution<float>& uniform, thrust::default_random_engine& rng){
	//start
	float offsetx = uniform(rng) - 0.5f;
	float offsety = uniform(rng) - 0.5f;
	float unuse;
	//bdpt doesn't support dof now
	//float2 aperture = UniformDisk(uniform(rng), uniform(rng), unuse);//for dof
	Ray ray = kernel_camera->GeneratePrimaryRay(x + offsetx, y + offsety, make_float2(0, 0));
	ray.tmin = kernel_epsilon;
	ray.medium = kernel_camera->medium == -1 ? nullptr : &kernel_mediums[kernel_camera->medium];
	float3 beta = { 1.f, 1.f, 1.f };

	int nVertex = 0;
	//set camera isect
	{
		Intersection cameraIsect;
		cameraIsect.pos = kernel_camera->position;
		cameraIsect.nor = -kernel_camera->w;
		BdptVertex vertex;
		vertex.beta = beta;
		vertex.isect = cameraIsect;
		vertex.delta = false;
		vertex.medium = ray.medium;
		vertex.fwd = 1.f;
		path[0] = vertex;
	}
	nVertex++;

	float forward = 0.f, rrPdf = 1.f;
	kernel_camera->PdfCamera(ray.d, unuse, forward);
	Intersection isect;
	int bounces = 0;
	for (; bounces < BDPT_MAX_DEPTH; ++bounces){
		if (!Intersect(ray, &isect)){
			break;
		}

		float3 pos = isect.pos;
		float3 nor = isect.nor;
		float2 uv = isect.uv;
		float sampledDist;
		bool sampledMedium = false;
		if (ray.medium){
			if (ray.medium->type == MT_HOMOGENEOUS)
				beta *= ray.medium->homogeneous.Sample(ray, uniform, rng, sampledDist, sampledMedium);
			else
				beta *= ray.medium->heterogeneous.Sample(ray, uniform, rng, sampledDist, sampledMedium);
		}
		if (IsBlack(beta)) break;
		if (sampledMedium){
			float3 samplePos = ray(sampledDist);

			float phase, pdf;
			float2 phaseU = make_float2(uniform(rng), uniform(rng));
			float3 dir;
			ray.medium->SamplePhase(phaseU, dir, phase, pdf);
			ray = Ray(samplePos, dir, ray.medium, kernel_epsilon);

			//set medium Intersection
			{
				BdptVertex vertex;
				Intersection mediumIsect;
				mediumIsect.pos = samplePos;
				mediumIsect.nor = { 0.f, 0.f, 0.f };
				mediumIsect.matIdx = -1;
				mediumIsect.lightIdx = -1;
				vertex.beta = beta;
				vertex.delta = false;
				vertex.isect = mediumIsect;
				vertex.medium = ray.medium;
				path[bounces + 1] = vertex;
				path[bounces + 1].fwd = ConvertPdf(forward, path[bounces].isect, path[bounces + 1].isect);
				forward = phase;
				path[bounces].rev = ConvertPdf(forward, path[bounces + 1].isect, path[bounces].isect);
			}
			nVertex++;
		}
		else{
			if (isect.matIdx == -1){
				bounces--;
				Medium* m = dot(ray.d, isect.nor) > 0 ? (isect.mediumOutside == -1 ? nullptr : &kernel_mediums[isect.mediumOutside])
					: (isect.mediumInside == -1 ? nullptr : &kernel_mediums[isect.mediumInside]);
				ray = Ray(pos, ray.d, m, kernel_epsilon);

				continue;
			}

			Material mat = kernel_materials[isect.matIdx];

			{
				BdptVertex vertex;
				vertex.beta = beta;
				vertex.isect = isect;
				vertex.delta = IsDelta(mat.type);
				vertex.medium = ray.medium;
				path[bounces + 1] = vertex;
				path[bounces + 1].fwd = ConvertPdf(forward, path[bounces].isect, path[bounces + 1].isect);
			}
			nVertex++;

			float3 u = make_float3(uniform(rng), uniform(rng), uniform(rng));
			float3 out, fr;
			float pdf;
			SampleBSDF(mat, -ray.d, nor, uv, isect.dpdu, u, out, fr, pdf);
			if (IsBlack(fr))
				break;
			beta *= fr*fabs(dot(out, nor)) / pdf;

			forward = pdf;
			if (IsDelta(mat.type)) forward = 0.f;
			//calc reverse pdf
			{
				float3 unuseFr;
				float pdf;
				Fr(mat, out, -ray.d, nor, uv, isect.dpdu, unuseFr, pdf);
				path[bounces].rev = ConvertPdf(pdf, path[bounces + 1].isect, path[bounces].isect);
			}

			Medium* m = dot(out, nor) > 0 ? (isect.mediumOutside == -1 ? nullptr : &kernel_mediums[isect.mediumOutside])
				: (isect.mediumInside == -1 ? nullptr : &kernel_mediums[isect.mediumInside]);
			m = dot(-ray.d, nor)*dot(out, nor) > 0 ? ray.medium : m;

			ray = Ray(pos, out, m, kernel_epsilon);
		}

		//russian roulette
		if (bounces > 3){
			rrPdf = clamp(1.f - Luminance(beta), 0.f, 1.f);
			if (uniform(rng) < rrPdf)
				break;

			beta /= (1 - rrPdf);
		}
	}

	return nVertex;
}

__device__ int GenerateLightPath(BdptVertex* path, thrust::uniform_real_distribution<float>& uniform, thrust::default_random_engine& rng){
	float3 beta = { 1.f, 1.f, 1.f };
	float choicePdf;
	int lightIdx = LookUpLightDistribution(uniform(rng), choicePdf);
	Area light = kernel_lights[lightIdx];
	float4 u = make_float4(uniform(rng), uniform(rng), uniform(rng), uniform(rng));
	Ray ray;
	float3 lightNor, radiance;
	float pdfA, pdfW;
	light.SampleLight(u, ray, lightNor, radiance, pdfA, pdfW, kernel_epsilon);
	ray.medium = light.medium == -1 ? nullptr : &kernel_mediums[light.medium];

	int nVertex = 0;
	//set light isect
	{
		Intersection lightIsect;
		lightIsect.pos = ray.o;
		lightIsect.nor = lightNor;
		lightIsect.lightIdx = lightIdx;
		BdptVertex vertex;
		vertex.beta = radiance;
		vertex.isect = lightIsect;
		vertex.delta = false;
		vertex.medium = ray.medium;
		vertex.fwd = pdfA * choicePdf;
		path[0] = vertex;
	}
	nVertex++;
	beta *= radiance*fabs(dot(ray.d, lightNor)) / (pdfA*pdfW*choicePdf);

	Intersection isect;
	float forward = pdfW, rrPdf = 1.f;
	int bounces = 0;
	for (; bounces < BDPT_MAX_DEPTH; ++bounces){
		if (!Intersect(ray, &isect)){
			break;
		}

		float3 pos = isect.pos;
		float3 nor = isect.nor;
		float2 uv = isect.uv;
		float sampledDist;
		bool sampledMedium = false;
		if (ray.medium){
			if (ray.medium->type == MT_HOMOGENEOUS)
				beta *= ray.medium->homogeneous.Sample(ray, uniform, rng, sampledDist, sampledMedium);
			else
				beta *= ray.medium->heterogeneous.Sample(ray, uniform, rng, sampledDist, sampledMedium);
		}
		if (IsBlack(beta)) break;
		if (sampledMedium){
			float3 samplePos = ray(sampledDist);

			float phase, pdf;
			float2 phaseU = make_float2(uniform(rng), uniform(rng));
			float3 dir;
			ray.medium->SamplePhase(phaseU, dir, phase, pdf);
			ray = Ray(samplePos, dir, ray.medium, kernel_epsilon);

			//set medium Intersection
			{
				BdptVertex vertex;
				Intersection mediumIsect;
				mediumIsect.pos = samplePos;
				mediumIsect.nor = { 0.f, 0.f, 0.f };
				mediumIsect.matIdx = -1;
				mediumIsect.lightIdx = -1;
				vertex.beta = beta;
				vertex.delta = false;
				vertex.isect = mediumIsect;
				vertex.medium = ray.medium;
				path[bounces + 1] = vertex;
				path[bounces + 1].fwd = ConvertPdf(forward, path[bounces].isect, path[bounces + 1].isect);
				forward = phase;
				path[bounces].rev = ConvertPdf(phase, path[bounces + 1].isect, path[bounces].isect);
			}
			nVertex++;
		}
		else{
			if (isect.matIdx == -1){
				bounces--;
				Medium* m = dot(ray.d, isect.nor) > 0 ? (isect.mediumOutside == -1 ? nullptr : &kernel_mediums[isect.mediumOutside])
					: (isect.mediumInside == -1 ? nullptr : &kernel_mediums[isect.mediumInside]);
				ray = Ray(pos, ray.d, m, kernel_epsilon);

				continue;
			}
			Material mat = kernel_materials[isect.matIdx];

			{
				BdptVertex vertex;
				vertex.beta = beta;
				vertex.isect = isect;
				vertex.delta = IsDelta(mat.type);
				vertex.medium = ray.medium;
				path[bounces + 1] = vertex;
				path[bounces + 1].fwd = ConvertPdf(forward, path[bounces].isect, path[bounces + 1].isect);
			}
			nVertex++;

			float3 u = make_float3(uniform(rng), uniform(rng), uniform(rng));
			float3 out, fr;
			float pdf;
			SampleBSDF(mat, -ray.d, nor, uv, isect.dpdu, u, out, fr, pdf, TransportMode::Importance);
			if (IsBlack(fr))
				break;
			beta *= fr*fabs(dot(out, nor)) / pdf;

			forward = pdf;
			if (IsDelta(mat.type)) forward = 0.f;
			//calc reverse pdf
			{
				float3 unuseFr;
				float pdf;
				Fr(mat, out, -ray.d, nor, uv, isect.dpdu, unuseFr, pdf);
				path[bounces].rev = ConvertPdf(pdf, path[bounces + 1].isect, path[bounces].isect);
			}
			Medium* m = dot(out, nor) > 0 ? (isect.mediumOutside == -1 ? nullptr : &kernel_mediums[isect.mediumOutside])
				: (isect.mediumInside == -1 ? nullptr : &kernel_mediums[isect.mediumInside]);
			m = dot(-ray.d, nor)*dot(out, nor) > 0 ? ray.medium : m;

			ray = Ray(pos, out, m, kernel_epsilon);
		}

		//russian roulette
		if (bounces > 3){
			rrPdf = clamp(1.f - Luminance(beta), 0.f, 1.f);
			if (uniform(rng) < rrPdf)
				break;

			beta /= (1 - rrPdf);
		}
	}

	return nVertex;
}

__device__ float MisWeight(BdptVertex* cameraPath, BdptVertex* lightPath, int s, int t){
	if (s + t == 2)//light source is directly visible
		return 1.f;

	//delta bsdf pdf is 0
	auto remap = [](float pdf)->float{
		return pdf == 0 ? 1.f : pdf;
	};

	float sumW = 0.f;
	float ri = 1.f;
	for (int i = s - 1; i > 0; --i){
		ri *= remap(cameraPath[i].rev) / remap(cameraPath[i].fwd);

		if (!cameraPath[i].delta && !cameraPath[i - 1].delta)
			sumW += ri;
	}

	ri = 1.f;
	for (int i = t - 1; i >= 0; --i){
		ri *= remap(lightPath[i].rev) / remap(lightPath[i].fwd);

		bool delta = lightPath[i == 0 ? 0 : i - 1].delta;
		if (!lightPath[i].delta && !delta)
			sumW += ri;
	}

	return 1.f / (sumW + 1.f);
}

__device__ float3 Connect(BdptVertex* cameraPath, BdptVertex* lightPath, int s, int t, int& raster,
	thrust::uniform_real_distribution<float>& uniform, thrust::default_random_engine& rng){
	float3 L = { 0.f, 0.f, 0.f };

	if (t == 0){
		//naive path tracing
		BdptVertex& cur = cameraPath[s - 1];
		BdptVertex& prev = cameraPath[s - 2];
		if (cur.isect.lightIdx == -1) return{ 0.f, 0.f, 0.f };

		float3 dir = normalize(prev.isect.pos - cur.isect.pos);
		Area light = kernel_lights[cur.isect.lightIdx];
		L += cur.beta*light.Le(cur.isect.nor, dir);
		if (IsBlack(L)) return L;

		Ray ray(cur.isect.pos, dir);
		float choicePdf = PdfFromLightDistribution(cur.isect.lightIdx);
		float pdfA, pdfW;
		light.Pdf(ray, cur.isect.nor, pdfA, pdfW);
		float curRev = cur.rev;
		float prevRev = prev.rev;
		cur.rev = pdfA*choicePdf;
		prev.rev = ConvertPdf(pdfW, cur.isect, prev.isect);
		float mis = MisWeight(cameraPath, lightPath, s, t);
		//reset
		cur.rev = curRev;
		prev.rev = prevRev;

		return mis*L;
	}
	else if (t == 1){
		//next event path tracing
		BdptVertex& prev = cameraPath[s - 2];
		BdptVertex& cur = cameraPath[s - 1];
		BdptVertex& next = lightPath[0];
		float3 in = normalize(prev.isect.pos - cur.isect.pos);
		bool isMedium = cur.isect.matIdx == -1;
		Material mat;
		if (!isMedium) mat = kernel_materials[cur.isect.matIdx];
		float choicePdf, lightPdf;
		int idx = LookUpLightDistribution(uniform(rng), choicePdf);
		Area light = kernel_lights[idx];
		float3 radiance, lightNor, lightPos;
		Ray shadowRay;
		float2 lightUniform = { uniform(rng), uniform(rng) };
		light.SampleLight(cur.isect.pos, lightUniform, radiance, shadowRay, lightNor, lightPdf, kernel_epsilon);
		lightPos = shadowRay(shadowRay.tmax + kernel_epsilon);
		shadowRay.medium = cur.medium;
		if (IsBlack(radiance)) return{ 0.f, 0.f, 0.f };
		if (!isMedium && IsDelta(mat.type)) return{ 0.f, 0.f, 0.f };
		float3 tr = Tr(shadowRay, uniform, rng);
		if (IsBlack(tr)) return{ 0.f, 0.f, 0.f };

		float3 fr;
		float nextPdf, G, phase;
		if (isMedium){
			cur.medium->Phase(in, shadowRay.d, phase, nextPdf);
			fr = make_float3(phase, phase, phase);
			G = 1.f;
		}
		else{
			Fr(mat, in, shadowRay.d, cur.isect.nor, cur.isect.uv, cur.isect.dpdu, fr, nextPdf);
			G = fabs(dot(cur.isect.nor, shadowRay.d));
		}
		L += cur.beta*tr*fr*radiance*G / (lightPdf*choicePdf);
		if (IsBlack(L)) return{ 0.f, 0.f, 0.f };

		BdptVertex tNext = next;
		float pdfA, pdfW;
		light.Pdf(shadowRay, lightNor, pdfA, pdfW);
		next.isect.pos = lightPos;
		next.isect.nor = lightNor;
		next.fwd = pdfA*choicePdf;
		next.rev = ConvertPdf(nextPdf, cur.isect, next.isect);
		float curRev = cur.rev;
		float prevRev = prev.rev;
		cur.rev = ConvertPdf(pdfW, next.isect, cur.isect);
		float pdf;
		if (isMedium) pdf = phase;
		else Fr(mat, shadowRay.d, in, cur.isect.nor, cur.isect.uv, cur.isect.dpdu, fr, pdf);
		prev.rev = ConvertPdf(pdf, cur.isect, prev.isect);
		float mis = MisWeight(cameraPath, lightPath, s, t);

		cur.rev = curRev;
		prev.rev = prevRev;
		next = tNext;

		return mis*L;
	}
	else if (s == 1){
		//light tracing
		BdptVertex& prev = lightPath[t - 2];
		BdptVertex& cur = lightPath[t - 1];
		BdptVertex& next = cameraPath[0];
		float3 in = normalize(prev.isect.pos - cur.isect.pos);
		bool isMedium = cur.isect.matIdx == -1;
		Material mat;
		if (!isMedium) mat = kernel_materials[cur.isect.matIdx];
		Ray shadowRay;
		float we, cameraPdf;
		kernel_camera->SampleCamera(cur.isect.pos, shadowRay, we, cameraPdf, raster, kernel_epsilon);
		shadowRay.medium = cur.medium;
		if (cameraPdf == 0) return{ 0.f, 0.f, 0.f };
		if (!isMedium && IsDelta(mat.type)) return{ 0.f, 0.f, 0.f };
		float3 tr = Tr(shadowRay, uniform, rng);
		if (IsBlack(tr)) return{ 0.f, 0.f, 0.f };

		float3 fr;
		float nextPdf, phase, costheta = fabs(dot(shadowRay.d, cur.isect.nor));
		if (isMedium){
			cur.medium->Phase(in, shadowRay.d, phase, nextPdf);
			fr = make_float3(phase, phase, phase);
			costheta = 1.f;
		}
		else Fr(mat, in, shadowRay.d, cur.isect.nor, cur.isect.uv, cur.isect.dpdu, fr, nextPdf);
		L += cur.beta*tr*fr*we*costheta / cameraPdf;
		if (IsBlack(L)) return{ 0.f, 0.f, 0.f };

		float nextRev = next.rev;
		float curRev = cur.rev;
		float prevRev = prev.rev;
		next.rev = ConvertPdf(nextPdf, cur.isect, next.isect);
		float pdfA, pdfW;
		kernel_camera->PdfCamera(-shadowRay.d, pdfA, pdfW);
		cur.rev = ConvertPdf(pdfW, next.isect, cur.isect);
		float pdf;
		if (isMedium) pdf = phase;
		else Fr(mat, shadowRay.d, in, cur.isect.nor, cur.isect.uv, cur.isect.dpdu, fr, pdf);
		prev.rev = ConvertPdf(pdf, cur.isect, prev.isect);
		float mis = MisWeight(cameraPath, lightPath, s, t);
		next.rev = nextRev;
		cur.rev = curRev;
		prev.rev = prevRev;

		return mis*L;
	}
	else{
		//other
		BdptVertex& c2 = cameraPath[s - 2];
		BdptVertex& c1 = cameraPath[s - 1];
		BdptVertex& l1 = lightPath[t - 1];
		BdptVertex& l2 = lightPath[t - 2];
		float3 l1Tol2 = normalize(l2.isect.pos - l1.isect.pos);
		float3 l1Toc1 = normalize(c1.isect.pos - l1.isect.pos);
		float3 c1Tol1 = -l1Toc1;
		float3 c1Toc2 = normalize(c2.isect.pos - c1.isect.pos);
		float3 dir = c1.isect.pos - l1.isect.pos;
		Material c1Mat, l1Mat;
		if (!c1.medium) c1Mat = kernel_materials[c1.isect.matIdx];
		if (!l1.medium) l1Mat = kernel_materials[l1.isect.matIdx];
		Ray shadowRay;
		shadowRay.o = c1.isect.pos;
		shadowRay.d = c1Tol1;
		shadowRay.medium = c1.medium;
		shadowRay.tmin = kernel_epsilon;
		shadowRay.tmax = length(dir) - kernel_epsilon;
		if (!c1.medium && IsDelta(c1Mat.type)) return{ 0.f, 0.f, 0.f };
		if (!l1.medium && IsDelta(l1Mat.type)) return{ 0.f, 0.f, 0.f };
		float3 tr = Tr(shadowRay, uniform, rng);
		if (IsBlack(tr)) return{ 0.f, 0.f, 0.f };
		float cos1 = l1.medium ? 1.f : fabs(dot(l1Toc1, l1.isect.nor));
		float cos2 = c1.medium ? 1.f : fabs(dot(c1Tol1, c1.isect.nor));

		float3 c1Fr, l1Fr;
		float l1Pdf, c1Pdf;
		float l1Phase, c1Phase;
		if (c1.medium){
			c1.medium->Phase(c1Toc2, c1Tol1, c1Phase, l1Pdf);
			c1Fr = make_float3(c1Phase, c1Phase, c1Phase);
		}
		else Fr(c1Mat, c1Toc2, c1Tol1, c1.isect.nor, c1.isect.uv, c1.isect.dpdu, c1Fr, l1Pdf);
		if (l1.medium){
			l1.medium->Phase(l1Tol2, l1Toc1, l1Phase, c1Pdf);
			l1Fr = make_float3(l1Phase, l1Phase, l1Phase);
		}
		else Fr(l1Mat, l1Tol2, l1Toc1, l1.isect.nor, l1.isect.uv, l1.isect.dpdu, l1Fr, c1Pdf);
		float3 G = tr*cos1*cos2 / dot(dir, dir);
		L += c1.beta*c1Fr*G*l1Fr*l1.beta;
		if (IsBlack(L)) return{ 0.f, 0.f, 0.f };

		float c2Rev = c2.rev;
		float c1Rev = c1.rev;
		float l1Rev = l1.rev;
		float l2Rev = l2.rev;
		c1.rev = ConvertPdf(c1Pdf, l1.isect, c1.isect);
		l1.rev = ConvertPdf(l1Pdf, c1.isect, l1.isect);
		float l2Pdf, c2Pdf;
		if (l1.medium) l1.medium->Phase(l1Toc1, l1Tol2, l1Phase, l2Pdf);
		else Fr(l1Mat, l1Toc1, l1Tol2, l1.isect.nor, l1.isect.uv, l1.isect.dpdu, l1Fr, l2Pdf);
		if (c1.medium) c1.medium->Phase(c1Tol1, c1Toc2, c1Phase, c2Pdf);
		else Fr(c1Mat, c1Tol1, c1Toc2, c1.isect.nor, c1.isect.uv, c1.isect.dpdu, c1Fr, c2Pdf);
		l2.rev = ConvertPdf(l2Pdf, l1.isect, l2.isect);
		c2.rev = ConvertPdf(c2Pdf, c1.isect, c2.isect);
		float mis = MisWeight(cameraPath, lightPath, s, t);
		c2.rev = c2Rev;
		c1.rev = c1Rev;
		l1.rev = l1Rev;
		l2.rev = l2Rev;

		return mis*L;
	}

	return L;
}

__global__ void BdptInit(){
	unsigned x = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned y = blockIdx.y*blockDim.y + threadIdx.y;
	unsigned pixel = x + y*blockDim.x*gridDim.x;

	kernel_color[pixel] = { 0.f, 0.f, 0.f };
}

__global__ void Bdpt(int iter, int maxDepth){
	unsigned x = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned y = blockIdx.y*blockDim.y + threadIdx.y;
	unsigned pixel = x + y*blockDim.x*gridDim.x;

	//init seed
	int threadIndex = (blockIdx.x + blockIdx.y * gridDim.x) * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;
	thrust::default_random_engine rng(WangHash(pixel) + WangHash(iter));
	thrust::uniform_real_distribution<float> uniform(0.0f, 1.0f);

	//too slow to use dynamic allocate
	BdptVertex cameraPath[BDPT_MAX_DEPTH + 2];
	BdptVertex lightPath[BDPT_MAX_DEPTH + 2];
	int nCamera = GenerateCameraPath(x, y, cameraPath, uniform, rng);
	int nLight = GenerateLightPath(lightPath, uniform, rng);
	for (int s = 1; s <= nCamera; ++s){
		for (int t = 0; t <= nLight; ++t){
			if ((s == 1 && t == 0) || (s == 1 && t == 1))
				continue;

			int raster;
			float3 L = Connect(cameraPath, lightPath, s, t, raster, uniform, rng);
			if (IsInf(L) || IsNan(L) || IsBlack(L))
				continue;

			if (s == 1){
				atomicAdd(&kernel_color[raster].x, L.x);
				atomicAdd(&kernel_color[raster].y, L.y);
				atomicAdd(&kernel_color[raster].z, L.z);
				continue;
			}

			atomicAdd(&kernel_color[pixel].x, L.x);
			atomicAdd(&kernel_color[pixel].y, L.y);
			atomicAdd(&kernel_color[pixel].z, L.z);
		}
	}
}
//**************************Bdpt End*******************************

//**************************Mlt Integrator*************************

//gaussian distribution for small step
class MLTSampler{

};

__global__ void Mlt(int iter, int maxDepth){

}
//**************************Mlt End********************************

//**************************PPM Integrator*************************
struct VisiblePoint{
	float3 ld; //direct light
	float3 ind; //indirect light
	float3 beta; //throughput
	float3 dir; 
	Intersection isect;

	float3 tau;
	float radius;
	float n;
	bool valid = false;
};

struct CPUGridNode{
	vector<int> vpIdx;
};

VisiblePoint* device_vps;
int* device_vpIdx, *device_vpOffset;
int totalNodes = 0;
__device__ VisiblePoint* vps;
__device__ int* vpIdx, *vpOffset;//grid info
__device__ float3 boundsMin, boundsMax;
__device__ int gridRes[3], hashSize;
__global__ void SPPMSetParam(int* idx, float3 fmin, float3 fmax, int x, int y, int z, int hsize){
	vpIdx = idx;
	boundsMin = fmin;
	boundsMax = fmax;
	gridRes[0] = x;
	gridRes[1] = y;
	gridRes[2] = z;
	hashSize = hsize;
}

//from pbrt-v3
__host__ __device__ bool ToGrid(float3& p, BBox& bounds, int gridRes[3], float3& pi){
	bool inBounds = true;
	float3 pg = bounds.Offset(p);
	for (int i = 0; i < 3; ++i){
		(&pi.x)[i] = (int)(gridRes[i] * (&pg.x)[i]);
		inBounds &= ((&pi.x)[i] >= 0 && (&pi.x)[i] < gridRes[i]);
		(&pi.x)[i] = clamp((int)(&pi.x)[i], 0, gridRes[i] - 1);
	}

	return inBounds;
}

__host__ __device__ unsigned int Hash(int x, int y, int z, int hashSize){
	//those magic number are some large primes
	return (unsigned int)((x * 73856093) ^ (y * 19349663) ^ (z * 83492791)) % hashSize;
}

//Still too slow, i will be very grateful if someone tells me how to optimize!!
void BuildHashTable(int width, int height){
	VisiblePoint* host_vps = new VisiblePoint[width*height];
	HANDLE_ERROR(cudaMemcpy(host_vps, device_vps, width*height*sizeof(VisiblePoint), cudaMemcpyDeviceToHost));

	int hSize = width*height;
	CPUGridNode* grid = new CPUGridNode[hSize];

	BBox gridBounds;
	float initRadius = 0.f;
	for (int i = 0; i < width*height; ++i){
		gridBounds.Expand(host_vps[i].isect.pos);
		if (host_vps[i].radius > initRadius) initRadius = host_vps[i].radius;
	}

	float3 radius3f = make_float3(initRadius, initRadius, initRadius);
	gridBounds.fmin -= radius3f;
	gridBounds.fmax += radius3f;
	float3 diag = gridBounds.Diagonal();
	float maxDiag = (&diag.x)[gridBounds.GetMaxExtent()];
	int baseGridRes = (int)(maxDiag / initRadius);
	int gRes[3];
	for (int i = 0; i < 3; ++i)
		gRes[i] = Max((int)(baseGridRes*(&diag.x)[i] / maxDiag), 1);

    int total = 0;
	for (int i = 0; i < width*height; ++i){
		VisiblePoint vp = host_vps[i];
		float3 pMin, pMax;
		ToGrid(vp.isect.pos - radius3f, gridBounds, gRes, pMin);
		ToGrid(vp.isect.pos + radius3f, gridBounds, gRes, pMax);
		for (int z = pMin.z; z <= pMax.z; ++z){
			for (int y = pMin.y; y <= pMax.y; ++y){
				for (int x = pMin.x; x <= pMax.x; ++x){
					int h = Hash(x, y, z, hSize);
					grid[h].vpIdx.push_back(i);
					total++;
				}
			}
		}
	}

	vector<int> temp(total), off(hSize + 1); off[0] = 0;
	int* start = &temp[0], offset = 0;
	for (int i = 0; i < hSize; ++i){
		memcpy(start + offset, &grid[i].vpIdx[0], grid[i].vpIdx.size()*sizeof(int));
		offset += grid[i].vpIdx.size();
		off[i + 1] = offset;
	}

	if (total != totalNodes){
		HANDLE_ERROR(cudaFree(device_vpIdx));
		HANDLE_ERROR(cudaMalloc(&device_vpIdx, total*sizeof(int)));
	}
	HANDLE_ERROR(cudaMemcpy(device_vpIdx, &temp[0], total*sizeof(int), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(device_vpOffset, &off[0], (hSize + 1)*sizeof(int), cudaMemcpyHostToDevice));
	
	SPPMSetParam << <1, 1 >> >(device_vpIdx, gridBounds.fmin, gridBounds.fmax, gRes[0], gRes[1], gRes[2], hSize);
	delete[] host_vps;
	
	delete[] grid;
}

__device__ void TraceRay(int pixel, Ray r, int iter, int maxDepth, float initRadius, thrust::uniform_real_distribution<float>& uniform, thrust::default_random_engine& rng){
	VisiblePoint& vp = vps[pixel];
	if (iter == 1){
		vp.radius = initRadius;
		vp.n = 0.f;
		vp.ld = { 0.f, 0.f, 0.f };
		vp.tau = { 0.f, 0.f, 0.f };
		vp.valid = false;
	}

	float3 beta = { 1.f, 1.f, 1.f };
	Ray ray = r;
	bool specular = false;
	for (int bounces = 0; bounces < maxDepth; ++bounces){
		Intersection isect;
		if (!Intersect(ray, &isect)){
			break;
		}

		float3 pos = isect.pos;
		float3 nor = isect.nor;
		float2 uv = isect.uv;
		float3 dpdu = isect.dpdu;
		Material mat = kernel_materials[isect.matIdx];

		float3 Ld = { 0.f, 0.f, 0.f };
		if (!IsDelta(mat.type) && isect.lightIdx == -1){
			float u = uniform(rng);
			float choicePdf;
			int idx = LookUpLightDistribution(u, choicePdf);
			float2 u1 = make_float2(uniform(rng), uniform(rng));
			float3 radiance, lightNor;
			Ray shadowRay;
			float lightPdf;
			kernel_lights[idx].SampleLight(pos, u1, radiance, shadowRay, lightNor, lightPdf, kernel_epsilon);

			if (!IsBlack(radiance) && !IntersectP(shadowRay)){
				float3 fr;
				float samplePdf;

				Fr(mat, -ray.d, shadowRay.d, nor, uv, dpdu, fr, samplePdf);

				float weight = PowerHeuristic(1, lightPdf * choicePdf, 1, samplePdf);
				Ld += weight*fr*radiance*fabs(dot(nor, shadowRay.d)) / (lightPdf*choicePdf);
			}

			float3 us = make_float3(uniform(rng), uniform(rng), uniform(rng));
			float3 out, fr;
			float pdf;
			SampleBSDF(mat, -ray.d, nor, uv, dpdu, us, out, fr, pdf);
			if (!(IsBlack(fr) || pdf == 0)){
				Intersection lightIsect;
				Ray lightRay(pos, out, nullptr, kernel_epsilon);
				if (Intersect(lightRay, &lightIsect) && lightIsect.lightIdx != -1){
					float3 p = lightIsect.pos;
					float3 n = lightIsect.nor;
					float3 radiance = { 0.f, 0.f, 0.f };
					radiance = kernel_lights[lightIsect.lightIdx].Le(n, -lightRay.d);
					if (!IsBlack(radiance)){
						float pdfA, pdfW;
						kernel_lights[lightIsect.lightIdx].Pdf(Ray(p, -out, nullptr, kernel_epsilon), n, pdfA, pdfW);
						float choicePdf = PdfFromLightDistribution(lightIsect.lightIdx);
						float lenSquare = dot(p - pos, p - pos);
						float costheta = fabs(dot(n, lightRay.d));
						float lPdf = pdfA * lenSquare / (costheta);
						float weight = PowerHeuristic(1, pdf, 1, lPdf * choicePdf);

						Ld += weight * fr * radiance * fabs(dot(out, nor)) / pdf;
					}
				}

			}
		}

		//light vp
		if (bounces == 0 || (specular && isect.lightIdx != -1)){
			Ld += kernel_lights[isect.lightIdx].Le(nor, -ray.d);
		}

		if (!IsNan(Ld)) vp.ld += beta*Ld;

		//delta material should be more careful
		if (IsDelta(mat.type) || (IsGlossy(mat.type) && mat.alphaU < 0.2f)){
			float3 fr, out;
			float pdf;
			float3 uniformBsdf = make_float3(uniform(rng), uniform(rng), uniform(rng));
			SampleBSDF(mat, -ray.d, nor, uv, dpdu, uniformBsdf, out, fr, pdf);
			if (IsBlack(fr)) return;

			beta *= fr*fabs(dot(out, nor)) / pdf;
			specular = IsDelta(mat.type);

			ray = Ray(pos, out, nullptr, kernel_epsilon);

			continue;
		}

		vp.beta = beta;
		vp.dir = -ray.d;
		vp.isect = isect;
		vp.valid = true;

		break;
	};
}

__device__ void TracePhoton(int maxDepth, thrust::uniform_real_distribution<float>& uniform, thrust::default_random_engine& rng){
	float3 beta = { 1.f, 1.f, 1.f };
	float choicePdf;
	int idx = LookUpLightDistribution(uniform(rng), choicePdf);
	Area light = kernel_lights[idx];
	float3 radiance, lightNor;
	float4 lightUniform = { uniform(rng), uniform(rng), uniform(rng), uniform(rng) };
	Ray ray;
	float pdfA, pdfW;
	light.SampleLight(lightUniform, ray, lightNor, radiance, pdfA, pdfW, kernel_epsilon);
	beta *= radiance*fabs(dot(lightNor, ray.d)) / (pdfA*pdfW*choicePdf);

	Intersection isect;
	for (int bounces = 0; bounces < maxDepth; ++bounces){
		if (!Intersect(ray, &isect)){
			break;
		}

		float3 pos = isect.pos;
		float3 nor = isect.nor;
		float2 uv = isect.uv;
		float3 dpdu = isect.dpdu;
		Material mat = kernel_materials[isect.matIdx];
		if (bounces > 0){//bounces = 0 are already taken into account
			float3 gridCoord;
			BBox gridBounds(boundsMin, boundsMax);
			if (ToGrid(pos, gridBounds, gridRes, gridCoord)){
				int h = Hash(gridCoord.x, gridCoord.y, gridCoord.z, hashSize);
				int start = vpOffset[h];
				int vpSize = vpOffset[h + 1] - start;
				for (int i = 0; i < vpSize; ++i){
					int idx = vpIdx[start + i];
					VisiblePoint& vp = vps[idx];
					if (!vp.valid) continue;
					float3 out = pos - vp.isect.pos;
					float distanceSquare = dot(out, out);
					if (distanceSquare > vp.radius*vp.radius) continue;
					Material vpMat = kernel_materials[vp.isect.matIdx];
					float3 fr;
					float pdf;
					Fr(vpMat, vp.dir, -ray.d, vp.isect.nor, vp.isect.uv, vp.isect.dpdu, fr, pdf);
					if (IsBlack(fr) || IsNan(fr)) continue;
					float3 b = fr * beta * vp.beta;
					b += vp.tau;

					//suppose just a photon hit the same visible point at the same time
					float alpha = 0.7f;
					float g = (vp.n + alpha) / (vp.n + 1.f);
					float rnew = vp.radius*sqrt(g);
					vp.tau = b*g;
					vp.n += alpha;
					vp.radius = rnew;
				}
			}
		}

		float3 fr, out;
		float3 bsdfUniform = make_float3(uniform(rng), uniform(rng), uniform(rng));
		float pdf;
		SampleBSDF(mat, -ray.d, nor, uv, dpdu, bsdfUniform, out, fr, pdf, TransportMode::Importance);
		if (pdf == 0) break;

		beta *= fr*fabs(dot(nor, out)) / pdf;

		ray = Ray(pos, out, nullptr, kernel_epsilon);

		if (bounces > 3){
			float illumate = clamp(1.f - Luminance(beta), 0.f, 1.f);
			if (uniform(rng) < illumate)
				break;

			beta /= (1 - illumate);
		}
	}
}

__global__ void StochasticProgressivePhotonmapperInit(VisiblePoint* v, int* offset){
	vps = v;
	vpOffset = offset;
}

//first pass trace eye ray
__global__ void StochasticProgressivePhotonmapperFP(int iter, int maxDepth, float initRadius = 0.5f){
	unsigned x = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned y = blockIdx.y*blockDim.y + threadIdx.y;
	unsigned pixel = x + y*blockDim.x*gridDim.x;

	//init seed
	int threadIndex = (blockIdx.x + blockIdx.y * gridDim.x) * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;
	thrust::default_random_engine rng(WangHash(pixel) + WangHash(iter));
	thrust::uniform_real_distribution<float> uniform(0.0f, 1.0f);

	float offsetx = uniform(rng) - 0.5f;
	float offsety = uniform(rng) - 0.5f;
	float unuse;
	//ppm doesn't support dof now
	//float2 aperture = UniformDisk(uniform(rng), uniform(rng), unuse);//for dof
	Ray ray = kernel_camera->GeneratePrimaryRay(x + offsetx, y + offsety, make_float2(0, 0));
	ray.tmin = kernel_epsilon;

	TraceRay(pixel, ray, iter, maxDepth, initRadius, uniform, rng);
}

//build hash table for vp
void StochasticProgressivePhotonmapperBuildHashTable(int width, int height){
	BuildHashTable(width, height);
}

//second pass trace photon
__global__ void StochasticProgressivePhotonmapperSP(int iter, int maxDepth){
	unsigned x = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned y = blockIdx.y*blockDim.y + threadIdx.y;
	unsigned pixel = x + y*blockDim.x*gridDim.x;

	//init seed
	int threadIndex = (blockIdx.x + blockIdx.y * gridDim.x) * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;
	thrust::default_random_engine rng(WangHash(pixel) + WangHash(iter*iter));
	thrust::uniform_real_distribution<float> uniform(0.0f, 1.0f);

	TracePhoton(maxDepth, uniform, rng);
}

//third pass density evaluate
__global__ void StochasticProgressivePhotonmapperTP(int iter, int photonsPerIteration){
	unsigned x = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned y = blockIdx.y*blockDim.y + threadIdx.y;
	unsigned pixel = x + y*blockDim.x*gridDim.x;

	VisiblePoint& vp = vps[pixel];

	float3 L = { 0.f, 0.f, 0.f };
	if (vp.valid){
		//as the number of iterations increases, the radius becomes
		//smaller and samller, eventually producing infinity indirect
		float3 indirect = vp.tau / (PI*vp.radius*vp.radius*photonsPerIteration*iter);
		//skip if color is not a number
		if (IsNan(indirect) || IsInf(indirect)) indirect = vp.ind;
		vp.ind = indirect;
		L = vp.ld / iter + indirect;
	}
	kernel_color[pixel] = L;
}
//**************************SPPM End********************************

//**************************Instant Radiosity Integrator************
#define IR_MAX_VPLS 32
struct Vpl{
	float3 beta;
	float3 dir;
	float3 pos;
	float3 nor;
	float2 uv;
	float3 dpdu;
	int matIdx;
};

__device__ Vpl vpls[IR_MAX_VPLS][IR_MAX_VPLS];
__device__ int numVpls[IR_MAX_VPLS];
int vplIter = IR_MAX_VPLS;

__global__ void GenerateVpl(int iter, int maxDepth){
	unsigned x = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned y = blockIdx.y*blockDim.y + threadIdx.y;
	unsigned pixel = x + y*blockDim.x*gridDim.x;

	thrust::default_random_engine rng(WangHash(pixel) + WangHash(iter));
	thrust::uniform_real_distribution<float> uniform(0.0f, 1.0f);

	numVpls[pixel] = 0;
	float3 beta = { 1.f, 1.f, 1.f };
	float choicePdf;
	int idx = LookUpLightDistribution(uniform(rng), choicePdf);
	Area light = kernel_lights[idx];
	float3 radiance, lightNor;
	Ray ray;
	float4 lightUniform = make_float4(uniform(rng), uniform(rng), uniform(rng), uniform(rng));
	float pdfA, pdfW;
	light.SampleLight(lightUniform, ray, lightNor, radiance, pdfA, pdfW, kernel_epsilon);
	beta *= radiance*fabs(dot(lightNor, ray.d)) / (pdfA*pdfW*choicePdf);
	{
		Vpl vpl;
		vpl.beta = radiance;
		vpl.dir.x = pdfA*choicePdf;
		vpl.pos = ray.o;
		vpl.nor = lightNor;
		vpls[pixel][numVpls[pixel]++] = vpl;
	}

	Intersection isect;
	for (int bounces = 0; bounces < maxDepth; ++bounces){
		if (!Intersect(ray, &isect)){
			break;
		}

		float3 pos = isect.pos;
		float3 nor = isect.nor;
		float2 uv = isect.uv;
		float3 dpdu = isect.dpdu;
		Material mat = kernel_materials[isect.matIdx];

		{
			Vpl vpl;
			vpl.beta = beta;
			vpl.dir = -ray.d;
			vpl.pos = isect.pos;
			vpl.nor = isect.nor;
			vpl.uv = isect.uv;
			vpl.dpdu = isect.dpdu;
			vpl.matIdx = isect.matIdx;
			vpls[pixel][numVpls[pixel]++] = vpl;
		}

		float3 fr, out;
		float3 bsdfUniform = make_float3(uniform(rng), uniform(rng), uniform(rng));
		float pdf;
		SampleBSDF(mat, -ray.d, nor, uv, dpdu, bsdfUniform, out, fr, pdf, TransportMode::Importance);
		if (IsBlack(fr)) break;

		beta *= fr*fabs(dot(out, nor)) / pdf;

		ray = Ray(pos, out, nullptr, kernel_epsilon);

		if (bounces > 3){
			float illumate = clamp(1.f - Luminance(beta), 0.f, 1.f);
			if (uniform(rng) < illumate)
				break;

			beta /= (1 - illumate);
		}
	}
}

__global__ void InstantRadiosity(int iter, int vplIter, int maxDepth, float bias){
	unsigned x = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned y = blockIdx.y*blockDim.y + threadIdx.y;
	unsigned pixel = x + y*blockDim.x*gridDim.x;

	thrust::default_random_engine rng(WangHash(pixel) + WangHash(iter));
	thrust::uniform_real_distribution<float> uniform(0.0f, 1.0f);

	float offsetx = uniform(rng) - 0.5f;
	float offsety = uniform(rng) - 0.5f;
	float unuse;
	float2 aperture = UniformDisk(uniform(rng), uniform(rng), unuse);//for dof
	Ray ray = kernel_camera->GeneratePrimaryRay(x + offsetx, y + offsety, aperture);
	ray.tmin = kernel_epsilon;
	float3 beta = { 1.f, 1.f, 1.f };
	float3 L = { 0.f, 0.f, 0.f };

	Intersection isect;
	for(int bounces = 0; bounces < maxDepth; ++ bounces){
		if (!Intersect(ray, &isect)) break;
		if (isect.lightIdx != -1){
			L += kernel_lights[isect.lightIdx].Le(isect.nor, -ray.d);
		}
		float3 pos = isect.pos;
		float3 nor = isect.nor;
		float2 uv = isect.uv;
		float3 dpdu = isect.dpdu;
		Material mat = kernel_materials[isect.matIdx];
		if (IsDelta(mat.type)){
			float3 fr, out;
			float3 bsdfUniform = make_float3(uniform(rng), uniform(rng), uniform(rng));
			float pdf;
			SampleBSDF(mat, -ray.d, nor, uv, dpdu, bsdfUniform, out, fr, pdf);
			if (IsBlack(fr)) break;
			beta *= fr*fabs(dot(nor, out)) / pdf;

			ray = Ray(pos, out, nullptr, kernel_epsilon);
			continue;
		}

		for (int i = 0; i < numVpls[vplIter]; ++i){
			Vpl vpl = vpls[vplIter][i];

			float3 dir = pos - vpl.pos;
			float3 out = normalize(dir);
			float squreDistance = dot(dir, dir);
			Ray shadowRay(pos, -out, nullptr, kernel_epsilon, sqrt(squreDistance) - kernel_epsilon);
			if (IntersectP(shadowRay)) continue;

			if (squreDistance < bias) squreDistance = bias;
			float c1 = fabs(dot(out, nor));
			float c2 = fabs(dot(out, vpl.nor));
			float G = c1*c2 / squreDistance;
			float3 fr1, fr2;
			float pdf1, pdf2;
			Fr(mat, -ray.d, -out, nor, uv, dpdu, fr1, pdf1);
			if (i == 0){
				if (dot(dir, vpl.nor) > 0.f)
					L += beta*fr1*G*vpl.beta / vpl.dir.x;
				continue;
			}
			Material m = kernel_materials[vpl.matIdx];
			if (IsDelta(m.type)) continue;
			Fr(m, vpl.dir, out, vpl.nor, vpl.uv, vpl.dpdu, fr2, pdf2);

			L += beta*fr1*G*fr2*vpl.beta;
		}

		break;
	} 

	if (IsNan(L) || IsInf(L)) return;

	kernel_color[pixel] = L;
}
//**************************Instant Radiosity Integrator End********

__global__ void Output(int iter, float3* output, bool reset, bool filmic, IntegratorType type){
	unsigned x = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned y = blockIdx.y*blockDim.y + threadIdx.y;
	unsigned pixel = x + y*blockDim.x*gridDim.x;

	if (reset) kernel_acc_image[pixel] = { 0, 0, 0 };

	float3 color = kernel_color[pixel];
	if (type != IT_SPPM){
		kernel_acc_image[pixel] += color;
		color = kernel_acc_image[pixel] / iter;
	}
	if (filmic) FilmicTonemapping(color);
	else GammaCorrection(color);
	output[pixel] = color;
}

__global__ void InitRender(
	Camera* camera,
	LinearBVHNode* bvh_nodes,
	Primitive* primitives,
	Material* materials,
	Bssrdf* bssrdfs,
	Medium* mediums,
	Area* lights,
	Infinite* infinite,
	uchar4** texs,
	float* light_distribution,
	int light_size,
	int ld_size,
	int* tex_size,
	float3* image,
	float3* color,
	float ep){
	kernel_camera = camera;
	kernel_linear = bvh_nodes;
	kernel_primitives = primitives;
	kernel_materials = materials;
	kernel_bssrdfs = bssrdfs;
	kernel_mediums = mediums;
	kernel_lights = lights;
	kernel_infinite = infinite;
	kernel_textures = texs;
	kernel_light_distribution = light_distribution;
	kernel_light_size = light_size;
	kernel_light_distribution_size = ld_size;
	kernel_texture_size = tex_size;
	kernel_acc_image = image;
	kernel_color = color;
	kernel_epsilon = ep;
}

void BeginRender(
	Scene& scene,
	unsigned width,
	unsigned height,
	float ep){
	int mesh_memory_use = 0;
	int material_memory_use = 0;
	int bvh_memory_use = 0;
	int light_memory_use = 0;
	int texture_memory_use = 0;
	int num_primitives = scene.bvh.prims.size();
	HANDLE_ERROR(cudaMalloc(&dev_camera, sizeof(Camera)));
	HANDLE_ERROR(cudaMemcpy(dev_camera, scene.camera, sizeof(Camera), cudaMemcpyHostToDevice));

	if (num_primitives){
		HANDLE_ERROR(cudaMalloc(&dev_primitives, num_primitives*sizeof(Primitive)));
		HANDLE_ERROR(cudaMemcpy(dev_primitives, &scene.bvh.prims[0], num_primitives*sizeof(Primitive), cudaMemcpyHostToDevice));
		mesh_memory_use += num_primitives*sizeof(Primitive);
	}
	if (scene.bvh.total_nodes > 0){
		HANDLE_ERROR(cudaMalloc(&dev_bvh_nodes, scene.bvh.total_nodes*sizeof(LinearBVHNode)));
		HANDLE_ERROR(cudaMemcpy(dev_bvh_nodes, scene.bvh.linear_root, scene.bvh.total_nodes*sizeof(LinearBVHNode), cudaMemcpyHostToDevice));
		bvh_memory_use += scene.bvh.total_nodes*sizeof(LinearBVHNode);
	}

	//copy material
	int num_materials = scene.materials.size();
	HANDLE_ERROR(cudaMalloc(&dev_materials, num_materials*sizeof(Material)));
	HANDLE_ERROR(cudaMemcpy(dev_materials, &scene.materials[0], num_materials*sizeof(Material), cudaMemcpyHostToDevice));
	material_memory_use += num_materials*sizeof(Material);

	int num_bssrdfs = scene.bssrdfs.size();
	if (num_bssrdfs){
		HANDLE_ERROR(cudaMalloc(&dev_bssrdfs, num_bssrdfs*sizeof(Bssrdf)));
		HANDLE_ERROR(cudaMemcpy(dev_bssrdfs, &scene.bssrdfs[0], num_bssrdfs*sizeof(Bssrdf), cudaMemcpyHostToDevice));
		material_memory_use += num_bssrdfs*sizeof(Bssrdf);
	}

	int num_mediums = scene.mediums.size();
	if (num_mediums){
		HANDLE_ERROR(cudaMalloc(&dev_mediums, num_mediums*sizeof(Medium)));
		HANDLE_ERROR(cudaMemcpy(dev_mediums, &scene.mediums[0], num_mediums*sizeof(Medium), cudaMemcpyHostToDevice));
		material_memory_use += num_mediums*sizeof(Medium);
	}
	//copy heterogeneous density data
	for (int i = 0; i < num_mediums; ++i){
		if (scene.mediums[i].type == MT_HETEROGENEOUS){
			Heterogeneous m = scene.mediums[i].heterogeneous;
			float* density;
			HANDLE_ERROR(cudaMalloc(&density, m.nx*m.ny*m.nz*sizeof(float)));
			material_memory_use += m.nx*m.ny*m.nz*sizeof(float);
			HANDLE_ERROR(cudaMemcpy(density, m.density, m.nx*m.ny*m.nz*sizeof(float), cudaMemcpyHostToDevice));
			HANDLE_ERROR(cudaMemcpy(&dev_mediums[i].heterogeneous.density, &density, sizeof(float*), cudaMemcpyHostToDevice));
			delete[] m.density;
		}
	}

	//copy light
	int num_lights = scene.lights.size();
	if (num_lights){
		HANDLE_ERROR(cudaMalloc(&dev_lights, num_lights*sizeof(Area)));
		HANDLE_ERROR(cudaMemcpy(dev_lights, &scene.lights[0], num_lights*sizeof(Area), cudaMemcpyHostToDevice));
		light_memory_use += num_lights*sizeof(Area);
	}

	//copy infinite light
	HANDLE_ERROR(cudaMalloc(&dev_infinite, sizeof(Infinite)));
	HANDLE_ERROR(cudaMemcpy(dev_infinite, &scene.infinite, sizeof(Infinite), cudaMemcpyHostToDevice));
	if (scene.infinite.isvalid){
		int width = scene.infinite.width, height = scene.infinite.height;
		float3* data;
		HANDLE_ERROR(cudaMalloc(&data, width*height*sizeof(float3)));
		texture_memory_use += width*height*sizeof(float3);
		HANDLE_ERROR(cudaMemcpy(data, scene.infinite.data, width*height*sizeof(float3), cudaMemcpyHostToDevice));
		HANDLE_ERROR(cudaMemcpy(&dev_infinite->data, &data, sizeof(float3*), cudaMemcpyHostToDevice));
	}

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

	int ld_size = scene.lightDistribution.size();
	HANDLE_ERROR(cudaMalloc(&dev_light_distribution, ld_size*sizeof(float)));
	HANDLE_ERROR(cudaMemcpy(dev_light_distribution, &scene.lightDistribution[0], ld_size*sizeof(float), cudaMemcpyHostToDevice));
	texture_memory_use += ld_size*sizeof(float);
	
	InitRender << <1, 1 >> >(dev_camera, dev_bvh_nodes,
		dev_primitives, dev_materials, dev_bssrdfs, dev_mediums, dev_lights, dev_infinite, dev_textures, dev_light_distribution, num_lights, ld_size,
		texture_size, dev_image, dev_color, ep);

	//init for progressive photon mapper
	if (scene.integrator.type == IT_SPPM){
		HANDLE_ERROR(cudaMalloc(&device_vps, width*height*sizeof(VisiblePoint)));
		HANDLE_ERROR(cudaMalloc(&device_vpOffset, (width*height + 1)*sizeof(int)));
		HANDLE_ERROR(cudaMalloc(&device_vpIdx, sizeof(int)));

		StochasticProgressivePhotonmapperInit << <1, 1 >> >(device_vps, device_vpOffset);
	}

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
}

void Render(Scene& scene, unsigned width, unsigned height, Camera* camera, unsigned iter, bool reset, float3* output){
	HANDLE_ERROR(cudaMemcpy(dev_camera, camera, sizeof(Camera), cudaMemcpyHostToDevice));
	int block_x = 32, block_y = 4;
	dim3 block(block_x, block_y);
	dim3 grid(width / block.x, height / block.y);

	IntegratorType type = scene.integrator.type;
	if (type == IT_AO)
		Ao << <grid, block >> >(iter, scene.integrator.maxDist);
	else if (type == IT_PT)
		Path << <grid, block >> >(iter, scene.integrator.maxDepth);
	else if (type == IT_VPT)
		Volpath << <grid, block >> >(iter, scene.integrator.maxDepth);
	else if (type == IT_LT){
		LightTracingInit << <grid, block >> >();
		LightTracing << <grid, block >> >(iter, scene.integrator.maxDepth);
	}
	else if (type == IT_BDPT){
		BdptInit << <grid, block >> >();
		Bdpt << <grid, block >> >(iter, scene.integrator.maxDepth);
	}
	else if (type == IT_SPPM){
		StochasticProgressivePhotonmapperFP << <grid, block >> >(iter, scene.integrator.maxDepth,
			scene.integrator.initRadius);

		//build hash grid on cpu
		StochasticProgressivePhotonmapperBuildHashTable(width, height);

		int photonsPerIteration = scene.integrator.photonsPerIteration;
		StochasticProgressivePhotonmapperSP << < photonsPerIteration / 10, 10 >> >(iter, scene.integrator.maxDepth);
		
		StochasticProgressivePhotonmapperTP << <grid, block >> >(iter, photonsPerIteration);
	}
	else if (type == IT_IR){
		if (vplIter == IR_MAX_VPLS){
			vplIter = 0;
			GenerateVpl << <IR_MAX_VPLS, 1 >> >(iter, scene.integrator.maxDepth);
		}
		InstantRadiosity << <grid, block >> >(iter, vplIter, scene.integrator.maxDepth, scene.integrator.vplBias);
		vplIter++;
	}

	grid.x = width / block.x;
	grid.y = height / block.y;
	Output << <grid, block >> >(iter, output, reset, camera->filmic, type);
}
