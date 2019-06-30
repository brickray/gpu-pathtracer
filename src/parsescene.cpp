#include "parsescene.h"
#include "primitive.h"
#include "area.h"

#include <rapidjson\include\rapidjson\document.h>
#include <rapidjson\include\rapidjson\writer.h>
#include <rapidjson\include\rapidjson\stringbuffer.h>

#include <fstream>
#include <sys\stat.h>

using namespace rapidjson;

int getFileLength(const char* filename){
	struct stat st;
	stat(filename, &st);
	return st.st_size;   //文件长度.
}

mat4 getMat4(Value& value){
	float x[16];
	int i = 0;
	Value::ConstValueIterator it = value.Begin();
	for (; it != value.End(); ++it, ++i){
		x[i] = it->GetDouble();
	}

	mat4 ret;
	memcpy(&ret[0], x, 16 * sizeof(float));
	return ret;
}

float3 getFloat3(Value& value){
	float x[3];
	int i = 0;

	Value::ConstValueIterator it = value.Begin();
	for (; it != value.End(); ++it, ++i){
		x[i] = it->GetDouble();
	}

	return{ x[0], x[1], x[2] };
}

bool LoadScene(const char* filename, GlobalConfig& config, Scene& scene){
	string file = filename;
	string base = file.substr(0, file.find_last_of('/') + 1);
	fstream in(filename);
	if (!in.good()){
		fprintf(stderr, "Scene file [\"%s\"] is not good\n", filename);
		return false;
	}

	int size = getFileLength(filename);
	char* buffer = new char[size + 1];
	memset(buffer, 0, size + 1);
	in.read(buffer, size);
	in.close();

	Document doc;
	doc.Parse(buffer);
	if (doc.HasParseError())
	{
		fprintf(stderr, "Parse scene error: %d\n", doc.GetParseError());
		return false;
	}

	Value::MemberIterator it = doc.MemberBegin();
	Value::MemberIterator ited = doc.MemberEnd();

	vector<string> mediumName;
	//parse medium
	{
		if (doc.HasMember("medium")){
			Value& s = doc["medium"];
			if (!s.IsArray()){
				fprintf(stderr, "Invalid Medium format\n");
				return false;
			}
			Value::ValueIterator it = s.Begin();
			for (; it != s.End(); ++it){
				Value& unit = *it;
				string type = (*it).HasMember("type") ? (*it)["type"].GetString() : "homogeneous";
				string name = (*it)["name"].GetString();
				float3 sigmaA = (*it).HasMember("sigmaA") ? getFloat3((*it)["sigmaA"]) : make_float3(1.f, 1.f, 1.f);
				float3 sigmaS = (*it).HasMember("sigmaS") ? getFloat3((*it)["sigmaS"]) : make_float3(1.f, 1.f, 1.f);
				float g = (*it).HasMember("g") ? (*it)["g"].GetDouble() : 0.f;
				float scale = (*it).HasMember("scale") ? (*it)["scale"].GetDouble() : 1.f;
				int iterMax = (*it).HasMember("iterMax") ? (*it)["iterMax"].GetInt() : 1000;
				sigmaA *= scale;
				sigmaS *= scale;
				Medium medium;
				medium.g = g;
				if (type == "homogeneous"){
					medium.type = MT_HOMOGENEOUS;
					medium.homogeneous.sigmaA = sigmaA;
					medium.homogeneous.sigmaS = sigmaS;
					medium.homogeneous.sigmaT = sigmaA + sigmaS;
				}
				else{
					float3 sigmaT = sigmaA + sigmaS;
					if (sigmaT.x != sigmaT.y || sigmaT.x != sigmaT.z){
						fprintf(stderr, "sigmaA and sigmaS requires uniform attenuation coefficient\n");
						exit(1);
					}
					int nx = (*it)["nx"].GetInt();
					int ny = (*it)["ny"].GetInt();
					int nz = (*it)["nz"].GetInt();
					float3 p0 = getFloat3((*it)["p0"]);
					float3 p1 = getFloat3((*it)["p1"]);
					string file = (*it)["density"].GetString();
					medium.type = MT_HETEROGENEOUS;
					medium.heterogeneous.sigmaA = sigmaA;
					medium.heterogeneous.sigmaS = sigmaS;
					medium.heterogeneous.sigmaT = sigmaT;
					medium.heterogeneous.nx = nx;
					medium.heterogeneous.ny = ny;
					medium.heterogeneous.nz = nz;
					medium.heterogeneous.p0 = p0;
					medium.heterogeneous.p1 = p1;
					medium.heterogeneous.iterMax = iterMax;
					medium.heterogeneous.density = new float[nx*ny*nz];
					ReadDensityFromFile((base + file).c_str(), nx, ny, nz, medium.heterogeneous.density);
					float max = 0.f;
					for (int i = 0; i < nx*ny*nz; ++i){
						if (medium.heterogeneous.density[i] > max)
							max = medium.heterogeneous.density[i];
					}
					medium.heterogeneous.invMaxDensity = 1.f / max;
				}
				scene.mediums.push_back(medium);
				mediumName.push_back(name);
			}
		}
	}

	auto getMedium = [mediumName](string m)->int{
		int mi = -1;
		for (int i = 0; i < mediumName.size(); ++i){
			if (mediumName[i] == m){
				mi = i;
				break;
			}
		}
		return mi;
	};
	//global config
	{
		if (doc.HasMember("screen_width") && doc.HasMember("screen_height")){
			config.width = doc["screen_width"].GetInt();
			config.height = doc["screen_height"].GetInt();
		}
		else{ //default screen size
			config.width = 512;
			config.height = 512;
		}

		config.epsilon = doc.HasMember("epsilon") ? doc["epsilon"].GetDouble() : 0.001f;

		if (doc.HasMember("camera")){
			Value& camera = doc["camera"];
			config.camera.environment = camera.HasMember("environment") ? camera["environment"].GetBool() : false;
			config.camera.position = camera.HasMember("position") ? getFloat3(camera["position"]) : make_float3(0, 0, 0);
			config.camera.fov = camera.HasMember("fov") ? camera["fov"].GetDouble() : 60.f;
			float3 up = camera.HasMember("up") ? getFloat3(camera["up"]) : make_float3(0, 1, 0);
			float3 lookat = camera.HasMember("lookat") ? getFloat3(camera["lookat"]) : make_float3(0, 0, -1);
			config.camera.Lookat(config.camera.position, lookat, up);
			config.camera.apertureRadius = camera.HasMember("apertureRadius") ? camera["apertureRadius"].GetDouble() : 0.f;
			config.camera.focalDistance = camera.HasMember("focalDistance") ? camera["focalDistance"].GetDouble() : 0.f;
			config.camera_move_speed = camera.HasMember("move_speed") ? camera["move_speed"].GetDouble() : 0.1f;
			config.camera.filmic = camera.HasMember("filmicTonemap") ? camera["filmicTonemap"].GetBool() : true;
			string m = camera.HasMember("medium") ? camera["medium"].GetString() : "";
			config.camera.medium = getMedium(m);
		}
		else{
			fprintf(stderr, "Scene file must define camera\n");
			return false;
		}
	}

	//integrator
	{
		string integrator = doc.HasMember("integrator") ? doc["integrator"].GetString():"pt";
		if (integrator == "ao"){
			scene.integrator.type = IT_AO;
			scene.integrator.maxDist = doc.HasMember("maxDist") ? doc["maxDist"].GetDouble() : 0.5f;
		}
		else if (integrator == "pt"){
			scene.integrator.type = IT_PT;
			scene.integrator.maxDepth = doc.HasMember("maxDepth") ? doc["maxDepth"].GetInt() : 5;
		}
		else if (integrator == "vpt"){
			scene.integrator.type = IT_VPT;
			scene.integrator.maxDepth = doc.HasMember("maxDepth") ? doc["maxDepth"].GetInt() : 5;
		}
		else if (integrator == "lt"){
			scene.integrator.type = IT_LT;
			scene.integrator.maxDepth = doc.HasMember("maxDepth") ? doc["maxDepth"].GetInt() : 5;
		}
		else if (integrator == "bdpt"){
			scene.integrator.type = IT_BDPT;
			scene.integrator.maxDepth = doc.HasMember("maxDepth") ? doc["maxDepth"].GetInt() : 5;
		}
		else if (integrator == "mlt"){
			scene.integrator.type = IT_MLT;
			scene.integrator.maxDepth = doc.HasMember("maxDepth") ? doc["maxDepth"].GetInt() : 5;
		}
		else if (integrator == "sppm"){
			scene.integrator.type = IT_SPPM;
			scene.integrator.maxDepth = doc.HasMember("maxDepth") ? doc["maxDepth"].GetInt() : 5;
			scene.integrator.initRadius = doc.HasMember("initRadius") ? doc["initRadius"].GetDouble() : 0.5f;
			scene.integrator.photonsPerIteration = doc.HasMember("photonsPerIteration") ? doc["photonsPerIteration"].GetInt() : 100000;
		}
		else if (integrator == "ir"){
			scene.integrator.type = IT_IR;
			scene.integrator.maxDepth = doc.HasMember("maxDepth") ? doc["maxDepth"].GetInt() : 5;
			scene.integrator.vplBias = doc.HasMember("vplBias") ? doc["vplBias"].GetDouble() : 0.5f;
		}
		else{
			printf("Unsupport integrator [%s]\n", integrator);
			printf("Choose one of them[ao, pt, vpt, lt, bdpt, mlt(underdeveloped), sppm, ir]\n");
			exit(1);
		}
	}

	vector<string> matName;
	vector<string> bssrdfName;
	//parse material
	{
		if (doc.HasMember("material")){
			Value& mat = doc["material"];
			if (!mat.IsArray()){
				fprintf(stderr, "Invalid material format\n");
				return false;
			}

			map<string, MaterialType> matMap;
			matMap["lambertian"] = MT_LAMBERTIAN;
			matMap["mirror"] = MT_MIRROR;
			matMap["dielectric"] = MT_DIELECTRIC;
			matMap["roughdielectric"] = MT_ROUGHDIELECTRIC;
			matMap["roughconduct"] = MT_ROUGHCONDUCTOR;
			matMap["substrate"] = MT_SUBSTRATE;
			map<string, int> texMap;

			Value::ValueIterator it = mat.Begin();
			for (; it != mat.End(); ++it){
				if ((*it).HasMember("bssrdf")){
					string mat_name = (*it)["name"].GetString();
					float3 sigmaA = (*it).HasMember("sigmaA") ? getFloat3((*it)["sigmaA"]) : make_float3(1.0, 1.0, 1.0);
					float3 sigmaSP = (*it).HasMember("sigmaSP") ? getFloat3((*it)["sigmaSP"]) : make_float3(1.0, 1.0, 1.0);
					float eta = (*it).HasMember("eta") ? (*it)["eta"].GetDouble() : 1.5f;
					float g = (*it).HasMember("g") ? (*it)["g"].GetDouble() : 0.f;
					float scale = (*it).HasMember("scale") ? (*it)["scale"].GetDouble() : 1.f;
					sigmaA *= scale;
					sigmaSP *= scale;
					Bssrdf bssrdf(sigmaA, sigmaSP, eta, g);
					if ((*it).HasMember("kd")){
						float3 kd = getFloat3((*it)["kd"]);
						float meanPathLength = (*it).HasMember("meanPathLength") ? (*it)["meanPathLength"].GetDouble() : 1.f;
						bssrdf.ConvertFromDiffuse(kd, meanPathLength, eta);
					}
					float3 aa = bssrdf.GetSigmaT();
					scene.bssrdfs.push_back(bssrdf);
					bssrdfName.push_back(mat_name);
					continue;
				}
				string mat_name = (*it)["name"].GetString();
				string bsdf = (*it)["bsdf"].GetString();
				float alphaU, alphaV;
				if ((*it).HasMember("alpha")){
					alphaU = (*it)["alpha"].GetDouble();
					alphaV = alphaU;
				}
				else{
					alphaU = (*it).HasMember("alphaU") ? (*it)["alphaU"].GetDouble() : 0.01f;
					alphaV = (*it).HasMember("alphaV") ? (*it)["alphaV"].GetDouble() : 0.01f;
				}
				bool remap = (*it).HasMember("remap") ? (*it)["remap"].GetBool() : false;
				if (remap){
					auto Remap = [](float roughness)->float{
						roughness = std::max(roughness, (float)1e-3);
						float x = log(roughness);
						return 1.62142f + 0.819955f * x + 0.1734f * x * x +
							0.0171201f * x * x * x + 0.000640711f * x * x * x * x;
					};
					alphaU = Remap(alphaU);
					alphaV = Remap(alphaV);
				}
				float insideIOR = (*it).HasMember("insideIOR") ? (*it)["insideIOR"].GetDouble() : 1.f;
				float outsideIOR = (*it).HasMember("outsideIOR") ? (*it)["outsideIOR"].GetDouble() : 1.f;
				float3 k = (*it).HasMember("k") ? getFloat3((*it)["k"]) : make_float3(0, 0, 0);
				float3 eta = (*it).HasMember("eta") ? getFloat3((*it)["eta"]) : make_float3(0, 0, 0);
				float3 specular = (*it).HasMember("specular") ? getFloat3((*it)["specular"]) : make_float3(1.f, 1.f, 1.f);
				float3 diffuse = make_float3(1.f, 1.f, 1.f);
				int texIdx = -1;
				if ((*it).HasMember("diffuse")){
					if ((*it)["diffuse"].IsString()){
						string file = (*it)["diffuse"].GetString();
						if (texMap.find(file) == texMap.end()){
							Texture tex((base + file).c_str());
							scene.textures.push_back(tex);
							texIdx = scene.textures.size() - 1;
							texMap[file] = scene.textures.size() - 1;
						}
						else
							texIdx = texMap[file];
					}
					else{
						diffuse = getFloat3((*it)["diffuse"]);
					}
				}
				Material mat;
				mat.type = matMap[bsdf];
				mat.alphaU = alphaU;
				mat.alphaV = alphaV;
				mat.insideIOR = insideIOR;
				mat.outsideIOR = outsideIOR;
				mat.k = k;
				mat.eta = eta;
				mat.diffuse = diffuse;
				mat.specular = specular;
				mat.textureIdx = texIdx;
				scene.materials.push_back(mat);
				matName.push_back(mat_name);
			}
		}
	}

	//parse scene
	{
		if (doc.HasMember("scene")){
			Value& s = doc["scene"];
			Value::ValueIterator it = s.Begin();
			for (; it != s.End(); ++it){
				Value& unit = *it;
				if (unit.HasMember("mesh")){
					string file = unit["mesh"].GetString();
					string mat_name = unit.HasMember("material") ? unit["material"].GetString() : "";
					float3 scale = unit.HasMember("scale") ? getFloat3(unit["scale"]) : make_float3(1, 1, 1);
					float3 translate = unit.HasMember("translate") ? getFloat3(unit["translate"]) : make_float3(0, 0, 0);
					float3 rotate = unit.HasMember("rotate") ? getFloat3(unit["rotate"]) : make_float3(0, 0, 0);
					string mediumInside = unit.HasMember("inside") ? unit["inside"].GetString() : "";
					string mediumOutside = unit.HasMember("outside") ? unit["outside"].GetString() : "";
					int mi = getMedium(mediumInside);
					int mo = getMedium(mediumOutside);
					mat4 trs, t, r, s;
					s = glm::scale(s, vec3(scale.x, scale.y, scale.z));
					t = glm::translate(t, vec3(translate.x, translate.y, translate.z));
					r = glm::rotate(r, radians(rotate.x), vec3(1, 0, 0));
					r = glm::rotate(r, radians(rotate.y), vec3(0, 1, 0));
					r = glm::rotate(r, radians(rotate.z), vec3(0, 0, 1));
					trs = t*r*s;
					Mesh mesh;
					mesh.matIdx = -1;
					mesh.bssrdfIdx = -1;


					if (mat_name != "" || !(mi != -1 || mo != -1)){
						int i;
						for (i = 0; i < matName.size(); ++i){
							if (matName[i] == mat_name){
								mesh.matIdx = i;
								break;
							}
						}
						if (i == matName.size()){
							for (i = 0; i < bssrdfName.size(); ++i){
								if (bssrdfName[i] == mat_name){
									mesh.bssrdfIdx = i;
									break;
								}
							}
							if (i == bssrdfName.size()){
								fprintf(stderr, "There is no material named:[\"%s\"]\n", mat_name.c_str());
								exit(1);
							}
						}
					}

					mesh.LoadObjFromFile((base + file).c_str(), aiProcess_Triangulate | aiProcess_GenSmoothNormals, trs);
					for (int i = 0; i < mesh.triangles.size(); ++i){
						Primitive primitive;
						primitive.type = GT_TRIANGLE;
						primitive.triangle = mesh.triangles[i];
						primitive.triangle.mediumInside = mi;
						primitive.triangle.mediumOutside = mo;
						scene.primitives.push_back(primitive);
					}
				}
				else if (unit.HasMember("line")){
					string mat_name = unit.HasMember("material") ? unit["material"].GetString() : "matte";
					float3 p0 = unit.HasMember("p0") ? getFloat3(unit["p0"]) : make_float3(0, 0, 0);
					float3 p1 = unit.HasMember("p1") ? getFloat3(unit["p1"]) : make_float3(1, 1, 1);
					float width0 = unit.HasMember("width0") ? unit["width0"].GetDouble() : 0.025f;
					float width1 = unit.HasMember("width1") ? unit["width1"].GetDouble() : 0.025f;
					float3 scale = unit.HasMember("scale") ? getFloat3(unit["scale"]) : make_float3(1, 1, 1);
					float3 translate = unit.HasMember("translate") ? getFloat3(unit["translate"]) : make_float3(0, 0, 0);
					float3 rotate = unit.HasMember("rotate") ? getFloat3(unit["rotate"]) : make_float3(0, 0, 0);
					mat4 trs, t, r, s;
					s = glm::scale(s, vec3(scale.x, scale.y, scale.z));
					t = glm::translate(t, vec3(translate.x, translate.y, translate.z));
					r = glm::rotate(r, radians(rotate.x), vec3(1, 0, 0));
					r = glm::rotate(r, radians(rotate.y), vec3(0, 1, 0));
					r = glm::rotate(r, radians(rotate.z), vec3(0, 0, 1));
					trs = t*r*s;

					vec3 v0 = Float3ToVec(p0);
					v0 = vec3(trs*vec4(v0, 1));
					p0 = VecToFloat3(v0);
					vec3 v1 = Float3ToVec(p1);
					v1 = vec3(trs*vec4(v1, 1));
					p1 = VecToFloat3(v1);

					Line line;
					line.p0 = p0;
					line.p1 = p1;
					line.width0 = width0;
					line.width1 = width1;
					int i;
					for (i = 0; i < matName.size(); ++i){
						if (matName[i] == mat_name){
							line.matIdx = i;
							break;
						}
					}
					if (i == matName.size()){
						fprintf(stderr, "There is no material named:[\"%s\"]\n", mat_name.c_str());
						exit(1);
					}
					Primitive prim;
					prim.type = GT_LINES;
					prim.line = line;
					scene.primitives.push_back(prim);
				}
				else if (unit.HasMember("sphere")){
					string mat_name = unit.HasMember("material") ? unit["material"].GetString() : "";
					float3 center = unit.HasMember("center") ? getFloat3(unit["center"]) : make_float3(0, 0, 0);
					float radius = unit.HasMember("radius") ? unit["radius"].GetDouble() : 1.f;
					string mediumInside = unit.HasMember("inside") ? unit["inside"].GetString() : "";
					string mediumOutside = unit.HasMember("outside") ? unit["outside"].GetString() : "";
					int mi = getMedium(mediumInside);
					int mo = getMedium(mediumOutside);
					Sphere sphere;
					sphere.origin = center;
					sphere.radius = radius;
					sphere.matIdx = -1;
					sphere.bssrdfIdx = -1;

					if (mat_name != "" || !(mi != -1 || mo != -1)){
						int i;
						for (i = 0; i < matName.size(); ++i){
							if (matName[i] == mat_name){
								sphere.matIdx = i;
								break;
							}
						}
						if (i == matName.size()){
							for (i = 0; i < bssrdfName.size(); ++i){
								if (bssrdfName[i] == mat_name){
									sphere.bssrdfIdx = i;
									break;
								}
							}
							if (i == bssrdfName.size()){
								fprintf(stderr, "There is no material named:[\"%s\"]\n", mat_name.c_str());
								exit(1);
							}
						}
					}

					Primitive prim;
					prim.type = GT_SPHERE;
					prim.sphere = sphere;
					prim.sphere.mediumInside = mi;
					prim.sphere.mediumOutside = mo;
					scene.primitives.push_back(prim);
				}
				else{
					fprintf(stderr, "Error scene file format\n");
					return false;
				}
			}
		}
		else{
			fprintf(stderr, "There is no primitives in the scene\n");
		}
	}

	{
		//parse light
		if (doc.HasMember("light")){
			Value& s = doc["light"];
			Value::ValueIterator it = s.Begin();
			for (; it != s.End(); ++it){
				Value& unit = *it;
				if (unit.HasMember("mesh")){
					//光源不支持medium
					string file = unit["mesh"].GetString();
					string mat_name = unit.HasMember("material") ? unit["material"].GetString() : "matte";
					float3 radiance = unit.HasMember("radiance") ? getFloat3(unit["radiance"]) : make_float3(0.f, 0.f, 0.f);
					string m = unit.HasMember("medium") ? unit["medium"].GetString() : "";
					float3 scale = unit.HasMember("scale") ? getFloat3(unit["scale"]) : make_float3(1, 1, 1);
					float3 translate = unit.HasMember("translate") ? getFloat3(unit["translate"]) : make_float3(0, 0, 0);
					float3 rotate = unit.HasMember("rotate") ? getFloat3(unit["rotate"]) : make_float3(0, 0, 0);
					mat4 trs, t, r, s;
					s = glm::scale(s, vec3(scale.x, scale.y, scale.z));
					t = glm::translate(t, vec3(translate.x, translate.y, translate.z));
					r = glm::rotate(r, radians(rotate.x), vec3(1, 0, 0));
					r = glm::rotate(r, radians(rotate.y), vec3(0, 1, 0));
					r = glm::rotate(r, radians(rotate.z), vec3(0, 0, 1));
					trs = t*r*s;
					Mesh mesh;
					mesh.bssrdfIdx = -1;

					int i;
					for (i = 0; i < matName.size(); ++i){
						if (matName[i] == mat_name){
							mesh.matIdx = i;
							break;
						}
					}
					if (i == matName.size()){
						fprintf(stderr, "There is no material named:[\"%s\"]\n", mat_name.c_str());
						exit(1);
					}

					mesh.LoadObjFromFile((base + file).c_str(), aiProcess_Triangulate | aiProcess_GenSmoothNormals, trs);
					for (int i = 0; i < mesh.triangles.size(); ++i){
						mesh.triangles[i].lightIdx = scene.lights.size();
						Primitive primitive;
						primitive.type = GT_TRIANGLE;
						primitive.triangle = mesh.triangles[i];
						scene.primitives.push_back(primitive);
						Area area;
						area.radiance = radiance;
						area.triangle = mesh.triangles[i];
						area.medium = getMedium(m);
						scene.lights.push_back(area);
					}
				}
				else if (unit.HasMember("infinite")){
					string file = unit["infinite"].GetString();
					int w, h;
					vector<float3> exr;
					if (!ImageIO::LoadExr((base + file).c_str(), w, h, exr)){
						fprintf(stderr, "Couldn't load hdr file \"%s\", only support .exr format\n", file.c_str());
						exit(1);
					}
					vec3 uu, vv, ww;
					if (unit.HasMember("rotate")){
						float3 r = getFloat3(unit["rotate"]);
						mat4 rs;
						rs = rotate(rs, radians(r.x), vec3(1, 0, 0));
						rs = rotate(rs, radians(r.y), vec3(0, 1, 0));
						rs = rotate(rs, radians(r.z), vec3(0, 0, 1));
						uu = vec3(rs * vec4(1, 0, 0, 0));
						vv = vec3(rs * vec4(0, 1, 0, 0));
						ww = vec3(rs * vec4(0, 0, 1, 0));
					}
					if (unit.HasMember("matrix")){
						mat4 rs = getMat4(unit["matrix"]);
						rs = inverse(rs);
						uu = vec3(rs * vec4(1, 0, 0, 0));
						vv = vec3(rs * vec4(0, 1, 0, 0));
						ww = vec3(rs * vec4(0, 0, 1, 0));
					}
					Infinite infinite;
					infinite.u = VecToFloat3(uu);
					infinite.v = VecToFloat3(vv);
					infinite.w = VecToFloat3(ww);
					infinite.width = w;
					infinite.height = h;
					infinite.data = new float3[w*h];
					for (int i = 0; i < w*h; ++i)
						infinite.data[i] = exr[i];
					infinite.isvalid = true;
					scene.infinite = infinite;
				}
				else{
					fprintf(stderr, "Only support area and infinite light\n");
				}
			}
		}
	}

	delete[] buffer;
	return true;
}