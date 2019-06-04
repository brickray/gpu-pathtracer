#include "mesh.h"
#include "scene.h"

void Mesh::LoadObjFromFile(std::string filename, unsigned int flags, mat4& trs){
	Assimp::Importer importer;
	const aiScene* scene = importer.ReadFile(filename, flags);
	if (!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode){
		fprintf(stderr, "Error when improt model: %s\n", importer.GetErrorString());
		exit(1);
	}
	
	processNode(scene->mRootNode, scene, trs);
	fprintf(stdout, "Load Model sucessfully: %s\n", filename.c_str());
	fprintf(stdout, "Merge [%d] triangles\n", triangles.size());
}


void Mesh::processNode(aiNode* node, const aiScene* scene, mat4& trs){
	for (int i = 0; i < node->mNumMeshes; ++i){
		aiMesh* aimesh = scene->mMeshes[node->mMeshes[i]];
		processMesh(aimesh, scene, trs);
	}

	for (int i = 0; i < node->mNumChildren; ++i){
		processNode(node->mChildren[i], scene, trs);
	}
}

void Mesh::processMesh(aiMesh* aimesh, const aiScene* scene, mat4& trs){
	for (int i = 0; i < aimesh->mNumVertices; ++i){
		Vertex vertex;
		vertex.v.x = aimesh->mVertices[i].x;
		vertex.v.y = aimesh->mVertices[i].y;
		vertex.v.z = aimesh->mVertices[i].z;
		vertex.n.x = aimesh->mNormals[i].x;
		vertex.n.y = aimesh->mNormals[i].y;
		vertex.n.z = aimesh->mNormals[i].z;
		if (aimesh->mTextureCoords[0]) {// have tex coordinate
			vertex.uv.x = aimesh->mTextureCoords[0][i].x;
			vertex.uv.y = aimesh->mTextureCoords[0][i].y;
		}
		else{
			vertex.uv.x = vertex.uv.y = 0;
		}

		vertices.push_back(vertex);
	}

	mat4 invT = transpose(inverse(trs));
	for (int i = 0; i < vertices.size(); ++i){
		vec3 v = Float3ToVec(vertices[i].v);
		vec3 n = Float3ToVec(vertices[i].n);

		v = vec3(trs*vec4(v, 1));

		n = normalize(vec3(invT*vec4(n, 0)));
		vertices[i].v = VecToFloat3(v);
		vertices[i].n = VecToFloat3(n);
		vertices[i].t = make_float3(0, 0, 0);
	}

	for (int i = 0; i < aimesh->mNumFaces; ++i){
		aiFace face = aimesh->mFaces[i];
		int idx1 = face.mIndices[0];
		int idx2 = face.mIndices[1];
		int idx3 = face.mIndices[2];
		float3 tangent = genTangent(idx1, idx2, idx3);
		vertices[idx1].t += tangent;
		vertices[idx2].t += tangent;
		vertices[idx3].t += tangent;
	}

	for (int i = 0; i < vertices.size(); ++i){
		vertices[i].t = normalize(vertices[i].t);
	}

	for (int i = 0; i < aimesh->mNumFaces; ++i){
		aiFace face = aimesh->mFaces[i];
		Triangle tri;
		int idx1 = face.mIndices[0];
		int idx2 = face.mIndices[1];
		int idx3 = face.mIndices[2];
		tri.v1 = vertices[idx1];
		tri.v2 = vertices[idx2];
		tri.v3 = vertices[idx3];
		tri.matIdx = matIdx;
		tri.bssrdfIdx = bssrdfIdx;
		tri.lightIdx = -1;
		triangles.push_back(tri);
	}
}

float3 Mesh::genTangent(int idx1, int idx2, int idx3){
	Vertex v1 = vertices[idx1];
	Vertex v2 = vertices[idx2];
	Vertex v3 = vertices[idx3];
	float2 duv21 = v2.uv - v1.uv, duv31 = v3.uv - v1.uv;
	float3 dp21 = v2.v - v1.v, dp31 = v3.v - v1.v;
	float det = duv21.x * duv31.y - duv21.y*duv31.x;
	bool degenerate = fabs(det) < 1e-8;
	if (!degenerate){
		float invdet = 1.f / det;
		float3 tangent = normalize((duv31.y*dp21 - duv21.y*dp31)*invdet);
		return tangent;
	}
	else{
		float3 uu, ww;
		float3 nn = normalize(cross(v2.v - v1.v, v3.v - v1.v));
		MakeCoordinate(nn, uu, ww);
		return uu;
	}
}