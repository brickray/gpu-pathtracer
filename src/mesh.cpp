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
	vector<Vertex> vertices;
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
	}

	for (int i = 0; i < aimesh->mNumFaces; ++i){
		aiFace face = aimesh->mFaces[i];
		Triangle tri;
		tri.v1 = vertices[face.mIndices[0]];
		tri.v2 = vertices[face.mIndices[1]];
		tri.v3 = vertices[face.mIndices[2]];
		tri.matIdx = matIdx;
		triangles.push_back(tri);
	}
}