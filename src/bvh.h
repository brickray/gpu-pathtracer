#ifndef H_BVH_H
#define H_BVH_H

#include "cutil_math.h"
#include "primitive.h"

struct BVHNode{
	BVHNode* left;
	BVHNode* right;  
	BBox bbox;
	bool is_leaf;
	vector<Primitive> primitives;

	BVHNode(){
		left = right = NULL;
	}
};

struct LinearBVHNode{
	BBox bbox;
	int second_child_offset;
	bool is_leaf;
	int start;
	int end;

	__host__ __device__ LinearBVHNode(){
		start = end = -1;
	}
};

class BVH{
public:
	LinearBVHNode* linear_root;
	int total_nodes;
	vector<Primitive> prims;
	BBox root_box;

public:
	BVH();

	void Build(vector<Primitive>& primitives);

private:
	BVHNode* split(vector<Primitive>& primitives, BBox& bbox);
	void flatten(BVHNode* node, int cur, int& next);
	void clearBVHNode(BVHNode* node);
};

#endif