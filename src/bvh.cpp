#include "bvh.h"
#include <xmmintrin.h>

BVH::BVH(){
	total_nodes = 0;
}

void BVH::Build(vector<Triangle>& triangles){
	if (triangles.size() == 0)
		return;
	BBox root_box;
	root_box.Reset();
	for (int i = 0; i < triangles.size(); ++i){
		root_box.Expand(triangles[i].GetBBox());
	}

	tris.reserve(triangles.size());

	BVHNode* root = split(triangles, root_box);
	root->bbox = root_box;
	triangles.clear();

	//flatten node
	linear_root = new LinearBVHNode[total_nodes];
	int next = 0;
	flatten(root, 0, next);
	clearBVHNode(root);
}

BVHNode* BVH::split(vector<Triangle>& triangles, BBox& bbox){
	++total_nodes;

	//detect if the bbox degenerate
	float3 diagonal = bbox.Diagonal();
	if (triangles.size() <= 4 || diagonal.x < 0.0001f || diagonal.y < 0.0001f || diagonal.z < 0.0001f){
		BVHNode* leaf = new BVHNode();
		leaf->bbox = bbox;
		leaf->is_leaf = true;
		for (int i = 0; i < triangles.size(); ++i)
			leaf->triangles.push_back(triangles[i]);

		return leaf;
	}

	int best_axis = -1;//0=x,1=y,2=z
	int best_bucket;
	float best_cost = triangles.size()*bbox.SurfaceArea();
	BBox best_left, best_right;
	struct Bucket{
		BBox bbox;
		int count;

		Bucket(){
			count = 0;
		}
	};
	const int bucket_num = 12;
	//find best split axis 
	for (int i = 0; i < 3; ++i){
		Bucket bucket[bucket_num];
		//place all triangles into correspond bucket
		for (int j = 0; j < triangles.size(); ++j){
			float3 center = triangles[j].GetBBox().Centric();
			float value = (i == 0) ? center.x : (i == 1) ? center.y : center.z;
			float value_start = (i == 0) ? bbox.fmin.x : (i == 1) ? bbox.fmin.y : bbox.fmin.z;
			float value_end = (i == 0) ? bbox.fmax.x : (i == 1) ? bbox.fmax.y : bbox.fmax.z;
			int no = (int)((value - value_start) / (value_end - value_start) * bucket_num);
			no = (no == 12) ? no - 1 : no;
			bucket[no].count++;
			bucket[no].bbox.Expand(triangles[j].GetBBox());
		}

		for (int j = 1; j < bucket_num; ++j){
			BBox b0, b1;
			int count0 = 0, count1 = 0;
			for (int k = 0; k < j; ++k){
				b0.Expand(bucket[k].bbox);
				count0 += bucket[k].count;
			}

			for (int k = j; k < bucket_num; ++k){
				b1.Expand(bucket[k].bbox);
				count1 += bucket[k].count;
			}

			//b0 will invalid when count0 equals to zero
			float surface_a = (count0 == 0) ? 0 : b0.SurfaceArea()*count0;
			float surface_b = (count1 == 0) ? 0 : b1.SurfaceArea()*count1;

			float cost = surface_a + surface_b;
			if (cost < best_cost){
				best_cost = cost;
				best_axis = i;
				best_bucket = j;
			}
		}
	}
	
	//can't find axis to split,hence just create a leaf node
	if (best_axis == -1){
		BVHNode* leaf = new BVHNode();
		leaf->bbox = bbox;
		leaf->is_leaf = true;
		for (int i = 0; i < triangles.size(); ++i)
			leaf->triangles.push_back(triangles[i]);

		return leaf;
	}

	std::vector<Triangle> left;
	std::vector<Triangle> right;

	float value_start = (best_axis == 0) ? bbox.fmin.x : (best_axis == 1) ? bbox.fmin.y : bbox.fmin.z;
	float value_end = (best_axis == 0) ? bbox.fmax.x : (best_axis == 1) ? bbox.fmax.y : bbox.fmax.z;
	for (int i = 0; i < triangles.size(); ++i){
		Triangle tri = triangles[i];
		float3 center = tri.GetBBox().Centric();
		float value = (best_axis == 0) ? center.x : (best_axis == 1) ? center.y : center.z;
		int no = (int)((value - value_start) / (value_end - value_start)*bucket_num);
		no = (no == bucket_num) ? no - 1 : no;
		if (no < best_bucket){
			left.push_back(tri);
			best_left.Expand(tri.GetBBox());
		}
		else{
			right.push_back(tri);
			best_right.Expand(tri.GetBBox());
		}
	}

	//create a inner node
	BVHNode* inner = new BVHNode();
	inner->bbox = bbox;
	inner->is_leaf = false;
	inner->left = split(left, best_left);
	inner->right = split(right, best_right);

	return inner;
}

void BVH::flatten(BVHNode* node, int cur, int& next){
	linear_root[cur].bbox = node->bbox;
	linear_root[cur].is_leaf = node->is_leaf;
	if (node->triangles.size()){
		linear_root[cur].start = tris.size();
		for (int i = 0; i < node->triangles.size(); ++i)
			tris.push_back(node->triangles[i]);
		linear_root[cur].end = tris.size() - 1;
	}

	if (node->left)
		flatten(node->left, cur + 1, ++next);
	
	if (node->right){
		linear_root[cur].second_child_offset = next + 1;
		flatten(node->right, next, ++next);//参数传递从右到左，所以这里第二个参数不需要+1
	}
	else
		linear_root[cur].second_child_offset = -1;

}

void BVH::clearBVHNode(BVHNode* node){
	if (node->is_leaf){
		delete node;
		return;
	}

	if (node->left)
		clearBVHNode(node->left);
	if (node->right)
		clearBVHNode(node->right);

	delete node;
}