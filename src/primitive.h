#ifndef H_PRIMITIVE_H
#define H_PRIMITIVE_H

#include "common.h"
#include "mesh.h"
#include "line.h"
#include "sphere.h"

enum GeometryType{
	GT_TRIANGLE = 0,
	GT_LINES,
	GT_SPHERE,
};

class Primitive{
public:
	GeometryType type;
	union{
		Triangle triangle;
		Line line;
		Sphere sphere;
	};
};

#endif