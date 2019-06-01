#ifndef H_PRIMITIVE_H
#define H_PRIMITIVE_H

#include "common.h"
#include "mesh.h"
#include "line.h"

enum GeometryType{
	GT_TRIANGLE = 0,
	GT_LINES,
};

class Primitive{
public:
	GeometryType type;
	union{
		Triangle triangle;
		Line line;
	};
};

#endif