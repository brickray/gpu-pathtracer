#ifndef H_CATMULLROM_H
#define H_CATMULLROM_H

#include "common.h"

//binary serach
int FindInterval(int size, float* node, float x){
	for (int i = 0; i < size - 1; ++i){
		if (x > node[i] && x < node[i + 1])
			return i;
	}
}

float CatmullRom(int size, float* node, float* value, float x){
	if (x<node[0] || x>node[size - 1]) return 0.f;
	int idx = FindInterval(size, node, x);
	float x0 = node[idx], x1 = node[idx + 1];
	float f0 = value[idx], f1 = value[idx + 1];
	float d0, d1;
	if (idx > 0){
		d0 = (f1 - value[idx - 1]) / (x1 - node[idx - 1]);
	}
	else{
		d0 = f1 - f0;
	}

	if (idx < size - 2){
		d1 = (value[idx + 1] - f0) / (node[idx + 1] - x0);
	}
	else{
		d1 = f1 - f0;
	}

	float t = (x - x0) / (x1 - x0), t2 = t*t, t3 = t2*t;
	return (2.f*t3 - 3.f*t2 + 1.f)*f0 + (-2.f*t3 + 3.f*t2)*f1 +
		(t3 - 2.f*t2 + t)*d0 + (t3 - t2)*d1;
}

bool CatmullRomWeights(int size, float* node, float x, int* offset, float* weights){
	if (x<node[0] || x>node[size - 1]) return false;
	int idx = FindInterval(size, node, x);

}

#endif