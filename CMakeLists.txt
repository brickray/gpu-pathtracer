cmake_minimum_required(VERSION 3.2)

PROJECT(brick_cuda_ray)

SET(CMAKE_BUILD_TYPE RELEASE)

INCLUDE_DIRECTORIES(include)
LINK_DIRECTORIES(lib)
LINK_LIBRARIES(assimp.lib freeglut.lib glew.lib)

ADD_SUBDIRECTORY(src)
#ADD_SUBDIRECTORY(tool)
