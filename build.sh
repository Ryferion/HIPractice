#!/bin/sh
rm -fr build
mkdir build
cd build
cmake -D__HIP_PLATFORM_AMD__ ..
make VERBOSE=1 -j 1