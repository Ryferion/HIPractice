#!/bin/sh
rm -fr build
mkdir build
cd build
cmake ..
make VERBOSE=1 -j 1