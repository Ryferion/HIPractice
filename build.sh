#!/bin/sh
rm -fr build
mkdir build
cd build
cmake .. -DCMAKE_PREFIX_PATH="/home/rquac004/.opt/rocm"
make -n VERBOSE=1 -j 1