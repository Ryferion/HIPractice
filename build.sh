#!/bin/sh
rm -fr build
mkdir build
cd build
#cmake .. -DCMAKE_C_COMPILER=/home/rquac004/.opt/rocm/llvm/bin/clang -DCMAKE_CXX_COMPILER=/home/rquac004/.opt/rocm/llvm/bin/clang++
cmake .. -DCMAKE_CXX_COMPILER=/home/rquac004/.opt/rocm/bin/hipcc
make VERBOSE=1 -j 1
