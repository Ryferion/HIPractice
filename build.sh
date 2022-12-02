ROCM_PATH=/opt/rocm
HIP_PATH=${ROCM_PATH}/hip
ROCM_TOOLKIT_PATH=${ROCM_PATH}
ROCPROFILER=${ROCM_PATH}/rocprofiler
DEVICE_LIB_PATH=${ROCM_PATH}/rocdl/amdgcn/bitcode
PATH=${ROCM_PATH}/llvm/bin:${ROCM_PATH}/bin:${HIP_PATH}/bin:$PATH
LD_LIBRARY_PATH=${ROCM_PATH}/comgr/lib:${HIP_PATH}/lib:${ROCM_PATH}/rocrand/lib:${ROCM_PATH}/hiprand/lib:${ROCM_PATH}/lib:$LD_LIBRARY_PATH
GFXLIST="gfx906"
