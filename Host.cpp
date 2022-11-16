#include <iostream>
#include "hip/hip_runtime.h"

#define __HIP_PLATFORM_HCC__
#define DEVICE_NUM 2

using namespace std;

// can get a corresponding error string by
#define HIP_CHECK(command) {        \
    hipError_t status = command;    \
    if (status != hipSuccess) {     \
        std::cerr << "Error: HIP reports " << hipGetErrorString(status) << std::endl;   \
        std::abort(); } }

__global__ void myKernel(int N, double *d_a)
{
    int i = threadIdx.x + blockIdx.x & blockDim.x;
    if (i < N)
    {
        d_a[i] *= 2.0;
    }
}


__global__ void reverse(double *d_a)
{
    __shared__ double s_a[256]; // array of doubles, shared in this block

    int tid = threadIdx.x;
    s_a[tid] = d_a[tid]; // each thread fills one entry
    
    // all wavefronts much reach this point before any wavefront is allowed to continue

    __syncthreads();
    d_a[tid] = s_a[255-tid]; // write out array in reverse order;
}




int main()
{
    int deviceCount = -1, deviceID = -1;

    HIP_CHECK(hipSetDevice(DEVICE_NUM)); // use GPU 2
    HIP_CHECK(hipGetDevice(&deviceID)); // get current device
    HIP_CHECK(hipGetDeviceCount(&deviceCount)); // how many devices there be (should be 8 on idk)
    // cout << " line: " << __LINE__ << " num devices: " << deviceCount << " current device ID: " << deviceID << endl;

    int N = 1000;
    size_t Nbytes = N*sizeof(double);
    double *h_a = (double*) malloc(Nbytes); // host memory

    double *d_a = NULL;
    HIP_CHECK(hipMalloc(&d_a, Nbytes)); // allocate Nbytes on device

    // copy data from host to device
    HIP_CHECK(hipMemcpy(d_a, h_a, Nbytes, hipMemcpyHostToDevice));

    dim3 blocks((N + 256 - 1)/256, 1, 1); // 3D dimensions of the grid of blocks
    dim3 threads(256, 1, 1); // 3D dimensions of a block of threads
    
    hipLaunchKernelGGL(myKernel, blocks, threads, 0, 0, N, d_a);

    HIP_CHECK(hipGetLastError());

    // copy data from device to host
    HIP_CHECK(hipMemcpy(h_a, d_a, Nbytes, hipMemcpyDeviceToHost)); // host waits for kernel to finish here since hipMemcpy is blocking
    
    hipStream_t stream;
    HIP_CHECK(hipStreamCreate(stream));

    const uint32_t CUMask = 0xffffffff;
    const uint32_t CUMask_size = 1;
    HIP_CHECK(hipExtStreamCreateWithCUMask(stream, CUMask_size, CUMask))

    free(h_a); // free host memory
    HIP_CHECK(hipFree(d_a)); // free device memory

}
