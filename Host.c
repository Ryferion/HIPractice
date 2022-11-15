#include "hip/hip_runtime.h"

#ifdef __HIP_PLATFORM_HCC__

hipSetDevice(2); // use GPU 2

int main() {

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
    
    free(h_a); // free host memory
    HIP_CHECK(hipFree(d_a)); // free device memory

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
    hipLaunchKernelGGL(reverse, dim3(1), dim3(256), 0, 0, d_a); // launch kernel
}