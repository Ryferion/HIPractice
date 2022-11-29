#include <iostream>
#include <fstream>

#include "hip/hip_runtime.h"

#define __HIP_PLATFORM_HCC__
#define DEVICE_NUM 2
#define TILE_SIZE 16

using namespace std;

// can get a corresponding error string by
#define HIP_CHECK(command) {        \
    hipError_t status = command;    \
    if (status != hipSuccess) {     \
        std::cerr << "Error: HIP reports " << hipGetErrorString(status) << std::endl;   \
        std::abort(); } }

__global__ void matrixMultiply(int row, int col, int out, const float *A, const float *B, float *C)
{
    /*
    A = row x col
    B = col x out
    C = row x out
    */
    
    // blocking to be more cache coherent
    __shared__ float sharedM1[TILE_SIZE][TILE_SIZE];
    __shared__ float sharedM2[TILE_SIZE][TILE_SIZE];
    
    int xThread = threadIdx.x;
    int yThread = threadIdx.y;

    int xIdx = xThread + blockIdx.x & blockDim.x;
    int yIdx = yThread + blockIdx.y & blockDim.y;
    
    float temp;
    
    int i = 0;
    for (i = 0; i < (TILE_SIZE + out - 1) / TILE_SIZE; i++)
    {
        int xPos = i * TILE_SIZE + xThread;
        if ((xIdx < col) && (xPos < out))
        {
            sharedM1[yThread][xThread] = A[xIdx * out + xPos];
        }
        else
        {
            sharedM1[yThread][xThread] = 0.0;
        }

        int yPos = i * TILE_SIZE + yThread;
        if ((yIdx < col) && (yPos < out))
        {
            sharedM2[yThread][xThread] = B[yIdx * out + yPos];
        }
        else
        {
            sharedM2[yThread][xThread] = 0.0;
        }

        __syncthreads();

        // combine blocks
        for (int j = 0; j < TILE_SIZE; j++)
        {
            temp += sharedM1[yThread][i] * sharedM2[i][xThread];
        }

        __syncthreads();

        if ((yIdx < row) && (xIdx < col))
        {
            C[yIdx * col + xIdx] = temp;
        }
    }
}

void matrixWrite(int rowSize, int colSize, float *input, string fileName)
{
    fstream outFile;
    outFile.open(fileName, std::fstream::out | std::fstream::trunc);

    for (int i = 0; i < rowSize; i++)
    {   
        for (int j = 0; j < colSize; j++)
        {
            outFile << input[i + i *j];
            if ((j + 1) != colSize)
            {
                outFile << " ";
            }
        }
        outFile << "\n";
    }
    outFile.close();
}


void matrixRead(string fileName, float *readTo, int size)
{
    // =================================== read in matrix ===================================
    int counter = 0;
    ifstream outFile (fileName);
    if (outFile.is_open())
    {
        string temp;
        while (outFile >> temp)
        {
            readTo[counter] = stof(temp);
            cout << readTo[counter] << " ";
            counter++;
        }
    }
}

int main(int argc, char **argv)
{
    cout << "C++ version: ";
    if (__cplusplus == 201703L) std::cout << "C++17\n";
    else if (__cplusplus == 201402L) std::cout << "C++14\n";
    else if (__cplusplus == 201103L) std::cout << "C++11\n";
    else if (__cplusplus == 199711L) std::cout << "C++98\n";
    else std::cout << "pre-standard C++\n";


    int deviceCount = -1, deviceID = -1;

    HIP_CHECK(hipSetDevice(DEVICE_NUM)); // use GPU 2
    HIP_CHECK(hipGetDevice(&deviceID)); 
    HIP_CHECK(hipGetDeviceCount(&deviceCount)); // how many devices there be (should be 8 on idk)
    
    cout << " Current Device: " << deviceID << endl;
    if (deviceID != 2)
    {
        return 0;
    }

    float *A_host, *B_host, *C_host;
    float *A_device, *B_device, *C_device;
    size_t A_size, B_size, C_size;

    /*
    A = row x col
    B = col x out
    C = row x out
    */

    int row, col, out;
    string matrixOne, matrixTwo, matrixThree;
        row = 8;
        col = 8;
        out = 8;
        matrixOne = "matrix1.txt";
        matrixTwo = "matrix2.txt";
        matrixThree = "matrix3.txt";

    if (atoi(argv[1]) != 1)
    {
        if (argv[1] != NULL) { matrixOne = argv[1]; } 
        if (argv[2] != NULL) { row = atoi(argv[2]); } 

        if (argv[3] != NULL) { matrixTwo = argv[3]; } 
        if (argv[4] != NULL) { col = atoi(argv[4]); } 

        if (argv[5] != NULL) { matrixThree = argv[5]; } 
        if (argv[6] != NULL) { out = atoi(argv[6]); } 
    }

    A_size = row * col;
    B_size = col * out;
    C_size = row * out;

    A_host = (float*) malloc( sizeof(float)*A_size);
    B_host = (float*) malloc( sizeof(float)*B_size);
    C_host = (float*) malloc( sizeof(float)*C_size);
    
    matrixRead(matrixOne, A_host, A_size);
    matrixRead(matrixTwo, B_host, B_size);

    // allocate memory for device
    HIP_CHECK(hipMalloc((void**) &A_device, sizeof(float) * A_size));
    HIP_CHECK(hipMalloc((void**) &B_device, sizeof(float) * B_size));
    HIP_CHECK(hipMalloc((void**) &C_device, sizeof(float) * C_size));
    
    // copy data from host to device
    HIP_CHECK(hipMemcpy(A_device, A_host, sizeof(float) * A_size, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(B_device, B_host, sizeof(float) * B_size, hipMemcpyHostToDevice));

    // set up block dim and thread dim
    dim3 blocks(col / TILE_SIZE + 1, row / TILE_SIZE + 1, 1); // 3D dimensions of the grid of blocks
    dim3 threads(TILE_SIZE, TILE_SIZE, 1); // 3D dimensions of a block of threads

    // launch kernel
    hipLaunchKernelGGL(matrixMultiply, blocks, threads, 0, 0, row, col, out, A_device, B_device, C_device);
    HIP_CHECK(hipGetLastError());

    // copy matrix data from device to host
    HIP_CHECK(hipMemcpy(C_host, C_device, sizeof(float) * C_size, hipMemcpyDeviceToHost)); // host waits for kernel to finish here since hipMemcpy is blocking
    

    // write to .tt
    // matrixWrite(row, out, C_host, matrixThree);

    // hipStream_t stream;
    // HIP_CHECK(hipStreamCreate(stream));

    // const uint32_t CUMask = 0xffffffff;
    // const uint32_t CUMask_size = 1;
    // HIP_CHECK(hipExtStreamCreateWithCUMask(stream, CUMask_size, CUMask))

    free(A_host); // free host memory
    HIP_CHECK(hipFree(A_device)); // free device memory
    free(B_host); // free host memory
    HIP_CHECK(hipFree(B_device)); // free device memory
    free(C_host); // free host memory
    HIP_CHECK(hipFree(C_device)); // free device memory

    return 0;
}
