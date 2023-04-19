#include <iostream>
#include <fstream>

#include <bitset>
#include <chrono>
#include <pthread.h>

#include "hip/hip_runtime.h"
#include "rocm_smi/rocm_smi.h"

#define __HIP_PLATFORM_HCC__
#define DEVICE_NUM 0
#define TILE_SIZE 8

using namespace std;
using namespace std::chrono;

// can get a corresponding error string by
#define HIP_CHECK(command) {        \
    hipError_t status = command;    \
    if (status != hipSuccess) {     \
        std::cerr << "Error: HIP reports " << hipGetErrorString(status) << std::endl;   \
        std::abort(); } }


#define PRINT_RSMI_ERR(RET) { \
  if (RET != RSMI_STATUS_SUCCESS) { \
    const char *err_str; \
    std::cout << "RSMI call returned " << (RET) \
      << " at line " << __LINE__ << std::endl; \
      rsmi_status_string((RET), &err_str); \
      std::cout << err_str << std::endl; \
  } \
}

#define CHK_RSMI_RET(RET) { \
  PRINT_RSMI_ERR(RET) \
  if (RET != RSMI_STATUS_SUCCESS) { \
    return (RET); \
  } \
}

#define CHK_RSMI_RET_I(RET) { \
  PRINT_RSMI_ERR(RET) \
  if (RET != RSMI_STATUS_SUCCESS) { \
    return static_cast<int>(RET); \
  } \
}

#define CHK_RSMI_PERM_RET(RET) { \
    if ((RET) == RSMI_STATUS_PERMISSION) { \
      std::cout << "This command requires root access." << std::endl; \
    } else { \
      CHK_RSMI_RET_I(RET) \
    } \
}

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

    int xIdx = xThread + blockIdx.x * blockDim.x; // current col
    int yIdx = yThread + blockIdx.y * blockDim.y; // current row
    
    float temp = 0;
    
    int i = 0;
    for (i = 0; i < (TILE_SIZE + out - 1) / TILE_SIZE; i++)
    {
        int xPos = i * TILE_SIZE + xThread;
        if ((yIdx < row) && (xPos < out))
        {
            sharedM1[yThread][xThread] = A[yIdx * out + xPos];
        }
        else
        {
            sharedM1[yThread][xThread] = 0.0;
        }

        int yPos = i * TILE_SIZE + yThread;
        if ((xIdx < col) && (yPos < out))
        {
            sharedM2[yThread][xThread] = B[xIdx * col + yPos];
        }
        else
        {
            sharedM2[yThread][xThread] = 0.0;
        }

        __syncthreads();

        // combine blocks
        for (int j = 0; j < TILE_SIZE; j++)
        {
            temp += sharedM1[yThread][j] * sharedM2[j][xThread];
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
            outFile << input[j + j * i];
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
            counter++;
        }
    }
}

struct powerArgs {
    int arg_mask1;
    int arg_mask2;
    int arg_status;
};

void* powerCheck(void *args)
{
    struct powerArgs *inputArgs = (struct powerArgs*) args;
    int mask1 = inputArgs->arg_mask1;
    int mask2 = inputArgs->arg_mask2;
    int status = inputArgs->arg_status;

    rsmi_status_t ret;
    std::string val_str;
    std::vector<std::string> val_vec;
    uint64_t val_ui64, val2_ui64;
    int64_t val_i64;
    uint32_t val_ui32;
    uint16_t val_ui16;
    rsmi_dev_perf_level_t pfl;
    rsmi_frequencies_t f;
    uint32_t num_monitor_devs = 0;
    rsmi_gpu_metrics_t p;

    ret = rsmi_init(0);
    // CHK_RSMI_RET_I(ret)
    ret = rsmi_dev_gpu_metrics_info_get(DEVICE_NUM, &p);
    // CHK_RSMI_RET(ret)
    

    if (status == 1) 
    {
        std::cout << "\t**GPU METRICS BEFORE KERNEL" << std::endl;    
    }
    if (status == 2)
    {
        std::cout << "\t**GPU METRICS AFTER KERNEL" << std::endl;
    }

    ret = rsmi_dev_temp_metric_get(DEVICE_NUM, 0, RSMI_TEMP_CURRENT, &val_i64);
    // CHK_RSMI_RET_I(ret)
    std::cout << "\t**Temperature: " << val_i64/1000 << "C" << std::endl;

    // ret = rsmi_dev_volt_metric_get(DEVICE_NUM, RSMI_VOLT_TYPE_VDDGFX,
    //                                            RSMI_VOLT_CURRENT, &val_i64);
    // // CHK_RSMI_RET_I(ret)
    // std::cout << "\t**Voltage: " << val_i64 << "mV" << std::endl;

    // ret = rsmi_dev_fan_speed_get(DEVICE_NUM, 0, &val_i64);
    // // CHK_RSMI_RET_I(ret)
    // ret = rsmi_dev_fan_speed_max_get(DEVICE_NUM, 0, &val_ui64);
    // // CHK_RSMI_RET_I(ret)
    // std::cout << "\t**Current Fan Speed: ";
    // std::cout << val_i64/static_cast<int64_t>(val_ui64)*100;
    // std::cout << "% ("<< val_i64 << "/" << val_ui64 << ")" << std::endl;

    // ret = rsmi_dev_fan_rpms_get(DEVICE_NUM, 0, &val_i64);
    // // CHK_RSMI_RET_I(ret)
    // std::cout << "\t**Current fan RPMs: " << val_i64 << std::endl;

    // ret = rsmi_dev_power_cap_get(DEVICE_NUM, 0, &val_ui64);
    // // CHK_RSMI_PERM_RET(ret)
    // std::cout << "\t**Current Power Cap: " << val_ui64 << "uW" <<std::endl;

    ret = rsmi_dev_power_ave_get(DEVICE_NUM, 0, &val_ui64);
    // CHK_RSMI_PERM_RET(ret)
    std::cout << "\t**Averge Power Usage: ";
    std::cout << static_cast<float>(val_ui64)/1000000 << " W" << std::endl;
    std::cout << "\t=======" << std::endl;

    if (status == 2)
    {
        std::cout << "\n\n";
    }
    
    return NULL;
}

struct hipArgs {
    int arg_mask1;
    int arg_mask2;
    int arg_row;
    int arg_col;
    int arg_out;
    string arg_firstMatrix;
    string arg_secondMatrix;
    string arg_thirdMatrix;
};

void* hip(void *args)
{
    struct hipArgs *inputArgs = (struct hipArgs*) args;
    int mask1 = inputArgs->arg_mask1;
    int mask2 = inputArgs->arg_mask2;
    int row = inputArgs->arg_row;
    int col = inputArgs->arg_row;
    int out = inputArgs->arg_out;
    string matrixOne = inputArgs->arg_firstMatrix;
    string matrixTwo = inputArgs->arg_secondMatrix;
    string matrixThree = inputArgs->arg_thirdMatrix;
    
    int iter = 0;

    // memory related variables
    uint32_t CUMask[2];
    const uint32_t CUMask_size = 2;

    uint32_t CUMaskMax[2];
    CUMaskMax[0] = 0xffffffff;
    CUMaskMax[1] = 0xffffffff;
    // just in case i stupidly put mask as 0
    {
        CUMask[0] = 0x00000001;
        CUMask[1] = 0x00000000; 
    }
    if (mask1 <= 0xffffffff)
    {   
        CUMask[0] = mask1;
    }
    if (mask2 <= 0xffffffff)
    {
        CUMask[1] = mask2; 
    }

    float *A_host, *B_host, *C_host;
    float *A_device, *B_device, *C_device;
    size_t A_size, B_size, C_size;

    // print mask to check
    // cout << " CUMask: " << std::bitset<32>(CUMask) << endl;
    
    // create streams
    hipStream_t streamMultiply;
    hipStream_t streamMemory;

    A_size = row * col;
    B_size = col * out;
    C_size = row * out;
    
    // allocate host memory
    HIP_CHECK(hipHostMalloc((void**) &A_host, sizeof(float) * A_size));
    HIP_CHECK(hipHostMalloc((void**) &B_host, sizeof(float) * B_size));
    HIP_CHECK(hipHostMalloc((void**) &C_host, sizeof(float) * C_size));
    
    // fill host matrices with stuff from text files
    matrixRead(matrixOne, A_host, A_size);
    matrixRead(matrixTwo, B_host, B_size);

    // matrix multiplication

    // allocate memory for device
    HIP_CHECK(hipMalloc((void**) &A_device, sizeof(float) * A_size));
    HIP_CHECK(hipMalloc((void**) &B_device, sizeof(float) * B_size));
    HIP_CHECK(hipMalloc((void**) &C_device, sizeof(float) * C_size));

    // copy data from host to device using stream...
    HIP_CHECK(hipExtStreamCreateWithCUMask(&streamMultiply, CUMask_size, CUMask)); 
    HIP_CHECK(hipExtStreamCreateWithCUMask(&streamMemory, CUMask_size, CUMask)); 

    // start timer: gear it towards kernel stuff
    auto start = high_resolution_clock::now();

    HIP_CHECK(hipMemcpyAsync(A_device, A_host, sizeof(float) * A_size, hipMemcpyHostToDevice, streamMemory));
    HIP_CHECK(hipMemcpyAsync(B_device, B_host, sizeof(float) * B_size, hipMemcpyHostToDevice, streamMemory));

    // set up block dim and thread dim
    dim3 blocks(col / TILE_SIZE + 1, row / TILE_SIZE + 1, 1); // 3D dimensions of the grid of blocks
    dim3 threads(TILE_SIZE, TILE_SIZE, 1); // 3D dimensions of a block of threads

    // // power thread launch right before kernel launch
    // pthread_t pthread_id2;
    // struct powerArgs *powerThreadBefore = (struct powerArgs *) malloc(sizeof(struct powerArgs));
    // powerThreadBefore->arg_mask1 = mask1;
    // powerThreadBefore->arg_mask2 = mask2;
    // powerThreadBefore->arg_status = 1;
    // pthread_create(&pthread_id2, NULL, powerCheck, (void *)powerThreadBefore);
    

    // launch kernel
    hipLaunchKernelGGL(matrixMultiply, blocks, threads, 0, streamMultiply, row, col, out, A_device, B_device, C_device);    

    // HIP_CHECK(hipGetLastError());

    HIP_CHECK(hipStreamSynchronize(streamMultiply));
    HIP_CHECK(hipStreamSynchronize(streamMemory));

    // copy matrix data from device to host
    HIP_CHECK(hipMemcpyAsync(C_host, C_device, sizeof(float) * C_size, hipMemcpyDeviceToHost, streamMemory)); // host waits for kernel to finish here since hipMemcpy is blocking

    // // end timer
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    cout << "Time taken by function: " << duration.count() << " microseconds." << endl;
    cout << "Current CU mask " << std::bitset<32>(CUMask[1]) << std::bitset<32>(CUMask[0]) <<  endl;

    pthread_t pthread_id3;
    struct powerArgs *powerThreadAfter = (struct powerArgs *) malloc(sizeof(struct powerArgs));
    powerThreadAfter->arg_mask1 = mask1;
    powerThreadAfter->arg_mask2 = mask2;
    powerThreadAfter->arg_status = 2;
    pthread_create(&pthread_id3, NULL, powerCheck, (void *)powerThreadAfter);
   

    // pthread_join(pthread_id2, NULL);
    // free(powerThreadBefore);

    pthread_join(pthread_id3, NULL);
    free(powerThreadAfter);

    HIP_CHECK(hipStreamDestroy(streamMultiply));
    HIP_CHECK(hipStreamDestroy(streamMemory));

    HIP_CHECK(hipFree(A_device)); // free device memory
    HIP_CHECK(hipFree(B_device)); // free device memory
    HIP_CHECK(hipFree(C_device)); // free device memory
    
    // write to .txt
    // matrixWrite(row, out, C_host, matrixThree);

    HIP_CHECK(hipHostFree(A_host)); // free pinned memory
    HIP_CHECK(hipHostFree(B_host)); // free pinned memory
    HIP_CHECK(hipHostFree(C_host)); // free pinned memory





    return NULL;
}

int main(int argc, char **argv)
{
    int deviceCount = -1, deviceID = -1, CUCount = -1;

    HIP_CHECK(hipSetDevice(DEVICE_NUM));
    HIP_CHECK(hipGetDevice(&deviceID)); 
    HIP_CHECK(hipGetDeviceCount(&deviceCount)); // how many devices there be (should be 8 on idk)
    
    hipDeviceProp_t deviceProps;
    HIP_CHECK(hipGetDeviceProperties(&deviceProps, deviceID))

    cout << " Current Device: " << deviceID << endl;
    cout << " CU count: " << deviceProps.multiProcessorCount << endl;

    /*
    A = row x col
    B = col x out
    C = row x out
    */

    int mask = 1;
    int maxIter = 1;
    int row, col, out;
    string matrixOne, matrixTwo, matrixThree;
        row = 4096;
        col = 4096;
        out = 4096;
        matrixOne = "matrix1.txt";
        matrixTwo = "matrix2.txt";
        matrixThree = "matrix3.txt";

    if (argv[1] != NULL)
    {
        row = atoi(argv[1]);
        col = row;
        out = col;
    }

    if (argv[2] != NULL)
    {
        maxIter = atoi(argv[2]);
    }

    int firstMask, secondMask;

    firstMask = 1;
    secondMask = 0;

    for (int iter = 0; iter < maxIter; iter++)
    {
        // process thread
        pthread_t pthread_id;
        struct hipArgs *mainThread = (struct hipArgs *) malloc(sizeof(struct hipArgs));
        mainThread->arg_mask1 = firstMask;
        mainThread->arg_mask2 = secondMask;
        mainThread->arg_row = row;
        mainThread->arg_col = col;
        mainThread->arg_out = out;
        mainThread->arg_firstMatrix = matrixOne;
        mainThread->arg_secondMatrix = matrixTwo;
        mainThread->arg_thirdMatrix = matrixThree;

        // // power thread
        // pthread_t pthread_id2;
        // struct powerArgs *powerThread = (struct powerArgs *) malloc(sizeof(struct powerArgs));
        // powerThread->arg_mask1 = firstMask;
        // powerThread->arg_mask2 = secondMask;

        pthread_create(&pthread_id, NULL, &hip, (void *)mainThread);
        // pthread_create(&pthread_id2, NULL, powerCheck, (void *)powerThread);

        // wait for both threads to finish
        pthread_join(pthread_id, NULL);
        // pthread_join(pthread_id2, NULL);

        // pthread_exit(NULL);
        free(mainThread);
        // free(powerThread);

        if (iter < 32)
        {
            firstMask = firstMask*2 + 1;
        }
        else if ((iter < 64) && (iter >= 32))
        {
            secondMask = secondMask*2 + 1;
        }
        else
        {
            std::cout << "Maximum mask reached\n";
        }
        // std::cout << iter << " " << std::bitset<32>(firstMask) << std::bitset<32>(secondMask) <<  endl;
    }
    
    return 0;
}
