#include <iostream>
#include <fstream>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <vector>
#include <cmath>
#include <random>

#define DEVICE_NUM 2
#define TILE_SIZE 8

using namespace std;


void matrixWrite(int rowSize, int colSize, vector<vector<float>> input, string fileName)
{
    fstream outFile;
    outFile.open(fileName, std::fstream::out | std::fstream::trunc);

    for (int i = 0; i < rowSize; i++)
    {   
        for (int j = 0; j < colSize; j++)
        {
            // cout << input[i][j];
            outFile << std::to_string(input[i][j]);
            if ((j + 1) != colSize)
            {
                outFile << " ";
            }
        }
        outFile << "\n";
    }
    outFile.close();
}


vector<vector<float>> matrixFill(int rowSize, int colSize, int val)
{
    vector<vector<float>> toFill;
    // =================================== read in matrix ===================================
    string temp;
    for (int i = 0; i < rowSize; i++)
    {
        vector<float> tempVec;
        for (int j = 0; j < colSize; j++)
        {
            float randomVal = 0;
            
            if (val != 1337)
            {
                randomVal = val;
            }
            else
            {
                randomVal = std::rand() % 1000 / 10;
            }
            tempVec.push_back(randomVal);
        }
        toFill.push_back(tempVec);
        tempVec.clear();
    }

    return toFill;
}

void matrixRead(string fileName, float* readTo, int size)
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
        // if (argv[1] != NULL) { matrixOne = argv[1]; } 
        // if (argv[2] != NULL) { row = atoi(argv[2]); } 

        // if (argv[3] != NULL) { matrixTwo = argv[3]; } 
        // if (argv[4] != NULL) { col = atoi(argv[4]); } 
        
        // if (argv[5] != NULL) { matrixThree = argv[5]; } 
        // if (argv[6] != NULL) { out = atoi(argv[6]); } 
        row = atoi(argv[1]);
        col = row;
        out = col;
    }

    vector<vector<float>> A;
    vector<vector<float>> B;
    vector<vector<float>> C;

    A = matrixFill(row, col, 1337);
    B = matrixFill(col, out, 1337);
    C = matrixFill(row, out, 0);
    
    matrixWrite(row, col, A, matrixOne);
    matrixWrite(col, out, B, matrixTwo);
    matrixWrite(row, out, C, matrixThree);

    A_size = row * col;
    B_size = col * out;
    C_size = row * out;

    A_host = (float*) malloc( sizeof(float)*A_size);
    B_host = (float*) malloc( sizeof(float)*B_size);
    C_host = (float*) malloc( sizeof(float)*C_size);
    
    // matrixRead(matrixOne, A_host, A_size);
    free(A_host);
    free(B_host);
    free(C_host);
}
