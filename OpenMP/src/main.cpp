#include <omp.h>
#include <random>
#include <iomanip>
#include <iostream>
#include <vector>

#include "CStopWatch.h"

void initializeMatrix(std::vector<std::vector<long long int>> &initMatrix){
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, 100000);

    for (int i = 0; i < initMatrix.size(); i++){
        for (int j = 0; j < initMatrix[0].size(); j++){
            initMatrix[i][j] = dis(gen);
        }
    }
}

void printMatrix(const std::vector<std::vector<long long int>> &printMatrix){
    for (int i = 0; i < printMatrix.size(); i++){
        for (int j = 0; j < printMatrix[0].size(); j++){
            std::cout << std::fixed << std::setprecision(2) << std::setw(8) << printMatrix[i][j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

void matrixMultiplication(int matrixSize){
    CStopWatch timerTotal;

    std::vector<std::vector<long long int>> matrixA(matrixSize, std::vector<long long int>(matrixSize));
    std::vector<std::vector<long long int>> matrixB(matrixSize, std::vector<long long int>(matrixSize));

    initializeMatrix(matrixA);
    initializeMatrix(matrixB);

    timerTotal.startTimer();

    std::vector<std::vector<long long int>> matrixC(matrixSize, std::vector<long long int>(matrixSize));

    const int blockSize = 32;
    #pragma omp parallel shared(matrixA, matrixB, matrixC)
    {
        #pragma omp for collapse(2) schedule(dynamic, 4)
        for (int i = 0; i < matrixSize; i+=blockSize){
            for (int j = 0; j < matrixSize; j+=blockSize){
                for (int k = 0; k < matrixSize; k+=blockSize){
                    for (int l = i; l < std::min(i+blockSize, matrixSize); l++){
                        for (int m = j; m < std::min(j+blockSize, matrixSize); m++){
                            long long int sum = matrixC[l][m];
                            for (int n = k; n < std::min(k+blockSize, matrixSize); n++){
                                sum += matrixA[l][n] * matrixB[n][m];
                            }
                            matrixC[l][m] = sum;
                        }
                    }
                }
            }
        }
    }

    timerTotal.stopTimer();
    std::cout << "Total time: " << timerTotal.getElapsedTime() << std::endl;
}

void runMatrixMultiplication(){
    for (int matrixSize = 1000; matrixSize <= 9000; matrixSize += 2000){
        std::cout << "Matrix multiplication for " << matrixSize << "x" << matrixSize << " Matrix:" << std::endl;
        for (int trial = 0; trial < 3; trial++){
            matrixMultiplication(matrixSize);
        }
        std::cout << std::endl << std::endl;
    }
}

int main(){
    int numThreads;
    int threadMin = 1, threadMax = 28, threadStep = 3;

    for (numThreads=threadMin; numThreads<=threadMax; numThreads+=threadStep){
        omp_set_num_threads(numThreads);
        std::cout << "Number of threads: " << numThreads << std::endl;
        runMatrixMultiplication();
        std::cout << std::endl;
    }

    return 0;
}