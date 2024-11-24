#include <mpi.h>
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
    int myRank, mySize;
    
    MPI_Status myStatus;

    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
    MPI_Comm_size(MPI_COMM_WORLD, &mySize);

    int numProcRows = (matrixSize+mySize-1)/mySize;
    if (myRank >= matrixSize % mySize && matrixSize % mySize != 0){
        numProcRows--;
    }

    std::vector<std::vector<long long int>> matrixB(matrixSize, std::vector<long long int>(matrixSize));
    std::vector<std::vector<long long int>> myMatrixA(numProcRows, std::vector<long long int>(matrixSize));
    std::vector<std::vector<long long int>> myMatrixC(numProcRows, std::vector<long long int>(matrixSize));

    if (myRank == 0){
        std::vector<std::vector<long long int>> matrixA(matrixSize, std::vector<long long int>(matrixSize));
        initializeMatrix(matrixA);
        initializeMatrix(matrixB);

        timerTotal.startTimer();
        
        if (matrixSize <= 10){
            std::cout << "Matrix A:" << std::endl;
            printMatrix(matrixA);
            std::cout << "Matrix B:" << std::endl;
            printMatrix(matrixB);
        }else{
            std::cout << "Sample values of Matrix A:" << std::endl;
            std::cout << "matrixA[0][0] = " << matrixA[0][0] << std::endl;
            std::cout << "matrixA[matrixSize-1][matrixSize-1] = " << matrixA[matrixSize-1][matrixSize-1] << std::endl << std::endl;
            std::cout << "Sample values of Matrix B:" << std::endl;
            std::cout << "matrixB[0][0] = " << matrixB[0][0] << std::endl;
            std::cout << "matrixB[matrixSize-1][matrixSize-1] = " << matrixB[matrixSize-1][matrixSize-1] << std::endl << std::endl;
        }

        for (int i = 0; i < mySize; i++){
            int yourRows = (matrixSize+mySize-1)/mySize;
            if (i >= matrixSize % mySize && matrixSize % mySize != 0){
                yourRows--;
            }

            for (int j = 0; j < yourRows; j++) {
                int masterRow = i+j*mySize;
                if (i == 0){
                    myMatrixA[j] = matrixA[masterRow];
                }else{
                    MPI_Send(matrixA[masterRow].data(), matrixSize, MPI_LONG_LONG, i, 0, MPI_COMM_WORLD);
                }
            }
        }
    }else{
        for (int i = 0; i < numProcRows; i++){
            MPI_Recv(myMatrixA[i].data(), matrixSize, MPI_LONG_LONG, 0, 0, MPI_COMM_WORLD, &myStatus);
        }
    }

    for (int i = 0; i < matrixSize; i++){
        MPI_Bcast(matrixB[i].data(), matrixSize, MPI_LONG_LONG, 0, MPI_COMM_WORLD);
    }

    for (int i = 0; i < numProcRows; i++){
        for (int j = 0; j < matrixSize; j++){
            myMatrixC[i][j] = 0;
            for (int k = 0; k < matrixSize; k++){
                myMatrixC[i][j] += myMatrixA[i][k]*matrixB[k][j];
            }
        }
    }

    if (myRank == 0){
        std::vector<std::vector<long long int>> matrixC(matrixSize, std::vector<long long int>(matrixSize));

        for (int i = 0; i < numProcRows; i++){
            int masterRow = i*mySize;
            matrixC[masterRow] = myMatrixC[i];
        }

        for (int i = 1; i < mySize; i++){
            int yourRows = (matrixSize+mySize-1)/mySize;
            if (i >= matrixSize % mySize && matrixSize % mySize != 0){
                yourRows--;
            }

            for (int j = 0; j < yourRows; j++){
                int masterRow = i+j*mySize;
                MPI_Recv(matrixC[masterRow].data(), matrixSize, MPI_LONG_LONG, i, 0, MPI_COMM_WORLD, &myStatus);
            }
        }

        timerTotal.stopTimer();
        std::cout << "Total time: " << timerTotal.getElapsedTime() << std::endl << std::endl;

        if (matrixSize <= 10){
            std::cout << "Product Matrix C:" << std::endl;
            printMatrix(matrixC);
        }else{
            std::cout << "Matrix multiplication for " << matrixSize << "x" << matrixSize << " Matrix:" << std::endl;
            std::cout << "Sample values of product Matrix C:" << std::endl;
            std::cout << "C[0][0] = " << matrixC[0][0] << std::endl;
            std::cout << "C[matrixSize-1][matrixSize-1] = " << matrixC[matrixSize-1][matrixSize-1] << std::endl << std::endl << std::endl;
        }
    }else{
        for (int i = 0; i < numProcRows; i++){
            MPI_Send(myMatrixC[i].data(), matrixSize, MPI_LONG_LONG, 0, 0, MPI_COMM_WORLD);
        }
    }
}

void runMatrixMultiplication(){
    for (int matrixSize = 1000; matrixSize <= 9000; matrixSize += 2000){
        for (int trial = 0; trial < 3; trial++){
            matrixMultiplication(matrixSize);
        }
    }
}

int main(){
    MPI_Init(NULL, NULL);

    runMatrixMultiplication();

    MPI_Finalize();

    return 0;
}