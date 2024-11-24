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
        std::cout << "Total time: " << timerTotal.getElapsedTime() << std::endl;
    }else{
        for (int i = 0; i < numProcRows; i++){
            MPI_Send(myMatrixC[i].data(), matrixSize, MPI_LONG_LONG, 0, 0, MPI_COMM_WORLD);
        }
    }
}

void runMatrixMultiplication(){
    int myRank;

    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);

    for (int matrixSize = 1000; matrixSize <= 9000; matrixSize += 2000){
        if (myRank == 0){
            std::cout << "Matrix multiplication for " << matrixSize << "x" << matrixSize << " Matrix:" << std::endl;
        }
        for (int trial = 0; trial < 3; trial++){
            matrixMultiplication(matrixSize);
        }
        if (myRank == 0){
            std::cout << std::endl << std::endl;
        }
    }
    if (myRank == 0){
        std::cout << std::endl;
    }
}

int main(){
    MPI_Init(NULL, NULL);

    runMatrixMultiplication();

    MPI_Finalize();

    return 0;
}