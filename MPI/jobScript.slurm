#!/bin/bash

#SBATCH --nodes=28
#SBATCH --ntasks-per-node=1
#SBATCH --time=1:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --account=PCS0288
#SBATCH --job-name=MM

module load boost

#module load intel
cd $SLURM_SUBMIT_DIR/Default
make clean && make OSC

echo "Number of Processes: 1" >> results.csv
mpiexec -n 1 MPI >> results.csv    

echo "Number of Processes: 4" >> results.csv
mpiexec -n 4 MPI >> results.csv    

echo "Number of Processes: 7" >> results.csv
mpiexec -n 7 MPI >> results.csv    

echo "Number of Processes: 10" >> results.csv
mpiexec -n 10 MPI >> results.csv    

echo "Number of Processes: 13" >> results.csv
mpiexec -n 13 MPI >> results.csv    

echo "Number of Processes: 16" >> results.csv
mpiexec -n 16 MPI >> results.csv    

echo "Number of Processes: 19" >> results.csv
mpiexec -n 19 MPI >> results.csv  

echo "Number of Processes: 22" >> results.csv
mpiexec -n 22 MPI >> results.csv 

echo "Number of Processes: 25" >> results.csv
mpiexec -n 25 MPI >> results.csv   

echo "Number of Processes: 28" >> results.csv
mpiexec -n 28 MPI >> results.csv