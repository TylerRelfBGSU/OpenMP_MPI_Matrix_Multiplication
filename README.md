## Information
This code is for replicability purposes for the research paper "Comparative Study of OpenMP and MPI Using Matrix Multiplication: Performance and Scalability Analysis". The following instructions can be used in the Ohio Supercomputer Center for each of the directories [MPI](./MPI), [OpenMP](./OpenMP), [Test-MPI](./Test-MPI), and [Test-OpenMP](./Test-OpenMP). The "Test" directories are for validating the results of the matrix multiplication.

## Instructions
Submit a job script to OSC using the `sbatch` command in OSC:
```
sbatch jobScript.slurm
```

This job script runs the `makefile` with "make OSC" which builds and runs the code in `src`, with the output going to the `results.csv` file.

On OSC, the output result will be found in the `Default` folder under the filename `results.csv`.

## Results
Graphs and tables of the results featured in the research paper are located in [MPI/Results](./MPI/Results) and [OpenMP/Results](./OpenMP/Results) as a Microsoft Excel Worksheet file.