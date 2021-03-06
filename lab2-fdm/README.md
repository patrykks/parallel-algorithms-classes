# How to run

#### 1 Using own computer

```mpiexec -n N python fdm_membrane.py R C r c p_r p_c```

where: 

R - number of rows of input grid

C - number of columns of input grid

r - number of rows of input grid send to each process (process local grid)

c - number of columns of input grid send to each process (process local grid)

p_r - number of rows of processes grid

p_c - number of columns of processes grid

N - number of executors

### Caution! 

r * p_r must be equal to R

c * p_c must be equal to C

p_r * p_c must be equal to N


#### 2 Using supercomputer with SLURM

2.1. Configure script `grid_batch_script.sh`

1. Configure fdm program parameters

2. Set a limit on the total run time of the job allocation (-t flag)

3. Set number of nodes.

4. Set number of tasks per node.

### Caution! 

N * tasks-per-nodes must be equal to p_r * p_c

# How to plot

1. Save output of program into txt file.
2. Run script ```python plot.py {INPUT_TXT_FILENAME} {OUTPUT_PDF_FILENAME}```

# Examples

#### 1 Using own computer

1.1. Run ```mpiexec -n 4 python fdm_membrane.py 64 64 32 32 2 2 > local_result.txt```

1.2. Run ```python plot.py local_result.txt local_plot.pdf```

#### 2.Using supercomputer with SLURM

2.1. Run ```sbatch grid_batch_script.sh``` 
As a result You will get: Submitted batch job {JOB_ID}

2.2. Check job status using ```squeue```

2.3. When the result will be ready new file with name slurm-{JOB_ID} appears in current directory.


2.4. Move generated file onto localhost using ```scp GRID_ADDRESS:PATH_TO_GENERATED_FILE_ON_GRID ./``` 

2.5. Remove first for lines from file (logs generated by grid)

2.6. Run ```python plot.py slurm-{JOB-ID}.out remote_plot.pdf```