#!/bin/bash

#SBATCH -p plgrid
#SBATCH -N 4
#SBATCH --ntasks-per-node 1
#SBATCH -t 02:00:00

module load plgrid/tools/python/2.7.9

mpiexec python fdm_membrane.py 64 64 32 32 2 2