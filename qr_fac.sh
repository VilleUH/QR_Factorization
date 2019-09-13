#!/bin/bash -l
#SBATCH -A g2018006
#SBATCH -t 1:00
#SBATCH -p node
#SBATCH -N 1
#SBATCH --qos=short

module load gcc openmpi
mpirun -np 20 --map-by core ./qr_fac 5000 5000