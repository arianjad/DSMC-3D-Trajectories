#!/usr/bin/env bash
# Source inside an allocated job; modules verified on Engaging 2026-09-24.
module load gcc/12.2.0 openmpi/4.1.4 cmake/3.27.9 miniforge/25.11.0-0 julia/1.10.4
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
